# Battery-B harness (seeds 105-108). Per PREREG.md: trains the real CNN + 200 shuffled
# controls once per seed (Source/degradation_aware_final_sweep.py functions,
# unchanged), then for each condition builds a test set per depth with the
# wrapper generation path, evaluates real + all 200 controls on the SAME
# test set, BH-FDR per (seed, condition) within the 23-depth grid.
#
# Checkpointing: after each (seed, condition) is done, appends its floor row
# to results/floors_<seeds>.csv and its per-depth rows to
# results/depth_detail_<seed>.csv, and flushes -- so the caller can git-add
# and commit incrementally. After each seed's conditions 1-7 finish, runs
# the condition-8 pure-noise probe and appends to
# results/noise_probe_<seeds>.csv.

import os
import sys
import csv
import time
import logging
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(HERE, "..", "..", "..", "Source")
sys.path.insert(0, SOURCE_DIR)
sys.path.insert(0, HERE)

from degradation_aware_final_sweep import (  # noqa: E402
    build_degradation_aware_dataset, train_cnn, bh_fdr_correction,
    EVAL_DEPTHS_CM, FREQ, N_SAMPLES_PER_CELL,
)
from physics_model import stiffness_classes  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

import config as cfg  # noqa: E402
from wrappers import (  # noqa: E402
    generate_degraded_windows_shifted, build_pure_noise_windows, snr_db_to_std,
)

torch.set_num_threads(cfg.N_TORCH_THREADS)

RESULTS_DIR = os.path.join(HERE, "results")
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

SEED_TAG = "105-108"
FLOORS_CSV = os.path.join(RESULTS_DIR, f"floors_{SEED_TAG}.csv")
NOISE_PROBE_CSV = os.path.join(RESULTS_DIR, f"noise_probe_{SEED_TAG}.csv")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "battery_run.log")), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("battery")

CLASS_NAMES_SORTED = sorted(stiffness_classes.keys())
DEPTHS = list(EVAL_DEPTHS_CM)
N_PER_CLASS = N_SAMPLES_PER_CELL // len(CLASS_NAMES_SORTED)


def bh_adjusted_pvalues(p_values):
    """Same monotone step-up formula as audit_fixes/lane5/recall_floor_main_run/lane5_harness.py's
    bh_adjusted_pvalues (adjusted p-values complementary to the imported
    boolean-only bh_fdr_correction)."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    raw_adj = ranked * n / np.arange(1, n + 1)
    adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    adjusted = np.empty(n)
    adjusted[order] = adj_sorted
    return adjusted


def resolve_per_class_shift(shift, class_name):
    """Turns a possibly-per-class shift dict (from config.py) into a
    concrete shift dict for generate_degraded_windows_shifted for ONE
    class, resolving stiffness_cv_per_class / eta_override_per_class."""
    resolved = {k: v for k, v in shift.items()
                if k not in ("stiffness_cv_per_class", "eta_override_per_class")}
    if "stiffness_cv_per_class" in shift and class_name in shift["stiffness_cv_per_class"]:
        resolved["stiffness_cv"] = shift["stiffness_cv_per_class"][class_name]
    if "eta_override_per_class" in shift and class_name in shift["eta_override_per_class"]:
        resolved["eta_override"] = shift["eta_override_per_class"][class_name]
    return resolved


def build_eval_cell(depth_cm, shift, encoder, n_per_class=N_PER_CLASS, rng=None):
    depth_m = depth_cm / 100
    X_cell, y_cell_names = [], []
    for name in CLASS_NAMES_SORTED:
        kpa = stiffness_classes[name]
        class_shift = resolve_per_class_shift(shift, name)
        windows = generate_degraded_windows_shifted(
            kpa, depth_m, FREQ, n_per_class, shift=class_shift, rng=rng,
        )
        X_cell.extend(windows)
        y_cell_names.extend([name] * len(windows))
    X_cell = np.array(X_cell)
    y_cell_encoded = encoder.transform(y_cell_names)
    X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)
    return X_cell_t, y_cell_encoded


def confusion_matrix_3x3(preds, y_true, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, preds):
        cm[t, p] += 1
    return cm


def run_condition(seed, cond_id, cond_label, shift, model_real, control_models, encoder,
                   detail_writer, detail_f):
    real_acc_by_depth = []
    p_values = []
    control_stats_by_depth = []
    confmats = {}

    rng = np.random.RandomState((seed * 100003 + abs(hash(cond_id))) % (2**32 - 1))

    for depth_cm in DEPTHS:
        X_cell_t, y_cell_encoded = build_eval_cell(depth_cm, shift, encoder, rng=rng)

        with torch.no_grad():
            preds_real = model_real(X_cell_t).argmax(dim=1).numpy()
        acc_real = float(np.mean(preds_real == y_cell_encoded))
        real_acc_by_depth.append(acc_real)
        confmats[float(depth_cm)] = confusion_matrix_3x3(preds_real, y_cell_encoded)

        control_accs = []
        for model_c in control_models:
            with torch.no_grad():
                preds_c = model_c(X_cell_t).argmax(dim=1).numpy()
            control_accs.append(float(np.mean(preds_c == y_cell_encoded)))
        control_accs = np.array(control_accs)
        control_stats_by_depth.append({
            "min": float(control_accs.min()), "median": float(np.median(control_accs)),
            "max": float(control_accs.max()), "std": float(control_accs.std()),
        })

        p_value = (np.sum(control_accs >= acc_real) + 1) / (cfg.N_SHUFFLES + 1)
        p_values.append(p_value)

    sig_flags = bh_fdr_correction(p_values, alpha=cfg.FDR_ALPHA)
    adj_p = bh_adjusted_pvalues(p_values)

    floor_depth = None
    for depth_cm, sig in zip(DEPTHS, sig_flags):
        if not sig:
            floor_depth = depth_cm
            break
    censored = floor_depth is None
    if censored:
        floor_depth = DEPTHS[-1]

    last_sig = None
    for depth_cm, sig in zip(DEPTHS, sig_flags):
        if sig:
            last_sig = depth_cm

    for i, depth_cm in enumerate(DEPTHS):
        detail_writer.writerow({
            "seed": seed, "condition": cond_id, "depth_cm": depth_cm,
            "real_acc": real_acc_by_depth[i], "raw_p_value": p_values[i],
            "bh_adjusted_p_value": adj_p[i], "fdr_significant": bool(sig_flags[i]),
            "control_acc_min": control_stats_by_depth[i]["min"],
            "control_acc_median": control_stats_by_depth[i]["median"],
            "control_acc_max": control_stats_by_depth[i]["max"],
            "control_acc_std": control_stats_by_depth[i]["std"],
        })
    detail_f.flush()

    return {
        "seed": seed, "condition": cond_id, "condition_label": cond_label,
        "floor_cm": floor_depth, "last_significant_cm": last_sig, "censored": censored,
    }, confmats


def run_noise_probe(seed, model_real, control_models, encoder):
    from constants import SENSOR_SNR_DB
    n_total = N_SAMPLES_PER_CELL
    rng = np.random.RandomState((seed * 999983 + 8) % (2**32 - 1))
    X = build_pure_noise_windows(n_total, SENSOR_SNR_DB, rng=rng)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        preds_real = model_real(X_t).argmax(dim=1).numpy()
    n_classes = len(encoder.classes_)
    real_frac = np.bincount(preds_real, minlength=n_classes) / n_total

    control_fracs = []
    for model_c in control_models:
        with torch.no_grad():
            preds_c = model_c(X_t).argmax(dim=1).numpy()
        control_fracs.append(np.bincount(preds_c, minlength=n_classes) / n_total)
    control_mean_frac = np.mean(control_fracs, axis=0)

    rows = []
    for cls_idx, cls_name in enumerate(encoder.classes_):
        rows.append({"seed": seed, "source": "real", "class": cls_name, "fraction": float(real_frac[cls_idx])})
    for cls_idx, cls_name in enumerate(encoder.classes_):
        rows.append({"seed": seed, "source": "control_mean", "class": cls_name, "fraction": float(control_mean_frac[cls_idx])})
    return rows


def run_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    t0 = time.time()
    X_train, y_train_names = build_degradation_aware_dataset()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)
    model_real = train_cnn(X_train, y_train_encoded)
    train_elapsed = time.time() - t0
    log.info(f"seed={seed} real model trained in {train_elapsed:.0f}s")

    t1 = time.time()
    control_models = []
    for i in range(cfg.N_SHUFFLES):
        y_shuffled = np.random.permutation(y_train_encoded)
        control_models.append(train_cnn(X_train, y_shuffled))
        if (i + 1) % 50 == 0:
            log.info(f"seed={seed} controls trained {i+1}/{cfg.N_SHUFFLES}")
    controls_elapsed = time.time() - t1
    log.info(f"seed={seed} 200 controls trained in {controls_elapsed:.0f}s")

    floors_file_exists = os.path.exists(FLOORS_CSV)
    depth_detail_path = os.path.join(RESULTS_DIR, f"depth_detail_{seed}.csv")
    confmat_path = os.path.join(RESULTS_DIR, f"confmat_{seed}.npz")

    floors_fieldnames = ["seed", "condition", "condition_label", "floor_cm", "last_significant_cm", "censored"]
    detail_fieldnames = ["seed", "condition", "depth_cm", "real_acc", "raw_p_value", "bh_adjusted_p_value",
                          "fdr_significant", "control_acc_min", "control_acc_median", "control_acc_max", "control_acc_std"]

    all_confmats = {}
    with open(FLOORS_CSV, "a", newline="") as ff, open(depth_detail_path, "w", newline="") as df:
        floors_writer = csv.DictWriter(ff, fieldnames=floors_fieldnames)
        if not floors_file_exists:
            floors_writer.writeheader()
        detail_writer = csv.DictWriter(df, fieldnames=detail_fieldnames)
        detail_writer.writeheader()

        for cond_id, cond_label, shift in cfg.CONDITIONS:
            tc0 = time.time()
            row, confmats = run_condition(seed, cond_id, cond_label, shift, model_real,
                                           control_models, encoder, detail_writer, df)
            for depth_cm, cm in confmats.items():
                all_confmats[f"{cond_id}__depth{depth_cm:.2f}"] = cm
            floors_writer.writerow(row)
            ff.flush()
            log.info(f"seed={seed} condition={cond_id} floor={row['floor_cm']:.2f}cm "
                      f"censored={row['censored']} ({time.time()-tc0:.0f}s)")

    np.savez_compressed(confmat_path, **{k: v for k, v in all_confmats.items()})

    noise_rows = run_noise_probe(seed, model_real, control_models, encoder)
    noise_file_exists = os.path.exists(NOISE_PROBE_CSV)
    with open(NOISE_PROBE_CSV, "a", newline="") as nf:
        noise_writer = csv.DictWriter(nf, fieldnames=["seed", "source", "class", "fraction"])
        if not noise_file_exists:
            noise_writer.writeheader()
        for r in noise_rows:
            noise_writer.writerow(r)
    log.info(f"seed={seed} noise probe done")

    total_elapsed = time.time() - t0
    log.info(f"seed={seed} TOTAL {total_elapsed:.0f}s (train={train_elapsed:.0f}s controls={controls_elapsed:.0f}s)")
    return total_elapsed


def main():
    seeds = cfg.SEEDS_TO_RUN
    done_seeds = set()
    if os.path.exists(FLOORS_CSV):
        with open(FLOORS_CSV) as f:
            for row in csv.DictReader(f):
                # a seed counts done only once it has all conditions + noise probe;
                # simplistic resume: rerun any seed not fully represented.
                pass
    first_seed_time = None
    fail_counts = {}
    for seed in seeds:
        confmat_path = os.path.join(RESULTS_DIR, f"confmat_{seed}.npz")
        noise_done = False
        if os.path.exists(NOISE_PROBE_CSV):
            with open(NOISE_PROBE_CSV) as f:
                noise_done = any(int(r["seed"]) == seed for r in csv.DictReader(f))
        if os.path.exists(confmat_path) and noise_done:
            log.info(f"seed={seed} already complete, skipping")
            continue
        attempts = 0
        while attempts < 3:
            try:
                elapsed = run_one_seed(seed)
                if first_seed_time is None:
                    first_seed_time = elapsed
                elif elapsed > 3 * first_seed_time:
                    log.error(f"seed={seed} took {elapsed:.0f}s > 3x first seed's {first_seed_time:.0f}s -- STOP condition per rule 5")
                    with open(os.path.join(HERE, "STATUS.md"), "w") as sf:
                        sf.write(f"# STATUS: stopped -- seed {seed} took {elapsed:.0f}s, "
                                 f"> 3x first seed's {first_seed_time:.0f}s.\n")
                    sys.exit(2)
                break
            except Exception as e:
                attempts += 1
                log.exception(f"seed={seed} attempt {attempts} failed: {e}")
                if attempts >= 3:
                    with open(os.path.join(HERE, "STATUS.md"), "w") as sf:
                        sf.write(f"# STATUS: stopped -- seed {seed} failed 3 times.\nLast error: {e}\n")
                    sys.exit(3)
    log.info("ALL SEEDS COMPLETE")


if __name__ == "__main__":
    main()

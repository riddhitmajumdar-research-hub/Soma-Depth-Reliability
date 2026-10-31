# Lane 5 addendum -- CLEAN (non-degradation-aware) comparison sweep.
# See audit_fixes/lane5/clean_trained_sweep/PREREG_clean.md for the full protocol.
#
# Deliberately a SEPARATE script from lane5_harness.py, importing nothing
# from it, so it cannot interact with that module's already-running
# process (PREREG_clean.md section 2). All training/eval logic is
# imported unchanged from Source/.
#
# UPDATE (PREREG_clean.md addendum 3): also saves, per seed, the real
# model + all 200 control models' weights (already did this) AND now
# per-depth 3x3 confusion matrices for the real model and every control.

import os
import sys
import csv
import time
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Source"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # lane5_config.py lives one level up

from degradation_aware_final_sweep import (   # noqa: E402 -- unmodified imports
    train_cnn,
    bh_fdr_correction,
    generate_degraded_windows,
    EVAL_DEPTHS_CM,
    FREQ,
)
from cnn_classifier import build_raw_dataset   # noqa: E402 -- the "clean" dataset builder
from physics_model import stiffness_classes    # noqa: E402
from sklearn.preprocessing import LabelEncoder

import lane5_config as cfg

torch.set_num_threads(cfg.N_TORCH_THREADS)

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "clean_sweep")
LOG_DIR = os.path.join(OUT_DIR, "logs")
WEIGHTS_DIR = os.path.join(OUT_DIR, "seed_weights")
CONTROLS_DIR = os.path.join(OUT_DIR, "controls")
CONFMAT_DIR = os.path.join(OUT_DIR, "confusion_matrices")
CSV_PATH = os.path.join(LOG_DIR, "lane5_clean_per_class_results.csv")
LOG_PATH = os.path.join(LOG_DIR, "lane5_clean_run.log")

for d in (LOG_DIR, WEIGHTS_DIR, CONTROLS_DIR, CONFMAT_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lane5_clean")

CLASS_NAMES_SORTED = sorted(stiffness_classes.keys())
N_CLASSES = len(CLASS_NAMES_SORTED)


def bh_adjusted_pvalues(p_values):
    """Identical helper to lane5_harness.bh_adjusted_pvalues -- duplicated
    here (not imported from lane5_harness.py) to keep this script fully
    independent of that module per PREREG_clean.md section 2."""
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


def confusion_matrix_3x3(y_true, y_pred):
    """3x3 confusion matrix, rows=true, cols=pred, class order =
    CLASS_NAMES_SORTED. No sklearn dependency needed for a fixed 3-class
    problem; trivial to verify by eye. Added per PREREG_clean.md addendum 3."""
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def eval_model_per_class(model, X_cell_t, y_cell_encoded):
    with torch.no_grad():
        preds = model(X_cell_t).argmax(dim=1).numpy()
    per_class = {}
    for cls_idx, name in enumerate(CLASS_NAMES_SORTED):
        mask = y_cell_encoded == cls_idx
        per_class[name] = float(np.mean(preds[mask] == y_cell_encoded[mask])) if mask.sum() > 0 else float("nan")
    overall_acc = float(np.mean(preds == y_cell_encoded))
    cm = confusion_matrix_3x3(y_cell_encoded, preds)
    return per_class, overall_acc, cm


def run_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    t_train0 = time.time()
    # CLEAN training set: noiseless, depth-0, no attenuation -- exactly
    # Stage 3c / floor_characterization_cnn.py's "clean CNN" (PREREG_clean.md 0).
    X_train, y_train_names = build_raw_dataset(seed=seed)
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)
    model_real = train_cnn(X_train, y_train_encoded)
    torch.save(model_real.state_dict(), os.path.join(WEIGHTS_DIR, f"clean_seed{seed}.pt"))
    train_elapsed = time.time() - t_train0

    t_controls0 = time.time()
    seed_control_dir = os.path.join(WEIGHTS_DIR, f"controls_seed{seed}")
    os.makedirs(seed_control_dir, exist_ok=True)
    control_models = []
    for i in range(cfg.N_SHUFFLES):
        y_shuffled = np.random.permutation(y_train_encoded)
        model_c = train_cnn(X_train, y_shuffled)
        control_models.append(model_c)
        torch.save(model_c.state_dict(), os.path.join(seed_control_dir, f"control{i:03d}.pt"))
    controls_elapsed = time.time() - t_controls0

    # Evaluation IDENTICAL to the main run: 100Hz, 27.05dB (baked into
    # generate_degraded_windows via constants.SENSOR_SNR_DB), same 23-depth
    # grid, N_SAMPLES_PER_CELL=300.
    n_per_class = cfg.N_EVAL_SAMPLES_PER_CELL // len(CLASS_NAMES_SORTED)
    depths = list(EVAL_DEPTHS_CM)

    real_recall_by_depth = []
    control_recall_by_depth = []
    real_acc_by_depth = []
    control_acc_by_depth = []  # list of np.array[200], needed for aggregate + multiplicity_robustness.py

    t_eval0 = time.time()
    for depth_cm in depths:
        depth_m = depth_cm / 100
        X_cell, y_cell_names = [], []
        for name in CLASS_NAMES_SORTED:
            kpa = stiffness_classes[name]
            windows = generate_degraded_windows(kpa, depth_m, FREQ, n_per_class)
            X_cell.extend(windows)
            y_cell_names.extend([name] * len(windows))
        X_cell = np.array(X_cell)
        y_cell_encoded = encoder.transform(y_cell_names)
        X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)

        real_recall, real_acc, real_cm = eval_model_per_class(model_real, X_cell_t, y_cell_encoded)
        real_acc_by_depth.append(real_acc)
        real_recall_by_depth.append(real_recall)
        np.save(os.path.join(CONFMAT_DIR, f"seed{seed}_depth{depth_cm:.2f}_real_confmat.npy"), real_cm)

        per_class_control_recalls = {name: [] for name in CLASS_NAMES_SORTED}
        control_accs = []
        control_cms = np.zeros((cfg.N_SHUFFLES, N_CLASSES, N_CLASSES), dtype=np.int32)
        for ci, model_c in enumerate(control_models):
            c_recall, c_acc, c_cm = eval_model_per_class(model_c, X_cell_t, y_cell_encoded)
            control_accs.append(c_acc)
            control_cms[ci] = c_cm
            for name in CLASS_NAMES_SORTED:
                per_class_control_recalls[name].append(c_recall[name])
        np.save(os.path.join(CONFMAT_DIR, f"seed{seed}_depth{depth_cm:.2f}_controls_confmat.npy"), control_cms)
        control_acc_by_depth.append(np.array(control_accs, dtype=np.float32))
        control_recall_by_depth.append({k: np.array(v, dtype=np.float32) for k, v in per_class_control_recalls.items()})
    eval_elapsed = time.time() - t_eval0

    # Aggregate floor, BH within seed (matches main run's family definition).
    agg_p_values = [
        (np.sum(control_acc_by_depth[i] >= real_acc_by_depth[i]) + 1) / (cfg.N_SHUFFLES + 1)
        for i in range(len(depths))
    ]
    agg_sig = bh_fdr_correction(agg_p_values, alpha=cfg.FDR_ALPHA)
    agg_adj_p = bh_adjusted_pvalues(agg_p_values)
    agg_floor = None
    for depth_cm, sig in zip(depths, agg_sig):
        if not sig:
            agg_floor = depth_cm
            break
    if agg_floor is None:
        agg_floor = depths[-1]

    # Save raw arrays needed by multiplicity_robustness.py: per-depth
    # control ACCURACY (for Westfall-Young max-T and pooled BH) and
    # per-depth per-class control RECALL.
    for i, depth_cm in enumerate(depths):
        np.save(os.path.join(CONTROLS_DIR, f"seed{seed}_depth{depth_cm:.2f}_ACC.npy"), control_acc_by_depth[i])
        for name in CLASS_NAMES_SORTED:
            np.save(
                os.path.join(CONTROLS_DIR, f"seed{seed}_depth{depth_cm:.2f}_{name}.npy"),
                control_recall_by_depth[i][name],
            )

    depth_detail_path = os.path.join(LOG_DIR, f"lane5_clean_aggregate_depth_detail_seed{seed}.csv")
    with open(depth_detail_path, "w", newline="") as f_detail:
        w = csv.DictWriter(f_detail, fieldnames=["seed", "depth_cm", "acc_real", "raw_p_value", "bh_adjusted_p_value", "fdr_significant"])
        w.writeheader()
        for i, depth_cm in enumerate(depths):
            w.writerow({
                "seed": seed, "depth_cm": depth_cm, "acc_real": real_acc_by_depth[i],
                "raw_p_value": agg_p_values[i], "bh_adjusted_p_value": agg_adj_p[i],
                "fdr_significant": bool(agg_sig[i]),
            })

    # Per-class depth-detail checkpoint -- fixes forward the gap found in
    # the main run's harness (real per-class recall + per-depth per-class
    # p-values were computed but never persisted there, only the final
    # floor). Written here so multiplicity_robustness.py CAN do the
    # per-class analysis for the clean sweep, unlike the main run.
    per_class_p_vals = {name: [] for name in CLASS_NAMES_SORTED}
    for name in CLASS_NAMES_SORTED:
        for i in range(len(depths)):
            p = (np.sum(control_recall_by_depth[i][name] >= real_recall_by_depth[i][name]) + 1) / (cfg.N_SHUFFLES + 1)
            per_class_p_vals[name].append(p)

    per_class_detail_path = os.path.join(LOG_DIR, f"lane5_clean_per_class_depth_detail_seed{seed}.csv")
    with open(per_class_detail_path, "w", newline="") as f_pc:
        fieldnames_pc = ["seed", "depth_cm"] + [
            f"{name}_{suf}" for name in CLASS_NAMES_SORTED for suf in ("recall_real", "raw_p_value", "bh_adjusted_p_value", "fdr_significant")
        ]
        w_pc = csv.DictWriter(f_pc, fieldnames=fieldnames_pc)
        w_pc.writeheader()
        per_class_adj_p = {name: bh_adjusted_pvalues(per_class_p_vals[name]) for name in CLASS_NAMES_SORTED}
        per_class_sig = {name: bh_fdr_correction(per_class_p_vals[name], alpha=cfg.FDR_ALPHA) for name in CLASS_NAMES_SORTED}
        for i, depth_cm in enumerate(depths):
            row_pc = {"seed": seed, "depth_cm": depth_cm}
            for name in CLASS_NAMES_SORTED:
                row_pc[f"{name}_recall_real"] = real_recall_by_depth[i][name]
                row_pc[f"{name}_raw_p_value"] = per_class_p_vals[name][i]
                row_pc[f"{name}_bh_adjusted_p_value"] = per_class_adj_p[name][i]
                row_pc[f"{name}_fdr_significant"] = bool(per_class_sig[name][i])
            w_pc.writerow(row_pc)

    per_class_floor = {}
    for name in CLASS_NAMES_SORTED:
        p_vals = per_class_p_vals[name]
        sig_flags = bh_fdr_correction(p_vals, alpha=cfg.FDR_ALPHA)
        # Floor = last depth at which the class remained FDR-significant.
        # (Bug fix: previous version broke on the FIRST
        # non-significant depth, which returns 0.0cm whenever depth=0
        # itself fails significance after BH correction across the full
        # depth sweep -- reporting "never detectable" instead of "how
        # deep before it stops being detectable". Same convention as the
        # aggregate floor and the degradation-aware harness.)
        floor_d = None
        for depth_cm, sig in zip(depths, sig_flags):
            if sig:
                floor_d = depth_cm
        per_class_floor[name] = floor_d if floor_d is not None else 0.0

    return {
        "seed": seed,
        "agg_floor_cm": agg_floor,
        "per_class_floor_cirrhosis": per_class_floor.get("cirrhosis"),
        "per_class_floor_healthy": per_class_floor.get("healthy"),
        "per_class_floor_significant_fibrosis": per_class_floor.get("significant_fibrosis"),
        "train_seconds": train_elapsed,
        "controls_seconds": controls_elapsed,
        "eval_seconds": eval_elapsed,
        "total_seconds": train_elapsed + controls_elapsed + eval_elapsed,
    }


def main():
    file_exists = os.path.exists(CSV_PATH)
    already_done = set()
    if file_exists:
        with open(CSV_PATH, "r") as f:
            for row in csv.DictReader(f):
                already_done.add(int(row["seed"]))
        log.info(f"Resuming: found existing clean-sweep results for seeds {sorted(already_done)}")

    fieldnames = [
        "seed", "agg_floor_cm",
        "per_class_floor_cirrhosis", "per_class_floor_healthy", "per_class_floor_significant_fibrosis",
        "train_seconds", "controls_seconds", "eval_seconds", "total_seconds",
    ]
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        remaining = [s for s in cfg.SEEDS_TO_RUN if s not in already_done]
        n_total = len(cfg.SEEDS_TO_RUN)
        for idx, seed in enumerate(remaining):
            t0 = time.time()
            log.info(f"--- [clean] Seed {seed} starting ({len(already_done) + idx + 1}/{n_total}) ---")
            row = run_one_seed(seed)
            elapsed = time.time() - t0
            done_so_far = len(already_done) + idx + 1
            eta_seconds = elapsed * (n_total - done_so_far)
            log.info(
                f"[clean] Seed {seed} done in {elapsed:.0f}s. floor={row['agg_floor_cm']:.2f}cm. "
                f"ETA for remaining {n_total - done_so_far} seeds: {eta_seconds/60:.1f} min"
            )
            writer.writerow(row)
            f.flush()

    log.info(f"All done. Full data in {CSV_PATH}")


if __name__ == "__main__":
    main()

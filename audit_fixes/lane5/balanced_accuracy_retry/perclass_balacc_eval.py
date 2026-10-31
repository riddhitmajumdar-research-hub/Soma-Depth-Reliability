# Lane 5 -- per-class one-vs-rest BALANCED ACCURACY re-evaluation.
# See audit_fixes/lane5/balanced_accuracy_retry/PREREG_perclass.md for the full protocol.
#
# EVAL-ONLY: trains nothing. Loads the main run's already-saved weights
# (real model + 200 controls per seed) and re-evaluates them on a fresh,
# deterministic test set per (seed, depth). Does not import or interact
# with lane5_harness.py's running process -- only reads its output files.

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
    bh_fdr_correction,
    generate_degraded_windows,
    EVAL_DEPTHS_CM,
    FREQ,
)
from cnn_classifier import SimpleCNN           # noqa: E402
from physics_model import stiffness_classes    # noqa: E402
from sklearn.preprocessing import LabelEncoder

import lane5_config as cfg  # frozen while the main run is alive; read-only import

torch.set_num_threads(2)  # as specified in the protocol -- runs concurrently with other work

HERE = os.path.dirname(__file__)
MAIN_WEIGHTS_DIR = os.path.join(HERE, "..", "recall_floor_main_run", "seed_weights")           # main run's output (read-only)
OUT_DIR = os.path.join(HERE, "perclass_balacc")
LOG_DIR = os.path.join(OUT_DIR, "logs")
CONFMAT_DIR = os.path.join(OUT_DIR, "confusion_matrices")
CSV_PATH = os.path.join(LOG_DIR, "perclass_balacc_results.csv")
LOG_PATH = os.path.join(LOG_DIR, "perclass_balacc_run.log")

for d in (LOG_DIR, CONFMAT_DIR):
    os.makedirs(d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("perclass_balacc")

CLASS_NAMES_SORTED = sorted(stiffness_classes.keys())
N_CLASSES = len(CLASS_NAMES_SORTED)
DEPTHS = list(EVAL_DEPTHS_CM)
POLL_INTERVAL_SECONDS = 60


def confusion_matrix_3x3(y_true, y_pred):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_class_stats_from_confmat(cm):
    """Returns dict[class_name] -> {recall, specificity, balanced_acc,
    precision, f1}. See PREREG_perclass.md section 2 for the definitions
    and the constant-predictor=0.5 invariant (verified in
    verify_constant_predictor_is_0p5() below)."""
    total = cm.sum()
    out = {}
    for idx, name in enumerate(CLASS_NAMES_SORTED):
        tp = cm[idx, idx]
        fn = cm[idx, :].sum() - tp
        fp = cm[:, idx].sum() - tp
        tn = total - tp - fn - fp
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        balanced_acc = (recall + specificity) / 2 if not (np.isnan(recall) or np.isnan(specificity)) else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0 else float("nan"))
        out[name] = {"recall": float(recall), "specificity": float(specificity),
                     "balanced_acc": float(balanced_acc), "precision": float(precision), "f1": float(f1)}
    return out


def verify_constant_predictor_is_0p5():
    """PREREG_perclass.md section 2 claims a constant predictor scores
    balanced_acc=0.5 for every class BY CONSTRUCTION. Verify, don't just
    assert -- run before trusting this script's output."""
    n = 90
    y_true = np.array([0] * 30 + [1] * 30 + [2] * 30)
    for const_class in range(N_CLASSES):
        y_pred = np.full(n, const_class)
        cm = confusion_matrix_3x3(y_true, y_pred)
        stats = per_class_stats_from_confmat(cm)
        for name, s in stats.items():
            assert abs(s["balanced_acc"] - 0.5) < 1e-9, (
                f"INVARIANT VIOLATED: constant predictor (class {const_class}) gave "
                f"balanced_acc={s['balanced_acc']} for class {name}, expected 0.5"
            )
    log.info("Verified: constant predictor -> balanced_acc=0.5 for every class (PREREG_perclass.md section 2).")


def seed_weights_complete(seed):
    real_path = os.path.join(MAIN_WEIGHTS_DIR, f"degradation_aware_seed{seed}.pt")
    controls_dir = os.path.join(MAIN_WEIGHTS_DIR, f"controls_seed{seed}")
    if not os.path.exists(real_path) or not os.path.isdir(controls_dir):
        return False
    n_controls = len([f for f in os.listdir(controls_dir) if f.startswith("control") and f.endswith(".pt")])
    return n_controls == cfg.N_SHUFFLES


def load_model(path):
    m = SimpleCNN(n_classes=N_CLASSES)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m


def bh_adjusted_pvalues(p_values):
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


def run_one_seed(seed):
    real_path = os.path.join(MAIN_WEIGHTS_DIR, f"degradation_aware_seed{seed}.pt")
    controls_dir = os.path.join(MAIN_WEIGHTS_DIR, f"controls_seed{seed}")
    model_real = load_model(real_path)
    control_models = [load_model(os.path.join(controls_dir, f"control{i:03d}.pt")) for i in range(cfg.N_SHUFFLES)]

    class_kpa_names = CLASS_NAMES_SORTED
    n_per_class = cfg.N_EVAL_SAMPLES_PER_CELL // len(class_kpa_names)
    encoder = LabelEncoder()
    encoder.fit(class_kpa_names)

    real_balacc_by_depth = {name: [] for name in CLASS_NAMES_SORTED}
    real_recall_by_depth = {name: [] for name in CLASS_NAMES_SORTED}
    real_f1_by_depth = {name: [] for name in CLASS_NAMES_SORTED}
    control_balacc_by_depth = {name: [] for name in CLASS_NAMES_SORTED}  # list of np.array[200] per depth
    raw_p_by_depth = {name: [] for name in CLASS_NAMES_SORTED}

    for depth_idx, depth_cm in enumerate(DEPTHS):
        # Fixed, deterministic test set per (seed, depth) -- PREREG_perclass.md section 1.
        np.random.seed(900000 + seed * 1000 + depth_idx)
        depth_m = depth_cm / 100
        X_cell, y_cell_names = [], []
        for name in class_kpa_names:
            kpa = stiffness_classes[name]
            windows = generate_degraded_windows(kpa, depth_m, FREQ, n_per_class)
            X_cell.extend(windows)
            y_cell_names.extend([name] * len(windows))
        X_cell = np.array(X_cell)
        y_cell_encoded = encoder.transform(y_cell_names)
        X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            preds_real = model_real(X_cell_t).argmax(dim=1).numpy()
        cm_real = confusion_matrix_3x3(y_cell_encoded, preds_real)
        stats_real = per_class_stats_from_confmat(cm_real)
        np.save(os.path.join(CONFMAT_DIR, f"seed{seed}_depth{depth_cm:.2f}_real_confmat.npy"), cm_real)

        cms_controls = np.zeros((cfg.N_SHUFFLES, N_CLASSES, N_CLASSES), dtype=np.int32)
        control_stats_per_class = {name: [] for name in CLASS_NAMES_SORTED}
        for ci, model_c in enumerate(control_models):
            with torch.no_grad():
                preds_c = model_c(X_cell_t).argmax(dim=1).numpy()
            cm_c = confusion_matrix_3x3(y_cell_encoded, preds_c)
            cms_controls[ci] = cm_c
            stats_c = per_class_stats_from_confmat(cm_c)
            for name in CLASS_NAMES_SORTED:
                control_stats_per_class[name].append(stats_c[name]["balanced_acc"])
        np.save(os.path.join(CONFMAT_DIR, f"seed{seed}_depth{depth_cm:.2f}_controls_confmat.npy"), cms_controls)

        for name in CLASS_NAMES_SORTED:
            real_balacc_by_depth[name].append(stats_real[name]["balanced_acc"])
            real_recall_by_depth[name].append(stats_real[name]["recall"])
            real_f1_by_depth[name].append(stats_real[name]["f1"])
            ctrl_arr = np.array(control_stats_per_class[name], dtype=np.float32)
            control_balacc_by_depth[name].append(ctrl_arr)
            p = (np.sum(ctrl_arr >= stats_real[name]["balanced_acc"]) + 1) / (cfg.N_SHUFFLES + 1)
            raw_p_by_depth[name].append(p)

    # BH within seed, per class (PREREG_perclass.md section 3).
    per_class_floor = {}
    per_class_first_below_045 = {}
    detail_rows = []
    for name in CLASS_NAMES_SORTED:
        p_vals = raw_p_by_depth[name]
        sig_flags = bh_fdr_correction(p_vals, alpha=cfg.FDR_ALPHA)
        adj_p = bh_adjusted_pvalues(p_vals)
        floor_d = None
        for depth_cm, sig in zip(DEPTHS, sig_flags):
            if not sig:
                floor_d = depth_cm
                break
        per_class_floor[name] = floor_d if floor_d is not None else DEPTHS[-1]

        below = None
        for i, depth_cm in enumerate(DEPTHS):
            if real_recall_by_depth[name][i] < cfg.PER_CLASS_RECALL_FLOOR_THRESHOLD:
                below = depth_cm
                break
        per_class_first_below_045[name] = below

        for i, depth_cm in enumerate(DEPTHS):
            detail_rows.append({
                "seed": seed, "depth_cm": depth_cm, "class": name,
                "recall_real": real_recall_by_depth[name][i],
                "balanced_acc_real": real_balacc_by_depth[name][i],
                "f1_real": real_f1_by_depth[name][i],
                "raw_p_value": p_vals[i],
                "bh_adjusted_p_value": adj_p[i],
                "fdr_significant": bool(sig_flags[i]),
            })

    detail_path = os.path.join(LOG_DIR, f"perclass_balacc_depth_detail_seed{seed}.csv")
    with open(detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "depth_cm", "class", "recall_real", "balanced_acc_real",
                                           "f1_real", "raw_p_value", "bh_adjusted_p_value", "fdr_significant"])
        w.writeheader()
        w.writerows(detail_rows)

    return {
        "seed": seed,
        "balacc_floor_cirrhosis": per_class_floor.get("cirrhosis"),
        "balacc_floor_healthy": per_class_floor.get("healthy"),
        "balacc_floor_significant_fibrosis": per_class_floor.get("significant_fibrosis"),
        "recall_below_0.45_cirrhosis": per_class_first_below_045.get("cirrhosis"),
        "recall_below_0.45_healthy": per_class_first_below_045.get("healthy"),
        "recall_below_0.45_significant_fibrosis": per_class_first_below_045.get("significant_fibrosis"),
    }


def main():
    verify_constant_predictor_is_0p5()

    file_exists = os.path.exists(CSV_PATH)
    already_done = set()
    if file_exists:
        with open(CSV_PATH, "r") as f:
            for row in csv.DictReader(f):
                already_done.add(int(row["seed"]))
        log.info(f"Resuming: found existing balacc results for seeds {sorted(already_done)}")

    fieldnames = [
        "seed", "balacc_floor_cirrhosis", "balacc_floor_healthy", "balacc_floor_significant_fibrosis",
        "recall_below_0.45_cirrhosis", "recall_below_0.45_healthy", "recall_below_0.45_significant_fibrosis",
    ]

    log.info("Polling for seeds with complete saved weights (main run's seed_weights/, read-only)...")
    while True:
        candidate_seeds = [s for s in cfg.SEEDS_TO_RUN if s not in already_done and seed_weights_complete(s)]
        if not candidate_seeds:
            all_expected_done = already_done.issuperset(set(cfg.SEEDS_TO_RUN))
            if all_expected_done:
                log.info("All 24 seeds processed. Done.")
                break
            log.info(f"No new seeds with complete weights yet ({len(already_done)}/{len(cfg.SEEDS_TO_RUN)} done). "
                      f"Sleeping {POLL_INTERVAL_SECONDS}s...")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
                file_exists = True
            for seed in candidate_seeds:
                t0 = time.time()
                log.info(f"--- Evaluating seed {seed} ({len(already_done) + 1}/{len(cfg.SEEDS_TO_RUN)}) ---")
                row = run_one_seed(seed)
                elapsed = time.time() - t0
                log.info(f"Seed {seed} done in {elapsed:.0f}s. balacc floors: "
                         f"cirrhosis={row['balacc_floor_cirrhosis']}, healthy={row['balacc_floor_healthy']}, "
                         f"significant_fibrosis={row['balacc_floor_significant_fibrosis']}")
                writer.writerow(row)
                f.flush()
                already_done.add(seed)


if __name__ == "__main__":
    main()

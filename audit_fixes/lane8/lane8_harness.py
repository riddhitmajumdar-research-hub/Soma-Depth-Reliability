# Lane 8 -- pre-registered Random Forest depth-floor certification harness.
# See audit_fixes/lane8/PREREG.md for the full protocol and disclosed
# judgment calls. This script BUILDS the harness; main() (the full 24-seed
# run) is not invoked by anything else in this step -- only
# lane8_benchmark.py's single-seed call is (see PREREG.md section 9).
#
# Imports training/eval-data/FDR logic UNCHANGED from
# Source/degradation_aware_final_sweep.py -- does not reimplement it.
# Only train_cnn is replaced, with train_rf (same "train, return a model
# with .predict()" interface). Read-only w.r.t. everything under Source/.

import os
import sys
import csv
import time
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Source"))

from degradation_aware_final_sweep import (   # noqa: E402 -- unmodified imports
    build_degradation_aware_dataset,
    bh_fdr_correction,
    generate_degraded_windows,
    EVAL_DEPTHS_CM,
    FREQ,
)
from physics_model import stiffness_classes   # noqa: E402
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

import lane8_config as cfg

HERE = os.path.dirname(__file__)
LOG_DIR = os.path.join(HERE, "logs")
CSV_PATH = os.path.join(LOG_DIR, "lane8_results.csv")
LOG_PATH = os.path.join(LOG_DIR, "lane8_run.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lane8")

CLASS_NAMES_SORTED = sorted(stiffness_classes.keys())


def train_rf(X, y_encoded, random_state):
    """New (not in original): RandomForest in place of train_cnn. Trains
    directly on the raw (n_samples, 100) window array -- no channel-dim
    reshape, since that reshape in the original is CNN-specific
    (nn.Conv1d input format), not part of "raw 100-sample window as
    features." See PREREG.md section 2-3 for the random_state rationale."""
    model = RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS,
        n_jobs=cfg.RF_N_JOBS,
        random_state=random_state,
    )
    model.fit(X, y_encoded)
    return model


def run_one_seed(seed, n_eval_samples_per_cell=300):
    np.random.seed(seed)

    t_train0 = time.time()
    X_train, y_train_names = build_degradation_aware_dataset()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)
    model_real = train_rf(X_train, y_train_encoded, random_state=seed)
    train_elapsed = time.time() - t_train0

    t_controls0 = time.time()
    control_models = []
    for i in range(cfg.N_SHUFFLES):
        y_shuffled = np.random.permutation(y_train_encoded)
        # See PREREG.md section 3: seed*10_000+i is a deterministic,
        # collision-free random_state for this seed x control grid.
        model_c = train_rf(X_train, y_shuffled, random_state=seed * 10_000 + i)
        control_models.append(model_c)
    controls_elapsed = time.time() - t_controls0

    n_per_class = n_eval_samples_per_cell // len(CLASS_NAMES_SORTED)
    depths = list(EVAL_DEPTHS_CM)

    t_eval0 = time.time()
    real_acc_by_depth = []
    control_accs_by_depth = []  # list of np.array[200]
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

        preds_real = model_real.predict(X_cell)
        acc_real = float(np.mean(preds_real == y_cell_encoded))
        real_acc_by_depth.append(acc_real)

        control_accs = np.array([
            np.mean(model_c.predict(X_cell) == y_cell_encoded) for model_c in control_models
        ], dtype=np.float64)
        control_accs_by_depth.append(control_accs)
    eval_elapsed = time.time() - t_eval0

    p_values = [
        (np.sum(control_accs_by_depth[i] >= real_acc_by_depth[i]) + 1) / (cfg.N_SHUFFLES + 1)
        for i in range(len(depths))
    ]
    sig_flags = bh_fdr_correction(p_values, alpha=cfg.FDR_ALPHA)

    floor_cm = None
    for depth_cm, sig in zip(depths, sig_flags):
        if not sig:
            floor_cm = depth_cm
            break
    if floor_cm is None:
        floor_cm = depths[-1]

    last_significant_depth = None
    for depth_cm, sig in zip(depths, sig_flags):
        if sig:
            last_significant_depth = depth_cm
    anomalous_recovery = any(
        sig and depth_cm > floor_cm for depth_cm, sig in zip(depths, sig_flags)
    )

    # Per the protocol: save real-model and control accuracies per
    # depth so p-values can be recomputed later without retraining.
    detail_path = os.path.join(LOG_DIR, f"lane8_depth_detail_seed{seed}.csv")
    with open(detail_path, "w", newline="") as f_detail:
        fieldnames = ["seed", "depth_cm", "real_acc", "p_value", "fdr_significant"] + [
            f"control_acc_{i:03d}" for i in range(cfg.N_SHUFFLES)
        ]
        writer = csv.DictWriter(f_detail, fieldnames=fieldnames)
        writer.writeheader()
        for i, depth_cm in enumerate(depths):
            row = {
                "seed": seed, "depth_cm": depth_cm,
                "real_acc": real_acc_by_depth[i],
                "p_value": p_values[i],
                "fdr_significant": bool(sig_flags[i]),
            }
            for j, c_acc in enumerate(control_accs_by_depth[i]):
                row[f"control_acc_{j:03d}"] = float(c_acc)
            writer.writerow(row)
    log.info(f"Seed {seed}: per-depth accuracy detail checkpointed to {detail_path}")

    return {
        "seed": seed,
        "naive_floor_cm_rf": floor_cm,
        "last_significant_depth_cm_rf": last_significant_depth,
        "anomalous_recovery_rf": anomalous_recovery,
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
        log.info(f"Resuming: found existing results for seeds {sorted(already_done)}")

    fieldnames = [
        "seed", "naive_floor_cm_rf", "last_significant_depth_cm_rf", "anomalous_recovery_rf",
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
            log.info(f"--- Seed {seed} starting ({len(already_done) + idx + 1}/{n_total}) ---")
            row = run_one_seed(seed)
            elapsed = time.time() - t0
            done_so_far = len(already_done) + idx + 1
            eta_seconds = elapsed * (n_total - done_so_far)
            log.info(
                f"Seed {seed} done in {elapsed:.0f}s. floor={row['naive_floor_cm_rf']:.2f}cm. "
                f"ETA for remaining {n_total - done_so_far} seeds: {eta_seconds/60:.1f} min"
            )
            writer.writerow(row)
            f.flush()

    log.info(f"All done. Full data in {CSV_PATH}")


if __name__ == "__main__":
    main()

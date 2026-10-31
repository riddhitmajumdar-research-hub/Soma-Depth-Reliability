"""
LANE 4 AMENDMENT v2, Step 1 -- diagnostics only, NO tuning.

Reruns the exact v1 single grouped-5-fold split (rep_seed=1000, identical to
run_lane4_comparison.py) and, for each fold and model, records:
  - train accuracy (on the outer TRAIN fold, post inner-CV model selection)
  - predicted-class histogram on the outer TEST fold
  - TEST-fold class counts (ground truth)
  - accuracy of a constant predictor that always predicts the TRAIN-fold
    majority class

Does not change any hyperparameter, grid, feature set, or protocol from v1.
Imports v1's functions directly (does not copy/reimplement them) so the
models trained here are byte-identical in construction to v1's.

READ-ONLY on v1's own output files (lane4_single_split.csv,
lane4_repeated_grouped_cv_raw.csv, run_stdout.log) -- does not modify or
delete them. Writes only diagnose_v1_output.csv and diagnose_v1_stdout.log
next to this script.

Run from repo root:
    python3 audit_fixes/lane4/diagnose_v1.py
"""

import os
import sys
import csv
from collections import Counter

import numpy as np
import torch

LANE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LANE_DIR)

import run_lane4_comparison as v1  # noqa: E402  (reuses v1's functions, not reimplemented)

from sklearn.preprocessing import StandardScaler, LabelEncoder  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

torch.set_num_threads(2)

LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


def majority_class_acc(y_train, y_test):
    majority = Counter(y_train).most_common(1)[0][0]
    return float(np.mean(y_test == majority)), int(majority)


def hist_str(y_pred, class_names):
    counts = Counter(y_pred)
    return "; ".join(f"{class_names[k]}={counts.get(k, 0)}" for k in sorted(class_names))


def main():
    X_raw, y_raw, groups_raw = v1.build_raw_dataset_with_groups()
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_raw)
    X_fft = v1.fft_magnitude_features(X_raw)
    class_names = {i: c for i, c in enumerate(encoder.classes_)}

    log("=" * 78)
    log("LANE 4 v2 STEP 1 -- v1 diagnostics (single grouped 5-fold split, rep_seed=1000)")
    log("=" * 78)
    log(f"Classes: {class_names}")
    log("")
    log("Inner-CV grouping check: v1's svm_inner_cv_select / rf_inner_cv_select / "
        "gru_inner_cv_select (run_lane4_comparison.py) all call "
        "StratifiedKFold(...).split(X_train, y_train) with no `groups` argument "
        "-- inner CV is stratified by label only, NOT grouped by group_id. "
        "This can put windows from the same realization on both sides of an "
        "inner train/validation split during hyperparameter selection. The "
        "OUTER 5-fold split itself is correctly grouped (via grouped_kfold_indices).")
    log("")

    rows = []
    for fold_num, (train_idx, test_idx) in enumerate(
            v1.grouped_kfold_indices(groups_raw, v1.N_FOLDS, rep_seed=1000), start=1):
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]
        maj_acc, maj_class = majority_class_acc(y_train, y_test)
        train_hist = hist_str(y_train, class_names)
        test_counts = hist_str(y_test, class_names)

        log(f"\n--- Fold {fold_num} ---")
        log(f"  train class hist: {train_hist}")
        log(f"  test  class counts (ground truth): {test_counts}")
        log(f"  train-majority class: {class_names[maj_class]}  "
            f"-> constant-predictor test accuracy = {maj_acc:.4f}")

        # ---- SVM ----
        svm_cfg = v1.svm_inner_cv_select(X_fft[train_idx], y_train, rep_seed=1000, fold_num=fold_num)
        scaler = StandardScaler()
        Xtr, Xte = scaler.fit_transform(X_fft[train_idx]), scaler.transform(X_fft[test_idx])
        svm = SVC(kernel="rbf", **svm_cfg)
        svm.fit(Xtr, y_train)
        svm_train_acc = accuracy_score(y_train, svm.predict(Xtr))
        svm_test_pred = svm.predict(Xte)
        svm_test_acc = accuracy_score(y_test, svm_test_pred)
        svm_pred_hist = hist_str(svm_test_pred, class_names)
        log(f"  SVM   cfg={svm_cfg}  train_acc={svm_train_acc:.4f}  test_acc={svm_test_acc:.4f}")
        log(f"        test predicted-class histogram: {svm_pred_hist}")
        rows.append({"fold": fold_num, "model": "svm", "train_acc": svm_train_acc,
                      "test_acc": svm_test_acc, "test_pred_hist": svm_pred_hist,
                      "test_class_counts": test_counts, "train_majority_class": class_names[maj_class],
                      "majority_baseline_acc": maj_acc, "selected_cfg": str(svm_cfg)})

        # ---- Random forest ----
        rf_cfg = v1.rf_inner_cv_select(X_fft[train_idx], y_train, rep_seed=1000, fold_num=fold_num)
        rf = RandomForestClassifier(n_estimators=300, random_state=v1.SEED, n_jobs=v1.N_JOBS, **rf_cfg)
        rf.fit(Xtr, y_train)
        rf_train_acc = accuracy_score(y_train, rf.predict(Xtr))
        rf_test_pred = rf.predict(Xte)
        rf_test_acc = accuracy_score(y_test, rf_test_pred)
        rf_pred_hist = hist_str(rf_test_pred, class_names)
        log(f"  RF    cfg={rf_cfg}  train_acc={rf_train_acc:.4f}  test_acc={rf_test_acc:.4f}")
        log(f"        test predicted-class histogram: {rf_pred_hist}")
        rows.append({"fold": fold_num, "model": "rf", "train_acc": rf_train_acc,
                      "test_acc": rf_test_acc, "test_pred_hist": rf_pred_hist,
                      "test_class_counts": test_counts, "train_majority_class": class_names[maj_class],
                      "majority_baseline_acc": maj_acc, "selected_cfg": str(rf_cfg)})

        # ---- GRU ----
        gru_cfg = v1.gru_inner_cv_select(X_raw[train_idx], y_train, rep_seed=1000, fold_num=fold_num)
        X_train_t = torch.tensor(X_raw[train_idx], dtype=torch.float32).unsqueeze(-1)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_test_t = torch.tensor(X_raw[test_idx], dtype=torch.float32).unsqueeze(-1)
        gru = v1.train_gru(X_train_t, y_train_t, gru_cfg["lr"], gru_cfg["dropout"])
        with torch.no_grad():
            gru_train_pred = gru(X_train_t).argmax(dim=1).numpy()
            gru_test_pred = gru(X_test_t).argmax(dim=1).numpy()
        gru_train_acc = accuracy_score(y_train, gru_train_pred)
        gru_test_acc = accuracy_score(y_test, gru_test_pred)
        gru_pred_hist = hist_str(gru_test_pred, class_names)
        log(f"  GRU   cfg={gru_cfg}  train_acc={gru_train_acc:.4f}  test_acc={gru_test_acc:.4f}")
        log(f"        test predicted-class histogram: {gru_pred_hist}")
        rows.append({"fold": fold_num, "model": "gru", "train_acc": gru_train_acc,
                      "test_acc": gru_test_acc, "test_pred_hist": gru_pred_hist,
                      "test_class_counts": test_counts, "train_majority_class": class_names[maj_class],
                      "majority_baseline_acc": maj_acc, "selected_cfg": str(gru_cfg)})

    with open(os.path.join(LANE_DIR, "diagnose_v1_output.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(os.path.join(LANE_DIR, "diagnose_v1_stdout.log"), "w") as f:
        f.write("\n".join(LINES) + "\n")

    log("\nSaved audit_fixes/lane4/diagnose_v1_output.csv and diagnose_v1_stdout.log")


if __name__ == "__main__":
    main()

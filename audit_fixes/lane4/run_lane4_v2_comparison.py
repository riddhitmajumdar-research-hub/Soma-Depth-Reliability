"""
LANE 4 v2 -- amendment run, per audit_fixes/lane4/PREREG_v2.md.

Does NOT modify or delete v1 outputs (lane4_single_split.csv,
lane4_repeated_grouped_cv_raw.csv, run_stdout.log, REPORT.md) -- writes only
new v2-suffixed files.

Changes from v1 (see PREREG_v2.md for full rationale):
  (a) SVM/RF features: raw 100-sample window concatenated with real+imag
      parts of the full rfft (100 + 102 = 202 features), replacing v1's
      FFT-magnitude-only features.
  (b) GRU: per-window standardization, gradient clipping (max_norm=5.0), and
      a train-accuracy collapse check with restart (up to max_restarts=5),
      mirroring the existing ResNet1D restart safeguard in
      Source/repeated_grouped_cv_log11.py.
  (c) Inner-CV hyperparameter selection for ALL THREE models switched from
      StratifiedKFold to StratifiedGroupKFold(n_splits=3), keyed on the
      TRAIN fold's own group_id values.

Everything else (outer split, seeds, grid values, n_jobs=2,
torch.set_num_threads(2), CSV column layout) is unchanged from v1.

Run from repo root:
    python3 audit_fixes/lane4/run_lane4_v2_comparison.py
"""

import os
import sys
import csv
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

LANE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LANE_DIR)

import run_lane4_comparison as v1  # noqa: E402 (reuses v1's dataset builders, split generator, grids)

from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

torch.set_num_threads(2)
N_JOBS = 2
SEED = v1.SEED

N_FOLDS = v1.N_FOLDS
N_REPETITIONS = v1.N_REPETITIONS
INNER_FOLDS = v1.INNER_FOLDS
SVM_GRID = v1.SVM_GRID
RF_GRID = v1.RF_GRID
GRU_GRID = v1.GRU_GRID
GRU_EPOCHS = v1.GRU_EPOCHS
GRAD_CLIP_NORM = 5.0
MAX_RESTARTS = 5


# ---------------------------------------------------------------------------
# v2 feature set: raw window + real/imag FFT (preserves phase / time shift)
# ---------------------------------------------------------------------------
def raw_plus_complex_fft_features(X_raw):
    spec = np.fft.rfft(X_raw, axis=1)  # (n, 51) complex
    feats = np.concatenate([X_raw, spec.real, spec.imag], axis=1)
    return feats


# ---------------------------------------------------------------------------
# GRU v2: per-window standardization, grad clipping, collapse-check restart
# ---------------------------------------------------------------------------
def standardize_windows(X_raw):
    mean = X_raw.mean(axis=1, keepdims=True)
    std = X_raw.std(axis=1, keepdims=True)
    return (X_raw - mean) / (std + 1e-8)


def majority_baseline_acc(y):
    majority = Counter(y).most_common(1)[0][0]
    return float(np.mean(y == majority))


def train_gru_v2(X_train_t, y_train_t, lr, dropout, n_epochs=GRU_EPOCHS):
    model = v1.SmallGRU(dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = crit(model(X_train_t), y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        opt.step()
    model.eval()
    return model


def train_gru_with_collapse_check(X_train_std, y_train_enc, lr, dropout):
    """Trains, and restarts from a fresh init if train accuracy does not
    beat the train-fold majority-class constant predictor, up to
    MAX_RESTARTS. Returns (model, n_restarts_used, train_acc)."""
    X_train_t = torch.tensor(X_train_std, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
    maj_acc = majority_baseline_acc(y_train_enc)

    n_restarts = 0
    model, train_acc = None, -1.0
    for attempt in range(MAX_RESTARTS):
        model = train_gru_v2(X_train_t, y_train_t, lr, dropout)
        with torch.no_grad():
            train_pred = model(X_train_t).argmax(dim=1).numpy()
        train_acc = accuracy_score(y_train_enc, train_pred)
        if train_acc > maj_acc:
            break
        n_restarts += 1
    return model, n_restarts, train_acc


def gru_inner_cv_select_v2(X_train_raw, y_train_enc, groups_train, rep_seed, fold_num):
    inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                  random_state=rep_seed * 1000 + fold_num)
    best_cfg, best_score = None, -1.0
    for cfg in GRU_GRID:
        scores = []
        for inner_tr, inner_vl in inner.split(X_train_raw, y_train_enc, groups_train):
            Xt = standardize_windows(X_train_raw[inner_tr])
            Xv = standardize_windows(X_train_raw[inner_vl])
            model, _, _ = train_gru_with_collapse_check(Xt, y_train_enc[inner_tr], cfg["lr"], cfg["dropout"])
            Xv_t = torch.tensor(Xv, dtype=torch.float32).unsqueeze(-1)
            yv_t = torch.tensor(y_train_enc[inner_vl], dtype=torch.long)
            with torch.no_grad():
                acc = (model(Xv_t).argmax(dim=1) == yv_t).float().mean().item()
            scores.append(acc)
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_cfg = mean_score, cfg
    return best_cfg


def svm_inner_cv_select_v2(X_train, y_train, groups_train, rep_seed, fold_num):
    inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                  random_state=rep_seed * 1000 + fold_num)
    best_cfg, best_score = None, -1.0
    for cfg in SVM_GRID:
        scores = []
        for inner_tr, inner_vl in inner.split(X_train, y_train, groups_train):
            scaler = StandardScaler()
            Xt = scaler.fit_transform(X_train[inner_tr])
            Xv = scaler.transform(X_train[inner_vl])
            model = SVC(kernel="rbf", **cfg)
            model.fit(Xt, y_train[inner_tr])
            scores.append(accuracy_score(y_train[inner_vl], model.predict(Xv)))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_cfg = mean_score, cfg
    return best_cfg


def rf_inner_cv_select_v2(X_train, y_train, groups_train, rep_seed, fold_num):
    inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True,
                                  random_state=rep_seed * 1000 + fold_num)
    best_cfg, best_score = None, -1.0
    for cfg in RF_GRID:
        scores = []
        for inner_tr, inner_vl in inner.split(X_train, y_train, groups_train):
            scaler = StandardScaler()
            Xt = scaler.fit_transform(X_train[inner_tr])
            Xv = scaler.transform(X_train[inner_vl])
            model = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                            n_jobs=N_JOBS, **cfg)
            model.fit(Xt, y_train[inner_tr])
            scores.append(accuracy_score(y_train[inner_vl], model.predict(Xv)))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_cfg = mean_score, cfg
    return best_cfg


def run_one_fold(X_raw, X_v2feat, y_enc, groups_all, train_idx, test_idx, rep_seed, fold_num, log):
    groups_train = groups_all[train_idx]

    # --- SVM on raw+real/imag-FFT features ---
    svm_cfg = svm_inner_cv_select_v2(X_v2feat[train_idx], y_enc[train_idx], groups_train, rep_seed, fold_num)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_v2feat[train_idx])
    Xte = scaler.transform(X_v2feat[test_idx])
    svm = SVC(kernel="rbf", **svm_cfg)
    svm.fit(Xtr, y_enc[train_idx])
    svm_acc = accuracy_score(y_enc[test_idx], svm.predict(Xte))

    # --- RF on same features ---
    rf_cfg = rf_inner_cv_select_v2(X_v2feat[train_idx], y_enc[train_idx], groups_train, rep_seed, fold_num)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=N_JOBS, **rf_cfg)
    rf.fit(Xtr, y_enc[train_idx])
    rf_acc = accuracy_score(y_enc[test_idx], rf.predict(Xte))

    # --- GRU: standardized windows, grad clip, collapse-check restart ---
    gru_cfg = gru_inner_cv_select_v2(X_raw[train_idx], y_enc[train_idx], groups_train, rep_seed, fold_num)
    X_train_std = standardize_windows(X_raw[train_idx])
    X_test_std = standardize_windows(X_raw[test_idx])
    gru, n_restarts, train_acc = train_gru_with_collapse_check(
        X_train_std, y_enc[train_idx], gru_cfg["lr"], gru_cfg["dropout"])
    X_test_t = torch.tensor(X_test_std, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_enc[test_idx], dtype=torch.long)
    with torch.no_grad():
        gru_acc = (gru(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()

    log(f"    svm cfg={svm_cfg} acc={svm_acc:.4f} | rf cfg={rf_cfg} acc={rf_acc:.4f} | "
        f"gru cfg={gru_cfg} restarts={n_restarts} train_acc={train_acc:.4f} test_acc={gru_acc:.4f}")

    return svm_acc, rf_acc, gru_acc, n_restarts


def main():
    LINES = []

    def log(msg=""):
        print(msg)
        LINES.append(str(msg))

    t0 = time.time()
    X_raw, y_raw, groups_raw = v1.build_raw_dataset_with_groups()
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_raw)
    X_v2feat = raw_plus_complex_fft_features(X_raw)
    log(f"Dataset: X_raw={X_raw.shape} X_v2feat={X_v2feat.shape} "
        f"classes={list(encoder.classes_)} n_groups={len(np.unique(groups_raw))}")

    # ---- single grouped 5-fold split (rep_seed=1000), identical to v1 ----
    single_rows = []
    total_restarts_single = 0
    for fold_num, (train_idx, test_idx) in enumerate(
            v1.grouped_kfold_indices(groups_raw, N_FOLDS, rep_seed=1000), start=1):
        log(f"[single-split] fold {fold_num}:")
        svm_acc, rf_acc, gru_acc, n_restarts = run_one_fold(
            X_raw, X_v2feat, y_enc, groups_raw, train_idx, test_idx, rep_seed=1000, fold_num=fold_num, log=log)
        total_restarts_single += n_restarts
        single_rows.append({"fold": fold_num, "svm_acc": svm_acc, "rf_acc": rf_acc, "gru_acc": gru_acc})

    with open(os.path.join(LANE_DIR, "lane4_v2_single_split.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "svm_acc", "rf_acc", "gru_acc"])
        writer.writeheader()
        writer.writerows(single_rows)

    svm_mean = np.mean([r["svm_acc"] for r in single_rows])
    svm_sd = np.std([r["svm_acc"] for r in single_rows], ddof=1)
    rf_mean = np.mean([r["rf_acc"] for r in single_rows])
    rf_sd = np.std([r["rf_acc"] for r in single_rows], ddof=1)
    gru_mean = np.mean([r["gru_acc"] for r in single_rows])
    gru_sd = np.std([r["gru_acc"] for r in single_rows], ddof=1)
    log(f"\n[v2 single-split summary] SVM={svm_mean:.4f}+-{svm_sd:.4f}  "
        f"RF={rf_mean:.4f}+-{rf_sd:.4f}  GRU={gru_mean:.4f}+-{gru_sd:.4f}  "
        f"(total GRU restarts used: {total_restarts_single})")

    # ---- 15x repeated grouped 5-fold ----
    repeated_rows = []
    total_restarts_repeated = 0
    for rep in range(N_REPETITIONS):
        rep_seed = 1000 + rep
        for fold_num, (train_idx, test_idx) in enumerate(
                v1.grouped_kfold_indices(groups_raw, N_FOLDS, rep_seed), start=1):
            svm_acc, rf_acc, gru_acc, n_restarts = run_one_fold(
                X_raw, X_v2feat, y_enc, groups_raw, train_idx, test_idx, rep_seed, fold_num, log=log)
            total_restarts_repeated += n_restarts
            repeated_rows.append({
                "repetition": rep, "fold": fold_num,
                "svm_acc": svm_acc, "rf_acc": rf_acc, "gru_acc": gru_acc,
            })
        rep_svm = np.mean([r["svm_acc"] for r in repeated_rows if r["repetition"] == rep])
        rep_rf = np.mean([r["rf_acc"] for r in repeated_rows if r["repetition"] == rep])
        rep_gru = np.mean([r["gru_acc"] for r in repeated_rows if r["repetition"] == rep])
        log(f"rep {rep + 1}/{N_REPETITIONS}  svm={rep_svm:.4f} rf={rep_rf:.4f} gru={rep_gru:.4f}  "
            f"(elapsed {time.time() - t0:.0f}s)")

    with open(os.path.join(LANE_DIR, "lane4_v2_repeated_grouped_cv_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repetition", "fold", "svm_acc", "rf_acc", "gru_acc"])
        writer.writeheader()
        writer.writerows(repeated_rows)

    svm_all = np.array([r["svm_acc"] for r in repeated_rows])
    rf_all = np.array([r["rf_acc"] for r in repeated_rows])
    gru_all = np.array([r["gru_acc"] for r in repeated_rows])
    log(f"\n[v2 15x-repeated summary, n={len(repeated_rows)} paired diffs] "
        f"SVM={svm_all.mean():.4f}+-{svm_all.std(ddof=1):.4f}  "
        f"RF={rf_all.mean():.4f}+-{rf_all.std(ddof=1):.4f}  "
        f"GRU={gru_all.mean():.4f}+-{gru_all.std(ddof=1):.4f}  "
        f"(total GRU restarts used: {total_restarts_repeated})")
    log(f"\nTotal runtime: {time.time() - t0:.0f}s")

    with open(os.path.join(LANE_DIR, "run_v2_stdout.log"), "w") as f:
        f.write("\n".join(LINES) + "\n")

    log("Saved audit_fixes/lane4/lane4_v2_single_split.csv, "
        "lane4_v2_repeated_grouped_cv_raw.csv, and run_v2_stdout.log")


if __name__ == "__main__":
    main()

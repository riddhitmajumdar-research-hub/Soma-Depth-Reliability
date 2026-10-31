"""
LANE 4 -- additional baseline architectures (SVM-RBF, Random Forest, small
GRU) under exactly the protocol pre-registered in audit_fixes/lane4/PREREG.md.

READ-ONLY on the rest of the repo. Imports (does not copy/reimplement):
  - Source/repeated_grouped_cv_log11.py :
        grouped_kfold_indices, build_baseline_dataset_with_groups is NOT
        reused (that builds FFT-free hand features for logistic regression;
        lane4's SVM/RF need their own FFT-magnitude features per PREREG) --
        but build_raw_dataset_with_groups (via cnn_classifier) and the fold
        splitter grouped_kfold_indices ARE reused verbatim, so folds line up
        with the existing baseline/CNN/ResNet1D paired data.
  - Source/cnn_classifier.py : build_raw_dataset_with_groups
  - Source/constants.py      : SEED

Writes only inside audit_fixes/lane4/. Does not retrain or touch anything
outside this lane, does not touch the running ResNet1D sweep (pid 7429).

Run from repo root:
    python3 audit_fixes/lane4/run_lane4_comparison.py
"""

import os
import sys
import csv
import time

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_DIR = os.path.join(REPO_ROOT, "Source")
LANE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SOURCE_DIR)

from constants import SEED  # noqa: E402  (imported, not hardcoded)
from cnn_classifier import build_raw_dataset_with_groups  # noqa: E402
from repeated_grouped_cv_log11 import grouped_kfold_indices  # noqa: E402 (reused, not reimplemented)

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

torch.set_num_threads(2)
N_JOBS = 2

np.random.seed(SEED)
torch.manual_seed(SEED)

N_FOLDS = 5
N_REPETITIONS = 15
INNER_FOLDS = 3
WINDOW_SIZE = 100  # samples, matches cnn_classifier.extract_raw_windows default

SVM_GRID = [
    {"C": 1.0, "gamma": "scale"},
    {"C": 10.0, "gamma": "scale"},
    {"C": 1.0, "gamma": 0.01},
    {"C": 10.0, "gamma": 0.01},
]

RF_GRID = [
    {"max_depth": None, "min_samples_leaf": 1},
    {"max_depth": 10, "min_samples_leaf": 1},
    {"max_depth": None, "min_samples_leaf": 4},
    {"max_depth": 10, "min_samples_leaf": 4},
]

GRU_GRID = [
    {"lr": 0.001, "dropout": 0.0},
    {"lr": 0.001, "dropout": 0.3},
    {"lr": 0.01, "dropout": 0.0},
    {"lr": 0.01, "dropout": 0.3},
]

GRU_EPOCHS = 60  # matches train_cnn's epoch count in repeated_grouped_cv_log11.py


# ---------------------------------------------------------------------------
# Feature extraction: FFT magnitude, bins up to Nyquist
# ---------------------------------------------------------------------------
def fft_magnitude_features(X_raw):
    # rfft of a length-100 real window gives 51 bins (0 .. Nyquist inclusive)
    spec = np.fft.rfft(X_raw, axis=1)
    return np.abs(spec)


class SmallGRU(nn.Module):
    def __init__(self, hidden_size=16, dropout=0.0, n_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        _, h_n = self.gru(x)
        h = h_n[-1]  # (batch, hidden_size)
        h = self.dropout(h)
        return self.fc(h)


def train_gru(X_train_t, y_train_t, lr, dropout, n_epochs=GRU_EPOCHS):
    model = SmallGRU(dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = crit(model(X_train_t), y_train_t)
        loss.backward()
        opt.step()
    model.eval()
    return model


def gru_inner_cv_select(X_train, y_train_enc, rep_seed, fold_num):
    """3-fold stratified inner CV on the TRAIN fold only; picks best of GRU_GRID."""
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                             random_state=rep_seed * 1000 + fold_num)
    best_cfg, best_score = None, -1.0
    for cfg in GRU_GRID:
        scores = []
        for inner_tr, inner_vl in inner.split(X_train, y_train_enc):
            Xt = torch.tensor(X_train[inner_tr], dtype=torch.float32).unsqueeze(-1)
            yt = torch.tensor(y_train_enc[inner_tr], dtype=torch.long)
            Xv = torch.tensor(X_train[inner_vl], dtype=torch.float32).unsqueeze(-1)
            yv = torch.tensor(y_train_enc[inner_vl], dtype=torch.long)
            model = train_gru(Xt, yt, cfg["lr"], cfg["dropout"])
            with torch.no_grad():
                acc = (model(Xv).argmax(dim=1) == yv).float().mean().item()
            scores.append(acc)
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score, best_cfg = mean_score, cfg
    return best_cfg


def svm_inner_cv_select(X_train, y_train, rep_seed, fold_num):
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                             random_state=rep_seed * 1000 + fold_num)
    best_cfg, best_score = None, -1.0
    for cfg in SVM_GRID:
        scores = []
        for inner_tr, inner_vl in inner.split(X_train, y_train):
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


def rf_inner_cv_select(X_train, y_train, rep_seed, fold_num):
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True,
                             random_state=rep_seed * 1000 + fold_num)
    best_cfg, best_score = None, -1.0
    for cfg in RF_GRID:
        scores = []
        for inner_tr, inner_vl in inner.split(X_train, y_train):
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


def run_one_fold(X_raw, X_fft, y_enc, train_idx, test_idx, rep_seed, fold_num):
    # --- SVM on FFT features ---
    svm_cfg = svm_inner_cv_select(X_fft[train_idx], y_enc[train_idx], rep_seed, fold_num)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_fft[train_idx])
    Xte = scaler.transform(X_fft[test_idx])
    svm = SVC(kernel="rbf", **svm_cfg)
    svm.fit(Xtr, y_enc[train_idx])
    svm_acc = accuracy_score(y_enc[test_idx], svm.predict(Xte))

    # --- Random forest on same (scaled) FFT features ---
    rf_cfg = rf_inner_cv_select(X_fft[train_idx], y_enc[train_idx], rep_seed, fold_num)
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=N_JOBS, **rf_cfg)
    rf.fit(Xtr, y_enc[train_idx])
    rf_acc = accuracy_score(y_enc[test_idx], rf.predict(Xte))

    # --- Small GRU on raw windows ---
    gru_cfg = gru_inner_cv_select(X_raw[train_idx], y_enc[train_idx], rep_seed, fold_num)
    X_train_t = torch.tensor(X_raw[train_idx], dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_enc[train_idx], dtype=torch.long)
    X_test_t = torch.tensor(X_raw[test_idx], dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_enc[test_idx], dtype=torch.long)
    gru = train_gru(X_train_t, y_train_t, gru_cfg["lr"], gru_cfg["dropout"])
    with torch.no_grad():
        gru_acc = (gru(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()

    return svm_acc, rf_acc, gru_acc, svm_cfg, rf_cfg, gru_cfg


def main():
    t0 = time.time()
    X_raw, y_raw, groups_raw = build_raw_dataset_with_groups()
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y_raw)
    X_fft = fft_magnitude_features(X_raw)
    print(f"Dataset: X_raw={X_raw.shape} X_fft={X_fft.shape} "
          f"classes={list(encoder.classes_)} n_groups={len(np.unique(groups_raw))}")

    # ---- single grouped 5-fold split (rep_seed=1000, matches repetition 0
    #      of the existing 15x script's rep_seed = 1000 + rep) ----
    single_rows = []
    for fold_num, (train_idx, test_idx) in enumerate(
            grouped_kfold_indices(groups_raw, N_FOLDS, rep_seed=1000), start=1):
        svm_acc, rf_acc, gru_acc, svm_cfg, rf_cfg, gru_cfg = run_one_fold(
            X_raw, X_fft, y_enc, train_idx, test_idx, rep_seed=1000, fold_num=fold_num)
        print(f"[single-split] fold {fold_num}: svm={svm_acc:.4f} (cfg={svm_cfg})  "
              f"rf={rf_acc:.4f} (cfg={rf_cfg})  gru={gru_acc:.4f} (cfg={gru_cfg})")
        single_rows.append({"fold": fold_num, "svm_acc": svm_acc, "rf_acc": rf_acc, "gru_acc": gru_acc})

    with open(os.path.join(LANE_DIR, "lane4_single_split.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "svm_acc", "rf_acc", "gru_acc"])
        writer.writeheader()
        writer.writerows(single_rows)

    svm_mean = np.mean([r["svm_acc"] for r in single_rows])
    svm_sd = np.std([r["svm_acc"] for r in single_rows], ddof=1)
    rf_mean = np.mean([r["rf_acc"] for r in single_rows])
    rf_sd = np.std([r["rf_acc"] for r in single_rows], ddof=1)
    gru_mean = np.mean([r["gru_acc"] for r in single_rows])
    gru_sd = np.std([r["gru_acc"] for r in single_rows], ddof=1)
    print(f"\n[single-split summary] SVM={svm_mean:.4f}+-{svm_sd:.4f}  "
          f"RF={rf_mean:.4f}+-{rf_sd:.4f}  GRU={gru_mean:.4f}+-{gru_sd:.4f}")

    # ---- 15x repeated grouped 5-fold ----
    repeated_rows = []
    for rep in range(N_REPETITIONS):
        rep_seed = 1000 + rep
        for fold_num, (train_idx, test_idx) in enumerate(
                grouped_kfold_indices(groups_raw, N_FOLDS, rep_seed), start=1):
            svm_acc, rf_acc, gru_acc, svm_cfg, rf_cfg, gru_cfg = run_one_fold(
                X_raw, X_fft, y_enc, train_idx, test_idx, rep_seed, fold_num)
            repeated_rows.append({
                "repetition": rep, "fold": fold_num,
                "svm_acc": svm_acc, "rf_acc": rf_acc, "gru_acc": gru_acc,
            })
        rep_svm = np.mean([r["svm_acc"] for r in repeated_rows if r["repetition"] == rep])
        rep_rf = np.mean([r["rf_acc"] for r in repeated_rows if r["repetition"] == rep])
        rep_gru = np.mean([r["gru_acc"] for r in repeated_rows if r["repetition"] == rep])
        print(f"rep {rep + 1}/{N_REPETITIONS}  svm={rep_svm:.4f} rf={rep_rf:.4f} gru={rep_gru:.4f}  "
              f"(elapsed {time.time() - t0:.0f}s)")

    with open(os.path.join(LANE_DIR, "lane4_repeated_grouped_cv_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repetition", "fold", "svm_acc", "rf_acc", "gru_acc"])
        writer.writeheader()
        writer.writerows(repeated_rows)

    svm_all = np.array([r["svm_acc"] for r in repeated_rows])
    rf_all = np.array([r["rf_acc"] for r in repeated_rows])
    gru_all = np.array([r["gru_acc"] for r in repeated_rows])
    print(f"\n[15x-repeated summary, n={len(repeated_rows)} paired diffs] "
          f"SVM={svm_all.mean():.4f}+-{svm_all.std(ddof=1):.4f}  "
          f"RF={rf_all.mean():.4f}+-{rf_all.std(ddof=1):.4f}  "
          f"GRU={gru_all.mean():.4f}+-{gru_all.std(ddof=1):.4f}")
    print(f"\nTotal runtime: {time.time() - t0:.0f}s")
    print("Saved audit_fixes/lane4/lane4_single_split.csv and "
          "audit_fixes/lane4/lane4_repeated_grouped_cv_raw.csv")


if __name__ == "__main__":
    main()

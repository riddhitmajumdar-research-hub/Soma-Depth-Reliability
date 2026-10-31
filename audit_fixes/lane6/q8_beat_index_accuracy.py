"""
Lane6c -- Q8. READ-ONLY on the repo. No saved Stage-3 (non-degradation)
checkpoints exist in Source/models/ (checked: only 4 files there, all
Stage-4/calibration). This script therefore trains fresh, using the EXACT
same protocol/hyperparameters as Source/significance_testing.py (5-fold
StratifiedKFold, random_state=42, same model classes/epoch counts/restart
logic) -- copied here rather than importing significance_testing.py as a
module, because that module executes top-level training + plt.show() +
plt.savefig() on import (not gated by `if __name__=="__main__"`), which
would have side effects outside this lane. Only imports of classes/functions
(SimpleCNN, ResNet1D, generate_signal, stiffness_classes, extract_pulses_with_features,
SEED) are taken from the repo unchanged; the training loop code below is a
verbatim copy of significance_testing.py's, only instrumented to also carry
a k (beat-index) tag per window/feature-row through the same split so
accuracy can be broken out by k afterward.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Source"))

from physics_model import generate_signal, stiffness_classes            # noqa: E402
from source_classifier import extract_pulses_with_features              # noqa: E402
from cnn_classifier import SimpleCNN                                    # noqa: E402
from resnet_classifier import ResNet1D                                  # noqa: E402
from constants import SEED                                              # noqa: E402

np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = Path(__file__).resolve().parent
N_FOLDS = 5
WINDOW_SIZE = 100


def extract_raw_windows_with_k(signal, time, heart_rate_hz, duration=3.0, window_size=WINDOW_SIZE):
    """Mirrors cnn_classifier.extract_raw_windows (Source/cnn_classifier.py:35-44) exactly, plus k."""
    beat_period = 1.0 / heart_rate_hz
    beat_times = np.arange(0, duration, beat_period)
    rows = []
    for k, beat_t in enumerate(beat_times):
        start_idx = np.searchsorted(time, beat_t)
        end_idx = start_idx + window_size
        if end_idx > len(signal):
            continue
        rows.append((signal[start_idx:end_idx], k))
    return rows


def build_baseline_dataset_with_k(n_runs_per_class=30, jitter_std=0.03, seed=SEED):
    """Mirrors source_classifier.build_dataset (Source/source_classifier.py:57-72) exactly, plus k
    (k assigned by enumerate order over extract_pulses_with_features's own beat_times loop, which
    Source/source_classifier.py:36 iterates in the same order/skip-condition as extract_raw_windows)."""
    np.random.seed(seed)
    X, y, ks = [], [], []
    for name, kpa in stiffness_classes.items():
        for _ in range(n_runs_per_class):
            jittered_hr = 1.15 + np.random.normal(0, jitter_std)
            time_arr, signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
            feature_rows = extract_pulses_with_features(signal, time_arr, jittered_hr, tof)
            for k, features in enumerate(feature_rows):
                X.append(features)
                y.append(name)
                ks.append(k)
    return np.array(X), np.array(y), np.array(ks)


def build_raw_dataset_with_k(n_runs_per_class=30, jitter_std=0.03, seed=SEED):
    """Mirrors cnn_classifier.build_raw_dataset (Source/cnn_classifier.py:49-62) exactly, plus k."""
    np.random.seed(seed)
    X, y, ks = [], [], []
    for name, kpa in stiffness_classes.items():
        for _ in range(n_runs_per_class):
            jittered_hr = 1.15 + np.random.normal(0, jitter_std)
            time_arr, signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
            for window, k in extract_raw_windows_with_k(signal, time_arr, jittered_hr):
                X.append(window)
                y.append(name)
                ks.append(k)
    return np.array(X), np.array(y), np.array(ks)


def main():
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 78)
    p("Q8 -- accuracy by beat index k, clean noise-free windows,")
    p("Stage-3 baseline / SimpleCNN / ResNet1D -- training protocol copied")
    p("verbatim from Source/significance_testing.py (no saved checkpoints existed)")
    p("=" * 78)

    # ---------------- Baseline ----------------
    X_feat, y_feat, ks_feat = build_baseline_dataset_with_k()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    baseline_correct_by_k = {}
    baseline_correct_by_class = {}
    baseline_total = 0
    baseline_correct_total = 0

    for train_idx, test_idx in skf.split(X_feat, y_feat):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_feat[train_idx])
        X_test = scaler.transform(X_feat[test_idx])
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_feat[train_idx])
        preds = model.predict(X_test)
        correct = preds == y_feat[test_idx]
        baseline_total += len(test_idx)
        baseline_correct_total += correct.sum()
        for k_val, c, cls in zip(ks_feat[test_idx], correct, y_feat[test_idx]):
            baseline_correct_by_k.setdefault(k_val, [0, 0])
            baseline_correct_by_k[k_val][0] += int(c)
            baseline_correct_by_k[k_val][1] += 1
            baseline_correct_by_class.setdefault(cls, [0, 0])
            baseline_correct_by_class[cls][0] += int(c)
            baseline_correct_by_class[cls][1] += 1

    p(f"\nBaseline (LogisticRegression, {N_FOLDS}-fold pooled test predictions, n={baseline_total}):")
    p(f"  Overall accuracy: {baseline_correct_total/baseline_total:.4%}")
    p("  By class:")
    for cls, (c, n) in sorted(baseline_correct_by_class.items()):
        p(f"    {cls:<22s} n={n:4d}  acc={c/n:.4%}")
    p("  By beat index k:")
    for k_val, (c, n) in sorted(baseline_correct_by_k.items()):
        p(f"    k={k_val}  n={n:4d}  acc={c/n:.4%}")

    # ---------------- CNN + ResNet1D ----------------
    X_raw, y_raw, ks_raw = build_raw_dataset_with_k()
    encoder = LabelEncoder()
    y_raw_enc = encoder.fit_transform(y_raw)

    skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    crit = nn.CrossEntropyLoss()

    cnn_correct_by_k, cnn_correct_by_class = {}, {}
    cnn_total, cnn_correct_total = 0, 0
    resnet_correct_by_k, resnet_correct_by_class = {}, {}
    resnet_total, resnet_correct_total = 0, 0

    for fold_num, (train_idx, test_idx) in enumerate(skf2.split(X_raw, y_raw_enc), start=1):
        X_train_t = torch.tensor(X_raw[train_idx], dtype=torch.float32).unsqueeze(1)
        y_train_t = torch.tensor(y_raw_enc[train_idx], dtype=torch.long)
        X_test_t = torch.tensor(X_raw[test_idx], dtype=torch.float32).unsqueeze(1)
        y_test_t = torch.tensor(y_raw_enc[test_idx], dtype=torch.long)

        cnn = SimpleCNN(n_classes=3)
        opt = torch.optim.Adam(cnn.parameters(), lr=0.001)
        for epoch in range(60):
            cnn.train()
            opt.zero_grad()
            loss = crit(cnn(X_train_t), y_train_t)
            loss.backward()
            opt.step()
        cnn.eval()
        with torch.no_grad():
            preds_cnn = cnn(X_test_t).argmax(dim=1).numpy()
        correct_cnn = preds_cnn == y_raw_enc[test_idx]
        cnn_total += len(test_idx)
        cnn_correct_total += correct_cnn.sum()

        train_sub, val_sub = train_test_split(
            train_idx, test_size=0.2, random_state=42, stratify=y_raw_enc[train_idx]
        )
        X_tr = torch.tensor(X_raw[train_sub], dtype=torch.float32).unsqueeze(1)
        y_tr = torch.tensor(y_raw_enc[train_sub], dtype=torch.long)
        X_vl = torch.tensor(X_raw[val_sub], dtype=torch.float32).unsqueeze(1)
        y_vl = torch.tensor(y_raw_enc[val_sub], dtype=torch.long)

        max_restarts = 5
        for attempt in range(max_restarts):
            resnet = ResNet1D(n_classes=3)
            opt2 = torch.optim.Adam(resnet.parameters(), lr=0.001)
            best_val_loss, no_improve, best_state = float("inf"), 0, None
            for epoch in range(150):
                resnet.train()
                opt2.zero_grad()
                loss = crit(resnet(X_tr), y_tr)
                loss.backward()
                opt2.step()
                resnet.eval()
                with torch.no_grad():
                    val_loss = crit(resnet(X_vl), y_vl).item()
                if val_loss < best_val_loss:
                    best_val_loss, no_improve = val_loss, 0
                    best_state = {k2: v.clone() for k2, v in resnet.state_dict().items()}
                else:
                    no_improve += 1
                if no_improve >= 15:
                    break
            resnet.load_state_dict(best_state)
            resnet.eval()
            with torch.no_grad():
                train_acc_check = (resnet(X_tr).argmax(dim=1) == y_tr).float().mean().item()
            if train_acc_check > 0.5:
                break
            else:
                p(f"    [Fold {fold_num}] ResNet1D collapsed on attempt {attempt+1} "
                  f"(train_acc={train_acc_check:.2%}) -- restarting")

        with torch.no_grad():
            preds_resnet = resnet(X_test_t).argmax(dim=1).numpy()
        correct_resnet = preds_resnet == y_raw_enc[test_idx]
        resnet_total += len(test_idx)
        resnet_correct_total += correct_resnet.sum()

        p(f"  Fold {fold_num}: CNN={correct_cnn.mean():.2%}  ResNet1D={correct_resnet.mean():.2%}")

        for k_val, c, cls_enc in zip(ks_raw[test_idx], correct_cnn, y_raw_enc[test_idx]):
            cnn_correct_by_k.setdefault(k_val, [0, 0]); cnn_correct_by_k[k_val][0] += int(c); cnn_correct_by_k[k_val][1] += 1
            cnn_correct_by_class.setdefault(cls_enc, [0, 0]); cnn_correct_by_class[cls_enc][0] += int(c); cnn_correct_by_class[cls_enc][1] += 1
        for k_val, c, cls_enc in zip(ks_raw[test_idx], correct_resnet, y_raw_enc[test_idx]):
            resnet_correct_by_k.setdefault(k_val, [0, 0]); resnet_correct_by_k[k_val][0] += int(c); resnet_correct_by_k[k_val][1] += 1
            resnet_correct_by_class.setdefault(cls_enc, [0, 0]); resnet_correct_by_class[cls_enc][0] += int(c); resnet_correct_by_class[cls_enc][1] += 1

    p(f"\nCNN (SimpleCNN, {N_FOLDS}-fold pooled test predictions, n={cnn_total}):")
    p(f"  Overall accuracy: {cnn_correct_total/cnn_total:.4%}")
    p("  By class:")
    for cls_enc, (c, n) in sorted(cnn_correct_by_class.items()):
        p(f"    {encoder.inverse_transform([cls_enc])[0]:<22s} n={n:4d}  acc={c/n:.4%}")
    p("  By beat index k:")
    for k_val, (c, n) in sorted(cnn_correct_by_k.items()):
        p(f"    k={k_val}  n={n:4d}  acc={c/n:.4%}")

    p(f"\nResNet1D ({N_FOLDS}-fold pooled test predictions, n={resnet_total}):")
    p(f"  Overall accuracy: {resnet_correct_total/resnet_total:.4%}")
    p("  By class:")
    for cls_enc, (c, n) in sorted(resnet_correct_by_class.items()):
        p(f"    {encoder.inverse_transform([cls_enc])[0]:<22s} n={n:4d}  acc={c/n:.4%}")
    p("  By beat index k:")
    for k_val, (c, n) in sorted(resnet_correct_by_k.items()):
        p(f"    k={k_val}  n={n:4d}  acc={c/n:.4%}")

    (OUT_DIR / "q8_beat_index_accuracy_stdout.log").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'q8_beat_index_accuracy_stdout.log'}")


if __name__ == "__main__":
    main()

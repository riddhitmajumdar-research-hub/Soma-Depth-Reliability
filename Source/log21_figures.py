"""
SOMA -- Log 21 figures: ETS Montreal Path 2 external validation.

Panel A: null distribution (200 shuffled-label controls) vs. real model
         accuracy, showing how far outside the null the real result sits.
Panel B: confusion matrix on the held-out Acqui6 test set, checking for
         class asymmetry the way Log 14's healthy->cirrhosis collapse
         showed this project needs to check rather than assume.

No retraining, and no import of ets_montreal_path2_retrain.py -- that
module has no __main__ guard, so importing it re-runs the entire training
script as a side effect. extract_windows() and the relevant constants are
copied inline instead. Only the saved model weights are loaded.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from cnn_classifier import SimpleCNN

DATA_ROOT = Path(__file__).resolve().parents[1] / "data/ets_montreal/extracted"
TEST_SEQUENCES = ["Acqui6"]
INCLUSION_ROWS = (1526, 1626)
BACKGROUND_ROWS = (300, 400)
CSV_PATH = "logs/ets_path2_shuffle_results.csv"
MODEL_SAVE_PATH = "models/ets_path2_real_model.pt"
OUT_DIR = Path("../outputs/stage4/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_windows(seq_names, label):
    X, y = [], []
    for seq in seq_names:
        rf_dir = DATA_ROOT / seq / "RF"
        rf_files = sorted(rf_dir.glob("RF*.mat"))
        for rf_path in rf_files:
            data = loadmat(str(rf_path))
            rf = data["RF"].astype(np.float32)
            for col in range(rf.shape[1]):
                inc_win = rf[INCLUSION_ROWS[0]:INCLUSION_ROWS[1], col].copy()
                bg_win = rf[BACKGROUND_ROWS[0]:BACKGROUND_ROWS[1], col].copy()
                inc_win = (inc_win - inc_win.mean()) / (inc_win.std() + 1e-8)
                bg_win = (bg_win - bg_win.mean()) / (bg_win.std() + 1e-8)
                X.append(inc_win); y.append(1)
                X.append(bg_win); y.append(0)
        print(f"  [{label}] {seq}: {len(rf_files)} RF files processed, "
              f"running total windows = {len(X)}", flush=True)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


print("Rebuilding held-out test set (Acqui6)...", flush=True)
X_test, y_test = extract_windows(TEST_SEQUENCES, "TEST")
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

model = SimpleCNN(n_classes=2)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
model.eval()

with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1).numpy()

acc_real = float(np.mean(preds == y_test))
print(f"Loaded model, held-out accuracy = {acc_real:.4f} (should match 0.9725)")

control_accs = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        control_accs.append(float(row["control_acc"]))
control_accs = np.array(control_accs)
print(f"Loaded {len(control_accs)} control accuracies, "
      f"mean={control_accs.mean():.4f}, std={control_accs.std():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.hist(control_accs, bins=25, color="#8FA6B2", edgecolor="white", alpha=0.85,
        label=f"Shuffled-label controls (n={len(control_accs)})")
ax.axvline(acc_real, color="#C0392B", linewidth=2.2,
           label=f"Real model accuracy = {acc_real:.4f}")
ax.axvline(control_accs.mean(), color="#34495E", linewidth=1.3, linestyle="--",
           label=f"Control mean = {control_accs.mean():.4f}")
ax.set_xlabel("Held-out accuracy (Acqui6)")
ax.set_ylabel("Count")
ax.set_title("ETS Montreal Path 2: real model vs. shuffled-label null\n(p = 0.0050)")
ax.legend(fontsize=8.5, loc="upper right")
ax.set_xlim(0.2, 1.02)

labels = ["Background", "Inclusion"]
cm = np.zeros((2, 2), dtype=int)
for t, p in zip(y_test, preds):
    cm[int(t), int(p)] += 1

ax2 = axes[1]
cmap = LinearSegmentedColormap.from_list("soma_teal", ["#F4F7F6", "#0F6E56"])
im = ax2.imshow(cm, cmap=cmap)
ax2.set_xticks([0, 1]); ax2.set_xticklabels(labels)
ax2.set_yticks([0, 1]); ax2.set_yticklabels(labels)
ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
ax2.set_title("Held-out confusion matrix (Acqui6, n=%d)" % len(y_test))

for i in range(2):
    for j in range(2):
        pct = 100 * cm[i, j] / cm[i].sum()
        color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
        ax2.text(j, i, f"{cm[i,j]}\n({pct:.1f}%)", ha="center", va="center",
                  color=color, fontsize=11)

fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()

out_path = OUT_DIR / "log21_ets_montreal_validation.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved figure to {out_path}")

bg_recall = cm[0, 0] / cm[0].sum()
inc_recall = cm[1, 1] / cm[1].sum()
print(f"\nBackground recall: {bg_recall:.4f}")
print(f"Inclusion recall:  {inc_recall:.4f}")
print(f"Asymmetry (|diff|): {abs(bg_recall - inc_recall):.4f}")

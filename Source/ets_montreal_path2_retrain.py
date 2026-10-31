# SOMA -- Item 5, Path 2: retrain SimpleCNN from scratch on real ETS Montreal RF.
#
# Row math (CORRECTED -- an earlier depth-per-row calculation in this pass
# was off by 10x; this version is verified against energy analysis of real
# RF0001.mat): depth_per_row = 6cm / 3152 rows = 0.0019027 cm/row.
#   Inclusion (Figure 4.2 geometry: centered, 2-4cm depth, confirmed via
#   visual inspection of frame0001.png AND real signal energy ~30x background)
#     -> rows 1051-2101 (full span), window used: 1526-1626 (centered, peak signal)
#   Background: rows 300-400 -- verified clean, far from main-bang (~rows 0-20),
#   inclusion, and the back-wall artifact (~rows 3130+, energy even higher than
#   inclusion -- deliberately avoided, not used as background).
#
# Held out by WHOLE SEQUENCE (Acqui6), not random split -- A-lines within one
# sequence are correlated (same phantom, same session), so random splitting
# would leak information and overstate accuracy.
#
# No subsampling: all 256 A-lines per file used, so as not
# to compromise scope. N_SHUFFLES=200 for real significance testing, matching
# the rigor standard used throughout the rest of this project.
#
# Checkpointed: control model results append to CSV as they complete, so an
# interrupted run loses nothing and can be inspected mid-flight.

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from pathlib import Path
import csv
import os
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from cnn_classifier import SimpleCNN

DATA_ROOT = Path(__file__).resolve().parents[1] / "data/ets_montreal/extracted"
TRAIN_SEQUENCES = ["Acqui1", "Acqui2", "Acqui3", "Acqui4", "Acqui5"]
TEST_SEQUENCES = ["Acqui6"]

INCLUSION_ROWS = (1526, 1626)   # centered in verified inclusion span, 100-sample window
BACKGROUND_ROWS = (300, 400)    # verified clean, far from all artifact zones
N_EPOCHS = 60
N_SHUFFLES = 200
SEED = 42
CSV_PATH = "logs/ets_path2_shuffle_results.csv"
MODEL_SAVE_PATH = "models/ets_path2_real_model.pt"

torch.manual_seed(SEED)
np.random.seed(SEED)


def extract_windows(seq_names, label):
    X, y = [], []
    for seq in seq_names:
        rf_dir = DATA_ROOT / seq / "RF"
        rf_files = sorted(rf_dir.glob("RF*.mat"))
        for rf_path in rf_files:
            data = loadmat(str(rf_path))
            rf = data["RF"].astype(np.float32)  # (3152, 256)
            for col in range(rf.shape[1]):
                inc_win = rf[INCLUSION_ROWS[0]:INCLUSION_ROWS[1], col].copy()
                bg_win = rf[BACKGROUND_ROWS[0]:BACKGROUND_ROWS[1], col].copy()

                inc_win = (inc_win - inc_win.mean()) / (inc_win.std() + 1e-8)
                bg_win = (bg_win - bg_win.mean()) / (bg_win.std() + 1e-8)

                X.append(inc_win)
                y.append(1)  # inclusion
                X.append(bg_win)
                y.append(0)  # background
        print(f"  [{label}] {seq}: {len(rf_files)} RF files processed, "
              f"running total windows = {len(X)}", flush=True)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_cnn(X, y, n_epochs=N_EPOCHS):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.long)
    model = SimpleCNN(n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        opt.step()
    model.eval()
    return model


def evaluate(model, X_test_t, y_test):
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).numpy()
    return np.mean(preds == y_test)


print("="*70)
print("SOMA ETS Montreal Path 2 -- real-RF retrain")
print("="*70)
t0 = time.time()

print("\nBuilding training set (Acqui1-5)...", flush=True)
X_train, y_train = extract_windows(TRAIN_SEQUENCES, "TRAIN")
print(f"Total training windows: {len(X_train)} "
      f"({(y_train==1).sum()} inclusion, {(y_train==0).sum()} background)")

print("\nBuilding held-out test set (Acqui6)...", flush=True)
X_test, y_test = extract_windows(TEST_SEQUENCES, "TEST")
print(f"Total test windows: {len(X_test)} "
      f"({(y_test==1).sum()} inclusion, {(y_test==0).sum()} background)")
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)

print(f"\nData prep done at {time.time()-t0:.1f}s. Training real model...", flush=True)
model_real = train_cnn(X_train, y_train)
acc_real = evaluate(model_real, X_test_t, y_test)
print(f"REAL MODEL held-out accuracy: {acc_real:.4f}  (elapsed {time.time()-t0:.1f}s)", flush=True)
torch.save(model_real.state_dict(), MODEL_SAVE_PATH)
print(f"Saved real model to {MODEL_SAVE_PATH}")

# Resumable: check what's already done
already_done = set()
if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            already_done.add(int(row["shuffle_idx"]))
    print(f"Found {len(already_done)} existing shuffle results, resuming from there.")

fieldnames = ["shuffle_idx", "control_acc"]
write_header = not os.path.exists(CSV_PATH)
with open(CSV_PATH, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for i in range(N_SHUFFLES):
        if i in already_done:
            continue
        y_shuffled = np.random.permutation(y_train)
        model_c = train_cnn(X_train, y_shuffled)
        acc_c = evaluate(model_c, X_test_t, y_test)
        writer.writerow({"shuffle_idx": i, "control_acc": acc_c})
        f.flush()
        if (i + 1) % 5 == 0:
            print(f"  shuffle {i+1}/{N_SHUFFLES} done, control_acc={acc_c:.4f}  "
                  f"(elapsed {time.time()-t0:.1f}s)", flush=True)

# Final summary
control_accs = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        control_accs.append(float(row["control_acc"]))
control_accs = np.array(control_accs)

p_value = (np.sum(control_accs >= acc_real) + 1) / (len(control_accs) + 1)

print("\n" + "="*70)
print("FINAL RESULT")
print("="*70)
print(f"Real model held-out accuracy:     {acc_real:.4f}")
print(f"Control mean accuracy (n={len(control_accs)}): {control_accs.mean():.4f}")
print(f"Control std:                      {control_accs.std():.4f}")
print(f"p-value:                          {p_value:.4f}")
print(f"Significant (p<=0.05):             {p_value <= 0.05}")
print(f"Total elapsed: {time.time()-t0:.1f}s")

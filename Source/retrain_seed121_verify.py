# SOMA -- Item 2: honest retrain + verify of seed 121 from the corrected
# 24-seed sweep (degradation_aware_final_sweep.py). Reuses that script's
# exact constants and build_degradation_aware_dataset()/train_cnn() logic
# unmodified -- only difference is this keeps the trained model object in
# scope to save its weights, which the original sweep driver never did.
# Target to diff against: final_sweep_log.txt line 63 --
#   naive_floor=8.18cm  last_sig=8.0  anomalous_recovery=False

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from degradation_aware_final_sweep import (
    build_degradation_aware_dataset,
    train_cnn,
    bh_fdr_correction,
    stiffness_classes,
    EVAL_DEPTHS_CM,
    N_SAMPLES_PER_CELL,
    N_SHUFFLES,
    FREQ,
)
from attenuation_model import attenuate_signal
from physics_model import generate_signal
from degradation_aware_final_sweep import generate_degraded_windows

SEED = 121

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Building training set for seed {SEED}...")
X_train, y_train_names = build_degradation_aware_dataset()
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train_names)

print("Training real model...")
model_real = train_cnn(X_train, y_train_encoded)

print(f"Training {N_SHUFFLES} shuffled-label controls...")
control_models = []
for i in range(N_SHUFFLES):
    y_shuffled = np.random.permutation(y_train_encoded)
    control_models.append(train_cnn(X_train, y_shuffled))
    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{N_SHUFFLES} done")

class_kpa_names = sorted(stiffness_classes.keys())
n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)

results = []
for depth_cm in EVAL_DEPTHS_CM:
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
    acc_real = np.mean(preds_real == y_cell_encoded)

    control_accs = []
    for model_c in control_models:
        with torch.no_grad():
            preds_c = model_c(X_cell_t).argmax(dim=1).numpy()
        control_accs.append(np.mean(preds_c == y_cell_encoded))
    control_accs = np.array(control_accs)

    p_value = (np.sum(control_accs >= acc_real) + 1) / (N_SHUFFLES + 1)
    results.append({"depth_cm": depth_cm, "acc_real": acc_real, "p_value": p_value})

all_p_values = [r["p_value"] for r in results]
significant_flags = bh_fdr_correction(all_p_values)
for r, sig in zip(results, significant_flags):
    r["fdr_significant"] = sig

floor_depth = None
for r in results:
    if not r["fdr_significant"]:
        floor_depth = r["depth_cm"]
        break
if floor_depth is None:
    floor_depth = EVAL_DEPTHS_CM[-1]

last_significant_depth = None
for r in results:
    if r["fdr_significant"]:
        last_significant_depth = r["depth_cm"]

anomalous_recovery = any(
    r["fdr_significant"] and r["depth_cm"] > floor_depth for r in results
)

print("\n" + "=" * 70)
print(f"SEED {SEED} REPRODUCTION RESULT")
print("=" * 70)
print(f"naive_floor_cm         = {floor_depth:.2f}")
print(f"last_significant_depth = {last_significant_depth}")
print(f"anomalous_recovery     = {anomalous_recovery}")
print("\nExpected (final_sweep_log.txt line 63):")
print("naive_floor=8.18cm  last_sig=8.0  anomalous_recovery=False")

match = (
    abs(floor_depth - 8.18) < 0.01
    and last_significant_depth == 8.0
    and anomalous_recovery is False
)
print(f"\nMATCH: {match}")

if match:
    save_path = "models/soma_certified_model_seed121_corrected_noise.pt"
    torch.save(model_real.state_dict(), save_path)
    print(f"Saved weights to {save_path}")
else:
    print("MISMATCH -- not saving weights. Do not proceed to item 4 until this is resolved.")

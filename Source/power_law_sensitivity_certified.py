# Stage 4+ -- Power-Law Attenuation Sensitivity, CERTIFIED (200-shuffle BH-FDR)
# File: Source/power_law_sensitivity_certified.py
#
# Same certification protocol as degradation_aware_cnn.py (Log 13/15): trains
# 200 shuffled-label control CNNs, uses their empirical spread as the null
# distribution, applies Benjamini-Hochberg FDR correction. The ONLY change
# from the original certification: evaluation signals use the power-law
# attenuation model (n=0.51, anchored at 20Hz) instead of Kelvin-Voigt.
# Control models are trained identically to the original (same KV-attenuated
# degradation-aware training set) -- this isolates the attenuation model as
# the only variable under test, keeping the null-generation process fixed.

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal as attenuate_signal_kv
from attenuation_model_powerlaw import attenuate_signal_powerlaw
from cnn_classifier import SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED, NOMINAL_FREQ_HZ, TRAINING_DEPTHS_CM, BASE_EVAL_DEPTHS_CM, DENSIFIED_DEPTHS_CM, SEED121_CERTIFIED_FLOOR_CM

torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_DEPTHS_CM = TRAINING_DEPTHS_CM
FREQ = NOMINAL_FREQ_HZ
N_SAMPLES_PER_DEPTH_PER_CLASS = 25
FINE_EVAL_DEPTHS_CM = DENSIFIED_DEPTHS_CM
EVAL_DEPTHS_CM = np.sort(np.unique(np.concatenate([BASE_EVAL_DEPTHS_CM, FINE_EVAL_DEPTHS_CM])))
N_SAMPLES_PER_CELL = 60
N_SHUFFLES = 200
# FIX (post-audit): was 67.03dB, the pre-Fix-#1 value Log 18 corrected everywhere
# else. This script (and its shuffle-control training) was never updated.
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)
MODEL_WEIGHTS_PATH = "../outputs/stage4/models/soma_certified_model_seed121_corrected_noise.pt"


def generate_windows(kpa, depth_m, frequency_hz, n_needed, attenuate_fn, jitter_std=0.03):
    windows = []
    while len(windows) < n_needed:
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        attenuated, alpha = attenuate_fn(clean_signal, kpa, depth_m, frequency_hz)
        noisy = attenuated + np.random.normal(0, SENSOR_NOISE_STD, size=attenuated.shape)
        rows = extract_raw_windows(noisy, time, jittered_hr)
        windows.extend(rows)
    return windows[:n_needed]


def build_training_set():
    # Matches degradation_aware_cnn.py exactly -- KV attenuation, same depths/counts
    X, y = [], []
    for depth_cm in TRAIN_DEPTHS_CM:
        depth_m = depth_cm / 100
        for name, kpa in stiffness_classes.items():
            windows = generate_windows(kpa, depth_m, FREQ, N_SAMPLES_PER_DEPTH_PER_CLASS, attenuate_signal_kv)
            X.extend(windows)
            y.extend([name] * len(windows))
    return np.array(X), np.array(y)


def train_cnn(X, y_encoded, n_classes=3, n_epochs=60):
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_encoded, dtype=torch.long)
    model = SimpleCNN(n_classes=n_classes)
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


def bh_fdr_correction(p_values, alpha=0.05):
    p_values = np.array(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool)
    max_passed_rank = np.max(np.where(passed)[0])
    significant = np.zeros(n, dtype=bool)
    significant[order[:max_passed_rank + 1]] = True
    return significant


if __name__ == "__main__":
    print(f"Loading verified seed-121 model weights (one seed from the certified 8.48cm +/- 1.81cm distribution) from {MODEL_WEIGHTS_PATH}...")
    model_real = SimpleCNN(n_classes=3)
    model_real.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    model_real.eval()
    print("Certified model loaded (no retraining).\n")

    encoder = LabelEncoder()
    encoder.fit(sorted(stiffness_classes.keys()))

    print(f"Rebuilding the exact KV training set to train {N_SHUFFLES} shuffled-label controls...")
    X_train, y_train_names = build_training_set()
    y_train_encoded = encoder.transform(y_train_names)
    print(f"  Total training samples: {len(X_train)}")

    print(f"Training {N_SHUFFLES} shuffled-label controls (identical protocol to Log 13)...")
    control_models = []
    for i in range(N_SHUFFLES):
        y_shuffled = np.random.permutation(y_train_encoded)
        control_models.append(train_cnn(X_train, y_shuffled))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_SHUFFLES} control models trained")

    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)

    results = []
    print(f"\nEvaluating certified model + all {N_SHUFFLES} controls on POWER-LAW attenuated signals across {len(EVAL_DEPTHS_CM)} depths...")
    for depth_cm in EVAL_DEPTHS_CM:
        depth_m = depth_cm / 100
        X_cell, y_cell_names = [], []
        for name in class_kpa_names:
            kpa = stiffness_classes[name]
            windows = generate_windows(kpa, depth_m, FREQ, n_per_class, attenuate_signal_powerlaw)
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

        results.append({
            "depth_cm": depth_cm, "acc_real": acc_real,
            "control_mean": control_accs.mean(), "p_value": p_value,
        })

    all_p_values = [r["p_value"] for r in results]
    significant_flags = bh_fdr_correction(all_p_values)
    for r, sig in zip(results, significant_flags):
        r["fdr_significant"] = sig

    print("\n" + "=" * 78)
    print("CERTIFIED POWER-LAW SENSITIVITY RESULTS (n=0.51, anchored 20Hz)")
    print("=" * 78)
    for r in results:
        print(f"  depth={r['depth_cm']:5.2f}cm  real={r['acc_real']:.2%}  "
              f"control={r['control_mean']:.2%}  p={r['p_value']:.4f}  sig={r['fdr_significant']}")

    floor_depth = None
    for r in results:
        if not r["fdr_significant"]:
            floor_depth = r["depth_cm"]
            break
    if floor_depth is not None:
        print(f"\nPower-law floor: {floor_depth:.2f}cm (KV certified floor was: {SEED121_CERTIFIED_FLOOR_CM:.2f}cm)")
    else:
        print(f"\nPower-law model never reached a certified floor within 0-15cm")
        print(f"(matches the pattern found for 20Hz under Kelvin-Voigt in Log 14)")

    import csv
    with open("../outputs/stage4/data/power_law_sensitivity_certified.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["depth_cm", "acc_real", "control_mean", "p_value", "fdr_significant"])
        for r in results:
            writer.writerow([r["depth_cm"], r["acc_real"], r["control_mean"], r["p_value"], r["fdr_significant"]])
    print("\nSaved certified results to outputs/stage4/data/power_law_sensitivity_certified.csv")

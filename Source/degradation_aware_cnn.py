# SOMA Signal Fidelity Investigation
# Stage 4 continuation -- Does degradation-aware training push the floor deeper?
# File: Source/degradation_aware_cnn.py
#
# --- FIX #1: corrected sensor noise floor -----------------------------------
# 67.03dB (1kHz, free-field microphone) -> 27.05dB (Yang et al. 2022,
# Biosensors PMC9312695 -- wearable skin-contact MEMS heart-sound sensor,
# 10-630Hz band, validated on a real subject). Full citation trail in
# floor_characterization.py.
#
# --- FIX #4-style: densified eval grid --------------------------------------
# The original 12-point 0-15cm grid (1.36cm spacing) is what produced the
# grid-artifact problem for 250Hz/500Hz elsewhere in this codebase. Kept the
# original 12 points for comparability, added fine points in the 3-8cm range
# where the floor is expected to land based on the basic (non-degradation-
# aware) CNN's corrected floor of 4.30cm.
#
# EXPECTATION GOING IN, stated before running: under the corrected noise,
# real signal at 100Hz is already far below the noise floor by 9-12cm
# (verified: SNR ratio 0.15 at 9cm, 0.03 at 12cm) -- two of this script's
# five TRAINING depths contain no recoverable information. This may limit
# how much degradation-aware training can push the floor past what the
# basic CNN already achieves (4.30cm), compared to how much it helped
# under the old, wrong noise model (5.45cm -> 8.18cm).

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED, NOMINAL_FREQ_HZ, TRAINING_DEPTHS_CM, BASE_EVAL_DEPTHS_CM, DENSIFIED_DEPTHS_CM, BASIC_CNN_CORRECTED_FLOOR_CM

torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_DEPTHS_CM = TRAINING_DEPTHS_CM
FREQ = NOMINAL_FREQ_HZ
N_SAMPLES_PER_DEPTH_PER_CLASS = 25

FINE_EVAL_DEPTHS_CM = DENSIFIED_DEPTHS_CM
EVAL_DEPTHS_CM = np.sort(np.unique(np.concatenate([BASE_EVAL_DEPTHS_CM, FINE_EVAL_DEPTHS_CM])))

N_SAMPLES_PER_CELL = 60
N_SHUFFLES = 200
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)


def generate_degraded_windows(kpa, depth_m, frequency_hz, n_needed, jitter_std=0.03):
    windows = []
    while len(windows) < n_needed:
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)
        noisy = attenuated + np.random.normal(0, SENSOR_NOISE_STD, size=attenuated.shape)
        rows = extract_raw_windows(noisy, time, jittered_hr)
        windows.extend(rows)
    return windows[:n_needed]


def build_degradation_aware_dataset():
    X, y = [], []
    for depth_cm in TRAIN_DEPTHS_CM:
        depth_m = depth_cm / 100
        for name, kpa in stiffness_classes.items():
            windows = generate_degraded_windows(kpa, depth_m, FREQ, N_SAMPLES_PER_DEPTH_PER_CLASS)
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
    print(f"Building degradation-aware training set (depths {TRAIN_DEPTHS_CM}cm at {FREQ}Hz)...")
    X_train, y_train_names = build_degradation_aware_dataset()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)
    print(f"  Total training samples: {len(X_train)}")

    print("Training the degradation-aware CNN...")
    model_real = train_cnn(X_train, y_train_encoded)

    print(f"Training {N_SHUFFLES} shuffled-label controls (same mixed-depth training data)...")
    control_models = []
    for i in range(N_SHUFFLES):
        y_shuffled = np.random.permutation(y_train_encoded)
        control_models.append(train_cnn(X_train, y_shuffled))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_SHUFFLES} control models trained")

    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)

    results = []
    print(f"\nEvaluating across {len(EVAL_DEPTHS_CM)} depths (0-15cm, densified 3-8cm) at {FREQ}Hz...")
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

        results.append({
            "depth_cm": depth_cm, "acc_real": acc_real,
            "control_mean": control_accs.mean(), "p_value": p_value,
        })

    all_p_values = [r["p_value"] for r in results]
    significant_flags = bh_fdr_correction(all_p_values)
    for r, sig in zip(results, significant_flags):
        r["fdr_significant"] = sig

    print("\n" + "="*70)
    print(f"DEGRADATION-AWARE CNN RESULTS (corrected noise; compare to basic CNN's {BASIC_CNN_CORRECTED_FLOOR_CM:.2f}cm floor)")
    print("="*70)
    for r in results:
        print(f"  depth={r['depth_cm']:5.2f}cm  real={r['acc_real']:.2%}  "
              f"control={r['control_mean']:.2%}  p={r['p_value']:.4f}  sig={r['fdr_significant']}")

    floor_depth = None
    for r in results:
        if not r["fdr_significant"]:
            floor_depth = r["depth_cm"]
            break
    if floor_depth is not None:
        print(f"\nDegradation-aware floor (corrected noise): {floor_depth:.2f}cm  "
              f"(basic clean-trained floor, corrected: {BASIC_CNN_CORRECTED_FLOOR_CM:.2f}cm; old uncorrected degradation-aware floor: 8.18cm)")
    else:
        print(f"\nDegradation-aware CNN never reached the floor within tested range")

    import pickle
    with open("results/stage4_degradation_aware_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved to results/stage4_degradation_aware_results.pkl")

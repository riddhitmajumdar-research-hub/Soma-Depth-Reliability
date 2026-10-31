# SOMA Signal Fidelity Investigation
# Stage 4 (CNN version) -- Floor Characterization Sweep, fully certified
# File: Source/floor_characterization_cnn.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import build_raw_dataset, SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED, BASE_EVAL_DEPTHS_CM

torch.manual_seed(SEED)
np.random.seed(SEED)

# --- FIX #4: grid resolution near the origin for 500Hz/250Hz ---------------
# (unchanged from before -- see prior version's comment for full rationale)
BASE_DEPTHS_CM = BASE_EVAL_DEPTHS_CM
FINE_NEAR_ORIGIN_CM = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.5, 1.8, 2.1]
DENSE_DEPTHS_CM = np.sort(np.unique(np.concatenate([BASE_DEPTHS_CM, FINE_NEAR_ORIGIN_CM])))

# --- FOLLOW-UP FIX: 100Hz densification around the borderline 5.45cm call --
# The first corrected run put 100Hz's floor at exactly 5.45cm with p=0.0597
# at that depth -- close enough to the 0.05 threshold after FDR correction
# that it's worth resolving properly rather than trusting the coarse
# 1.36cm-spaced grid. Densifying 4.09-6.82cm specifically, where the
# collapse (90%->83%->80%->71.67%->51.67%->36.67%->33.33%) actually happens.
FINE_100HZ_CM = [4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9, 6.1, 6.3, 6.5]
DENSE_100HZ_CM = np.sort(np.unique(np.concatenate([BASE_DEPTHS_CM, FINE_100HZ_CM])))

FREQUENCIES_HZ = [20, 100, 250, 500]
DEPTHS_BY_FREQ = {
    20: BASE_DEPTHS_CM,
    100: DENSE_100HZ_CM,
    250: DENSE_DEPTHS_CM,
    500: DENSE_DEPTHS_CM,
}

N_SAMPLES_PER_CELL = 60
N_SHUFFLES = 200

# --- FIX #1: sensor noise floor (see floor_characterization.py for full citation trail) ---
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
    print("Training the real CNN (clean signals, exactly as Stage 3c)...")
    X_clean, y_clean = build_raw_dataset()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_clean)
    class_names = list(encoder.classes_)
    model_real = train_cnn(X_clean, y_encoded)

    print(f"Training {N_SHUFFLES} shuffled-label control CNNs (this is the slow part)...")
    control_models = []
    for i in range(N_SHUFFLES):
        y_shuffled = np.random.permutation(y_encoded)
        control_models.append(train_cnn(X_clean, y_shuffled))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_SHUFFLES} control models trained")

    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)

    results = []
    total_cells = sum(len(DEPTHS_BY_FREQ[f]) for f in FREQUENCIES_HZ)
    print(f"\nRunning sweep: {total_cells} total (freq, depth) cells "
          f"x {N_SAMPLES_PER_CELL} samples/cell "
          f"(20Hz: {len(BASE_DEPTHS_CM)} depths; 100Hz: {len(DENSE_100HZ_CM)} depths, "
          f"densified 4-7cm; 250Hz/500Hz: {len(DENSE_DEPTHS_CM)} depths each, "
          f"densified near origin)...")
    for freq in FREQUENCIES_HZ:
        for depth_cm in DEPTHS_BY_FREQ[freq]:
            depth_m = depth_cm / 100
            X_cell = []
            y_cell_names = []
            for name in class_kpa_names:
                kpa = stiffness_classes[name]
                windows = generate_degraded_windows(kpa, depth_m, freq, n_per_class)
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
                "freq": freq, "depth_cm": depth_cm,
                "acc_real": acc_real,
                "control_mean": control_accs.mean(),
                "p_value": p_value,
            })
        print(f"  Completed frequency {freq}Hz ({len(DEPTHS_BY_FREQ[freq])} depths)")

    all_p_values = [r["p_value"] for r in results]
    significant_flags = bh_fdr_correction(all_p_values)
    for r, sig in zip(results, significant_flags):
        r["fdr_significant"] = sig

    print("\n" + "="*70)
    print("CNN FLOOR IDENTIFICATION")
    print("="*70)
    for freq in FREQUENCIES_HZ:
        freq_results = sorted([r for r in results if r["freq"] == freq], key=lambda r: r["depth_cm"])
        floor_depth = None
        for r in freq_results:
            if not r["fdr_significant"]:
                floor_depth = r["depth_cm"]
                break
        if floor_depth is not None:
            print(f"  {freq:4d}Hz  ->  floor at {floor_depth:.2f}cm")
        else:
            print(f"  {freq:4d}Hz  ->  never reaches the null within tested range")

    print("\nAccuracy by depth (100Hz, densified 4-7cm):")
    for r in sorted([r for r in results if r["freq"] == 100], key=lambda r: r["depth_cm"]):
        print(f"  depth={r['depth_cm']:5.2f}cm  real={r['acc_real']:.2%}  "
              f"control={r['control_mean']:.2%}  p={r['p_value']:.4f}  sig={r['fdr_significant']}")

    import pickle
    with open("results/stage4_cnn_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nSaved to results/stage4_cnn_results.pkl")

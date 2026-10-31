# SOMA Signal Fidelity Investigation
# Stage 4 -- 20Hz depth extension (resolving the open question)
# File: Source/extend_20hz.py

import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import build_raw_dataset, SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

DEPTHS_CM = np.linspace(15, 40, 10)
FREQ = 20
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
    print("Training the real CNN (clean signals, same as Stage 3c)...")
    X_clean, y_clean = build_raw_dataset()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_clean)
    model_real = train_cnn(X_clean, y_encoded)

    torch.save(model_real.state_dict(), "models/stage4_cnn_model.pt")
    with open("results/stage4_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    print("Saved trained model to models/stage4_cnn_model.pt and encoder to results/stage4_encoder.pkl")

    print(f"Training {N_SHUFFLES} shuffled-label control CNNs...")
    control_models = []
    for i in range(N_SHUFFLES):
        y_shuffled = np.random.permutation(y_encoded)
        control_models.append(train_cnn(X_clean, y_shuffled))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_SHUFFLES} control models trained")

    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)

    results = []
    print(f"\nExtending 20Hz sweep: {len(DEPTHS_CM)} depths from 15-40cm...")
    for depth_cm in DEPTHS_CM:
        depth_m = depth_cm / 100
        X_cell = []
        y_cell_names = []
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
            "freq": FREQ, "depth_cm": depth_cm,
            "acc_real": acc_real, "control_mean": control_accs.mean(), "p_value": p_value,
        })

    all_p_values = [r["p_value"] for r in results]
    significant_flags = bh_fdr_correction(all_p_values)
    for r, sig in zip(results, significant_flags):
        r["fdr_significant"] = sig

    print("\n20Hz extended results (15-40cm):")
    for r in results:
        print(f"  depth={r['depth_cm']:5.2f}cm  real={r['acc_real']:.2%}  "
              f"control={r['control_mean']:.2%}  p={r['p_value']:.4f}  sig={r['fdr_significant']}")

    floor_depth = None
    for r in results:
        if not r["fdr_significant"]:
            floor_depth = r["depth_cm"]
            break
    if floor_depth is not None:
        print(f"\n20Hz floor found at {floor_depth:.2f}cm")
    else:
        print(f"\n20Hz still hasn't reached the floor even by 40cm -- may need further extension")

    with open("results/stage4_20hz_extended.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved to results/stage4_20hz_extended.pkl")

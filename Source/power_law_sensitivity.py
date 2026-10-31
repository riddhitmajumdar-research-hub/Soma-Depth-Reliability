# Stage 4+ -- Power-Law Attenuation Sensitivity Test
# File: Source/power_law_sensitivity.py
#
# Re-evaluates the CERTIFIED degradation-aware CNN (loaded, not retrained)
# against signals attenuated with the power-law model instead of Kelvin-Voigt,
# across the same 0-15cm depth grid used to certify the 8.18cm floor.
# This isolates one question: how much does the certified floor depend on
# the Kelvin-Voigt alpha~omega^2 frequency-scaling assumption specifically?

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal as attenuate_signal_kv
from attenuation_model_powerlaw import attenuate_signal_powerlaw
from cnn_classifier import SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED, NOMINAL_FREQ_HZ, BASE_EVAL_DEPTHS_CM, DENSIFIED_DEPTHS_CM

torch.manual_seed(SEED)
np.random.seed(SEED)

FREQ = NOMINAL_FREQ_HZ
FINE_EVAL_DEPTHS_CM = DENSIFIED_DEPTHS_CM
EVAL_DEPTHS_CM = np.sort(np.unique(np.concatenate([BASE_EVAL_DEPTHS_CM, FINE_EVAL_DEPTHS_CM])))
N_SAMPLES_PER_CELL = 60
# FIX (post-audit): was 67.03dB, the pre-Fix-#1 value Log 18 corrected everywhere
# else. This script was never updated. Now matches the corrected 27.05dB used
# throughout the rest of the certified pipeline.
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)
MODEL_WEIGHTS_PATH = "../outputs/stage4/models/soma_certified_model_seed121_corrected_noise.pt"

class_labels = sorted(stiffness_classes.keys())


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


def evaluate_at_depths(model, attenuate_fn, label):
    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)
    encoder = LabelEncoder()
    encoder.fit(class_labels)

    results = []
    for depth_cm in EVAL_DEPTHS_CM:
        depth_m = depth_cm / 100
        X_cell, y_cell_names = [], []
        for name in class_kpa_names:
            kpa = stiffness_classes[name]
            windows = generate_windows(kpa, depth_m, FREQ, n_per_class, attenuate_fn)
            X_cell.extend(windows)
            y_cell_names.extend([name] * len(windows))
        X_cell = np.array(X_cell)
        y_cell_encoded = encoder.transform(y_cell_names)
        X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            preds = model(X_cell_t).argmax(dim=1).numpy()
        acc = np.mean(preds == y_cell_encoded)
        results.append({"depth_cm": depth_cm, "accuracy": acc})

    print(f"\n{label}:")
    for r in results:
        print(f"  depth={r['depth_cm']:5.2f}cm  acc={r['accuracy']:.2%}")
    return results


if __name__ == "__main__":
    print(f"Loading verified seed-121 model weights (one seed from the certified 8.48cm +/- 1.81cm distribution) from {MODEL_WEIGHTS_PATH}...")
    model = SimpleCNN(n_classes=3)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    model.eval()
    print("Certified model loaded (no retraining performed).")

    kv_results = evaluate_at_depths(model, attenuate_signal_kv, "KELVIN-VOIGT ATTENUATION (certified, matches Log 15)")
    pl_results = evaluate_at_depths(model, attenuate_signal_powerlaw, "POWER-LAW ATTENUATION (n=0.51, anchored at 20Hz)")

    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 70)
    print(f"{'Depth (cm)':>12s}{'KV accuracy':>15s}{'PL accuracy':>15s}{'Difference':>14s}")
    for r_kv, r_pl in zip(kv_results, pl_results):
        diff = r_pl["accuracy"] - r_kv["accuracy"]
        print(f"{r_kv['depth_cm']:>12.2f}{r_kv['accuracy']:>15.2%}{r_pl['accuracy']:>15.2%}{diff:>+14.2%}")

    import csv
    with open("../outputs/stage4/data/power_law_sensitivity.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["depth_cm", "kv_accuracy", "pl_accuracy"])
        for r_kv, r_pl in zip(kv_results, pl_results):
            writer.writerow([r_kv["depth_cm"], r_kv["accuracy"], r_pl["accuracy"]])
    print("\nSaved comparison data to outputs/stage4/data/power_law_sensitivity.csv")

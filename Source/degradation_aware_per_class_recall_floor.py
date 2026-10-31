# SOMA Signal Fidelity Investigation
# Part 3(b) -- per-class recall (healthy / significant_fibrosis / cirrhosis)
# at each tested depth for the certified degradation-aware CNN, with 95% CIs.
#
# SCOPE NOTE: only seed 121's weights were persisted from the 24-seed
# certified sweep (degradation_aware_final_sweep.py; README Log 20) --
# reproduced/verified in Source/retrain_seed121_verify.py. This script
# therefore reports the per-class breakdown for THAT one certified model
# (seed121, own floor = 8.18cm, SEED121_CERTIFIED_FLOOR_CM), not a
# per-class average across all 24 seeds -- the aggregate 8.48cm +/- 1.81cm
# is the 24-seed floor-depth distribution, a different quantity, and is
# used here only as the comparison point the question asks about.
#
# Method: for each depth, draw N_REPLICATES independent eval batches
# (fresh noise draw + fresh physics-model sampling each time, same
# protocol as the training/eval generator elsewhere in this codebase) and
# compute per-class recall on each replicate. Report mean recall and a
# 95% CI (normal approximation over replicate means) per class per depth.

import numpy as np
import torch
import csv
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows
from physics_model import generate_signal
from constants import SENSOR_SNR_DB, SEED, NOMINAL_FREQ_HZ, SEED121_CERTIFIED_FLOOR_CM

MODEL_PATH = "models/soma_certified_model_seed121_corrected_noise.pt"
FREQ = NOMINAL_FREQ_HZ
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)
N_PER_CLASS = 40
N_REPLICATES = 20
CSV_PATH = "logs/degradation_aware_per_class_recall_by_depth.csv"

# Depths spanning below/at/above both the single-seed floor (8.18cm) and
# the 24-seed aggregate mean floor (8.48cm), plus the safety gate (6.50cm).
EVAL_DEPTHS_CM = [5.0, 6.0, 6.5, 7.0, 7.5, 7.89, 8.0, 8.18, 8.48, 9.0, 10.0, 12.0]

class_names_sorted = sorted(stiffness_classes.keys())  # encoder's fit order
encoder = LabelEncoder()
encoder.fit(class_names_sorted)


def generate_windows(kpa, depth_m, frequency_hz, n_needed, rng, jitter_std=0.03):
    windows = []
    while len(windows) < n_needed:
        jittered_hr = 1.15 + rng.normal(0, jitter_std)
        time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)
        noisy = attenuated + rng.normal(0, SENSOR_NOISE_STD, size=attenuated.shape)
        rows = extract_raw_windows(noisy, time, jittered_hr)
        windows.extend(rows)
    return windows[:n_needed]


def eval_one_replicate(model, depth_cm, rng):
    depth_m = depth_cm / 100
    X, y_names = [], []
    for name in class_names_sorted:
        kpa = stiffness_classes[name]
        windows = generate_windows(kpa, depth_m, FREQ, N_PER_CLASS, rng)
        X.extend(windows)
        y_names.extend([name] * len(windows))
    X = np.array(X)
    y_enc = encoder.transform(y_names)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).numpy()

    per_class_recall = {}
    for name in class_names_sorted:
        cls_idx = encoder.transform([name])[0]
        mask = y_enc == cls_idx
        per_class_recall[name] = np.mean(preds[mask] == y_enc[mask])
    overall_acc = np.mean(preds == y_enc)
    return per_class_recall, overall_acc


def main():
    model = SimpleCNN(n_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    rows = []
    print(f"Certified seed-121 model, own floor={SEED121_CERTIFIED_FLOOR_CM}cm, "
          f"24-seed aggregate mean floor=8.48cm +/- 1.81cm")
    print(f"{N_REPLICATES} independent eval replicates x {N_PER_CLASS}/class per depth\n")

    for depth_cm in EVAL_DEPTHS_CM:
        recalls_by_class = {name: [] for name in class_names_sorted}
        overall_accs = []
        for rep in range(N_REPLICATES):
            rng = np.random.default_rng(1000 * int(depth_cm * 100) + rep)
            per_class_recall, overall_acc = eval_one_replicate(model, depth_cm, rng)
            for name in class_names_sorted:
                recalls_by_class[name].append(per_class_recall[name])
            overall_accs.append(overall_acc)

        print(f"depth={depth_cm:>5.2f}cm  overall_acc={np.mean(overall_accs):.4f}")
        row = {"depth_cm": depth_cm, "overall_acc_mean": np.mean(overall_accs),
               "overall_acc_std": np.std(overall_accs, ddof=1)}
        for name in class_names_sorted:
            vals = np.array(recalls_by_class[name])
            mean = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(N_REPLICATES)
            ci_lo, ci_hi = mean - 1.96 * se, mean + 1.96 * se
            print(f"    {name:>22}: recall={mean:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
            row[f"{name}_recall_mean"] = mean
            row[f"{name}_recall_ci_lo"] = max(0.0, ci_lo)
            row[f"{name}_recall_ci_hi"] = min(1.0, ci_hi)
        rows.append(row)

    fieldnames = ["depth_cm", "overall_acc_mean", "overall_acc_std"] + \
        [f"{name}_recall_{suf}" for name in class_names_sorted
         for suf in ("mean", "ci_lo", "ci_hi")]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {CSV_PATH}")

    # Flag: does any single class collapse to near-chance (~33%) at a
    # shallower depth than the aggregate 8.48cm figure implies?
    print("\n--- Class-specific floor check (vs 8.48cm aggregate) ---")
    for row in rows:
        if row["depth_cm"] >= 8.48:
            continue
        for name in class_names_sorted:
            if row[f"{name}_recall_mean"] < 0.45:
                print(f"  FLAG: {name} recall={row[f'{name}_recall_mean']:.3f} "
                      f"at depth={row['depth_cm']}cm (shallower than 8.48cm aggregate)")


if __name__ == "__main__":
    main()

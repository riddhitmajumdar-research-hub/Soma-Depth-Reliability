# SOMA Signal Fidelity Investigation
# Stage 4 continuation -- Does degradation-aware training push the floor deeper?
# CALIBRATION CAPTURE VARIANT -- adds softmax probability logging for Step 7 (UCE / reliability diagram)
# File: Source/degradation_aware_cnn.py (or degradation_aware_cnn_calibration.py)
#
# SUPERSEDED for any current/ongoing calibration work by
# degradation_aware_calibration_corrected.py (Log 23). Treat this file as a
# historical artifact, not actively maintained.

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED, NOMINAL_FREQ_HZ

torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_DEPTHS_CM = [0, 3, 6, 9, 12]
FREQ = NOMINAL_FREQ_HZ
N_SAMPLES_PER_DEPTH_PER_CLASS = 25
# Intentionally frozen at the pre-Fix-#4, pre-densification 12-point grid
# (NOT constants.py's BASE_EVAL_DEPTHS_CM/DENSIFIED_DEPTHS_CM union). This is
# the exact script that produced Log 16's original UCE numbers
# (0.2266/0.2838) -- changing the grid now would make this script neither a
# faithful reproduction of the original nor a clean match for its successor
# (degradation_aware_calibration_corrected.py, which uses the densified grid).
EVAL_DEPTHS_CM = np.linspace(0, 15, 12)
N_SAMPLES_PER_CELL = 60
N_SHUFFLES = 200
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)

# NEW: toggle for the expensive 200-shuffle floor-certification block.
# Not needed for calibration (Step 7) -- only re-enable if you also want
# to reproduce the FDR significance table in this same run.
RUN_SHUFFLE_CONTROLS = False


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
    class_names_ordered = list(encoder.classes_)  # order matches softmax column order
    print(f"  Total training samples: {len(X_train)}")
    print(f"  Class order (matches softmax columns): {class_names_ordered}")

    print("Training the degradation-aware CNN...")
    model_real = train_cnn(X_train, y_train_encoded)

    control_models = []
    if RUN_SHUFFLE_CONTROLS:
        print(f"Training {N_SHUFFLES} shuffled-label controls (same mixed-depth training data)...")
        for i in range(N_SHUFFLES):
            y_shuffled = np.random.permutation(y_train_encoded)
            control_models.append(train_cnn(X_train, y_shuffled))
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{N_SHUFFLES} control models trained")
    else:
        print("Skipping shuffle-control training (RUN_SHUFFLE_CONTROLS=False) -- calibration-only run.")

    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)

    results = []
    calibration_records = []  # NEW: per-sample rows for Step 7

    print(f"\nEvaluating across {len(EVAL_DEPTHS_CM)} depths (0-15cm) at {FREQ}Hz...")
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
            logits_real = model_real(X_cell_t)
            logits_real_np = logits_real.numpy()   # NEW (this retrain): raw pre-softmax logits, saved for real temperature scaling
            probs_real = torch.softmax(logits_real, dim=1).numpy()   # NEW: full probability vector
            preds_real = probs_real.argmax(axis=1)
        acc_real = np.mean(preds_real == y_cell_encoded)

        # NEW: log one row per individual test sample -- this is what Step 7 needs
        for i in range(len(y_cell_encoded)):
            calibration_records.append({
                "depth_cm": depth_cm,
                "true_label": class_names_ordered[y_cell_encoded[i]],
                "pred_label": class_names_ordered[preds_real[i]],
                **{f"prob_{cls}": probs_real[i, j] for j, cls in enumerate(class_names_ordered)},
                **{f"logit_{cls}": logits_real_np[i, j] for j, cls in enumerate(class_names_ordered)},
            })

        control_accs = []
        if RUN_SHUFFLE_CONTROLS:
            for model_c in control_models:
                with torch.no_grad():
                    preds_c = model_c(X_cell_t).argmax(dim=1).numpy()
                control_accs.append(np.mean(preds_c == y_cell_encoded))
        control_accs = np.array(control_accs) if control_accs else np.array([np.nan])

        p_value = ((np.sum(control_accs >= acc_real) + 1) / (N_SHUFFLES + 1)) if RUN_SHUFFLE_CONTROLS else np.nan

        results.append({
            "depth_cm": depth_cm, "acc_real": acc_real,
            "control_mean": control_accs.mean(), "p_value": p_value,
        })

    if RUN_SHUFFLE_CONTROLS:
        all_p_values = [r["p_value"] for r in results]
        significant_flags = bh_fdr_correction(all_p_values)
        for r, sig in zip(results, significant_flags):
            r["fdr_significant"] = sig

        print("\n" + "="*70)
        print("DEGRADATION-AWARE CNN RESULTS (compare directly to Log 13's 5.45cm floor)")
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
            print(f"\nDegradation-aware floor: {floor_depth:.2f}cm  (clean-trained floor was: 5.45cm)")
        else:
            print(f"\nDegradation-aware CNN never reached the floor within 0-15cm")

        import pickle
        with open("results/stage4_degradation_aware_results.pkl", "wb") as f:
            pickle.dump(results, f)
        print("Saved to results/stage4_degradation_aware_results.pkl")
    else:
        print("\n" + "="*70)
        print("PER-DEPTH ACCURACY (calibration-only run, no significance testing)")
        print("="*70)
        for r in results:
            print(f"  depth={r['depth_cm']:5.2f}cm  real_acc={r['acc_real']:.2%}")

    # NEW: save per-sample calibration data for Step 7 (reliability diagram + UCE)
    import csv
    csv_path = "stage4_calibration_data.csv"
    fieldnames = ["depth_cm", "true_label", "pred_label"] + [f"prob_{c}" for c in class_names_ordered] + [f"logit_{c}" for c in class_names_ordered]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(calibration_records)
    print(f"\nSaved {len(calibration_records)} per-sample calibration rows to {csv_path}")
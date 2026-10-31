# SOMA -- Calibration data regeneration, corrected
#
# The original calibration-capture script (commit f6b776f, degradation_aware_cnn.py's
# calibration variant) predates two fixes applied everywhere else in this project:
#   - Fix #1 (Log 18): sensor noise SNR 67.03dB -> 27.05dB (~100x harsher, band-matched)
#   - Fix #4 (Log 18): the coarse 12-point depth grid was densified in 3-8cm
# Log 19/20 re-certified the depth-floor accuracy/significance numbers under these
# fixes, but never re-ran calibration -- so Log 16's UCE/reliability-diagram numbers
# have been describing a model trained on noise physics the rest of the project
# no longer uses. This script regenerates that data correctly, in one clean run,
# and saves the trained weights + both probabilities and raw logits this time.
#
# Deliberately calibration-only: no shuffle-control training. Floor re-certification
# under corrected noise is already settled (Log 19: 4.30cm; Log 20: 8.48cm +/- 1.81cm
# degradation-aware) -- re-deriving that here would be redundant and would reintroduce
# the exact non-determinism problem that sent us down this path in the first place.

import csv
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED, NOMINAL_FREQ_HZ, TRAINING_DEPTHS_CM, BASE_EVAL_DEPTHS_CM, DENSIFIED_DEPTHS_CM

torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_DEPTHS_CM = TRAINING_DEPTHS_CM
FREQ = NOMINAL_FREQ_HZ
N_SAMPLES_PER_DEPTH_PER_CLASS = 25

FINE_EVAL_DEPTHS_CM = DENSIFIED_DEPTHS_CM
EVAL_DEPTHS_CM = np.sort(np.unique(np.concatenate([BASE_EVAL_DEPTHS_CM, FINE_EVAL_DEPTHS_CM])))

N_SAMPLES_PER_CELL = 60
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)

MODEL_SAVE_PATH = "models/soma_calibration_corrected_model_weights.pt"
CSV_PATH = "stage4_calibration_data_corrected.csv"


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


if __name__ == "__main__":
    print(f"Building degradation-aware training set (depths {TRAIN_DEPTHS_CM}cm at {FREQ}Hz)...")
    X_train, y_train_names = build_degradation_aware_dataset()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)
    class_names_ordered = list(encoder.classes_)
    print(f"  Total training samples: {len(X_train)}")
    print(f"  Class order (matches prob_/logit_ columns): {class_names_ordered}")
    print(f"  Sensor SNR: {SENSOR_SNR_DB} dB (corrected value)")
    print(f"  Eval grid: {len(EVAL_DEPTHS_CM)} depths (densified)")

    print("\nTraining the degradation-aware CNN...")
    model_real = train_cnn(X_train, y_train_encoded)

    torch.save(model_real.state_dict(), MODEL_SAVE_PATH)
    print(f"Saved model weights to {MODEL_SAVE_PATH}")

    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)

    calibration_records = []
    print(f"\nEvaluating across {len(EVAL_DEPTHS_CM)} depths at {FREQ}Hz...")
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
            logits_real_np = logits_real.numpy()
            probs_real = torch.softmax(logits_real, dim=1).numpy()
            preds_real = probs_real.argmax(axis=1)
        acc_real = np.mean(preds_real == y_cell_encoded)
        print(f"  depth={depth_cm:6.2f}cm  acc={acc_real:.2%}")

        for i in range(len(y_cell_encoded)):
            calibration_records.append({
                "depth_cm": depth_cm,
                "true_label": class_names_ordered[y_cell_encoded[i]],
                "pred_label": class_names_ordered[preds_real[i]],
                **{f"prob_{cls}": probs_real[i, j] for j, cls in enumerate(class_names_ordered)},
                **{f"logit_{cls}": logits_real_np[i, j] for j, cls in enumerate(class_names_ordered)},
            })

    fieldnames = (["depth_cm", "true_label", "pred_label"]
                  + [f"prob_{c}" for c in class_names_ordered]
                  + [f"logit_{c}" for c in class_names_ordered])
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(calibration_records)
    print(f"\nSaved {len(calibration_records)} per-sample calibration rows to {CSV_PATH}")

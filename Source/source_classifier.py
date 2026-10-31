# SOMA Signal Fidelity Investigation
# Stage 3 — Classifier (baseline, tested on clean signals)
# File: Source/classifier.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

from physics_model import generate_signal, stiffness_classes
from constants import SEED


def extract_pulses_with_features(signal, time, heart_rate_hz, tof, duration=3.0,
                                  window_size=100, sample_rate=500):
    """
    BUG FOUND AND FIXED: the previous version scanned the whole signal for
    threshold crossings to find pulses. But each pulse OSCILLATES through
    zero about 15 times over its 150ms duration (a 100Hz carrier wave) --
    so that approach detected a new "pulse" at every zero-crossing, not once
    per real heartbeat. It was measuring scattered points across the whole
    pulse decay, not the true arrival time.

    FIX: generate_signal() already tells us exactly when each pulse should
    arrive (beat_t + tof). Instead of rediscovering that blindly, we go
    directly to each known arrival time and extract one window there. This
    is legitimate for clean synthetic data -- Stage 4, with real noise, will
    need a more robust envelope-based detection method, but that's a
    separate, harder problem to solve when we get there.
    """
    beat_period = 1.0 / heart_rate_hz
    beat_times = np.arange(0, duration, beat_period)
    features_list = []

    for beat_t in beat_times:
        arrival_t = beat_t + tof
        start_idx = np.searchsorted(time, arrival_t)
        end_idx = start_idx + window_size
        if end_idx > len(signal):
            continue

        window = signal[start_idx:end_idx]
        peak_local_idx = np.argmax(np.abs(window))
        peak_idx_abs = start_idx + peak_local_idx
        peak_time_abs = time[peak_idx_abs]

        time_in_cycle = peak_time_abs % beat_period
        peak_amplitude = np.max(np.abs(window))
        energy = np.sum(window ** 2)

        features_list.append([time_in_cycle, peak_amplitude, energy])

    return features_list


def build_dataset(n_runs_per_class=30, jitter_std=0.03, seed=SEED):
    np.random.seed(seed)
    X = []
    y = []

    for name, kpa in stiffness_classes.items():
        for _ in range(n_runs_per_class):
            jittered_hr = 1.15 + np.random.normal(0, jitter_std)
            time, signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)

            feature_rows = extract_pulses_with_features(signal, time, jittered_hr, tof)
            for features in feature_rows:
                X.append(features)
                y.append(name)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    print("Building dataset from repeated signal generation...")
    X, y = build_dataset()
    print(f"  Total samples: {len(X)}")
    print(f"  Samples per class: {dict(zip(*np.unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples...")

    print("\n--- Diagnostic: time_in_cycle by class (should differ meaningfully) ---")
    for label in sorted(stiffness_classes.keys()):
        class_values = X[y == label][:, 0]
        print(f"  {label:22s} mean={class_values.mean()*1000:.3f}ms  "
              f"std={class_values.std()*1000:.3f}ms  "
              f"min={class_values.min()*1000:.3f}ms  max={class_values.max()*1000:.3f}ms")
    print("--- end diagnostic ---\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nBaseline classifier accuracy on CLEAN signals: {accuracy:.2%}")
    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    labels = sorted(stiffness_classes.keys())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"{'':22s}" + "".join(f"{l[:12]:>14s}" for l in labels))
    for i, label in enumerate(labels):
        print(f"{label:22s}" + "".join(f"{v:14d}" for v in cm[i]))

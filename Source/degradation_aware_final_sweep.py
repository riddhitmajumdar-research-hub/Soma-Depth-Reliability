# SOMA Signal Fidelity Investigation
# Stage 4 -- Degradation-aware CNN, FINAL seed sweep (higher sample count)
# File: Source/degradation_aware_final_sweep.py
#
# N_SAMPLES_PER_CELL raised from 60 to 300 (100/class) to tighten the
# per-depth accuracy margin of error from ~12pts to ~5.7pts -- this is what
# caused earlier spurious late "recoveries" in significance at deep,
# already-collapsed depths. Cheap change: control-model TRAINING cost (the
# expensive part) depends on training-set size, not eval-set size, so this
# barely changes runtime.
#
# 24 fresh seeds (101-124). Prints an aggregate summary automatically once
# all seeds finish -- mean/median/std/range and a recommended reportable
# range for the summary.

import numpy as np
import torch
import torch.nn as nn
import csv
import os
import time
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows

TRAIN_DEPTHS_CM = [0, 3, 6, 9, 12]
FREQ = 100
N_SAMPLES_PER_DEPTH_PER_CLASS = 25

BASE_EVAL_DEPTHS_CM = np.linspace(0, 15, 12)
FINE_EVAL_DEPTHS_CM = [3.5, 4.0, 4.3, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
EVAL_DEPTHS_CM = np.sort(np.unique(np.concatenate([BASE_EVAL_DEPTHS_CM, FINE_EVAL_DEPTHS_CM])))

N_SAMPLES_PER_CELL = 300   # raised from 60
N_SHUFFLES = 200
from constants import SENSOR_SNR_DB
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)

SEEDS_TO_RUN = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124]  # 24 seeds
CSV_PATH = "logs/degradation_aware_final_results.csv"


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


def run_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train_names = build_degradation_aware_dataset()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)
    model_real = train_cnn(X_train, y_train_encoded)

    # Persist this seed's trained weights -- previously only summary stats
    # were kept for all 24 seeds, meaning 23/24 had no reloadable model.
    os.makedirs("seed_weights", exist_ok=True)
    torch.save(model_real.state_dict(), f"seed_weights/degradation_aware_seed{seed}.pt")

    control_models = []
    for i in range(N_SHUFFLES):
        y_shuffled = np.random.permutation(y_train_encoded)
        control_models.append(train_cnn(X_train, y_shuffled))

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

    return {
        "seed": seed,
        "naive_floor_cm": floor_depth,
        "last_significant_depth_cm": last_significant_depth,
        "anomalous_recovery": anomalous_recovery,
    }


def print_final_summary(rows):
    floors = [r["naive_floor_cm"] for r in rows]
    n = len(floors)
    mean = sum(floors) / n
    variance = sum((x - mean) ** 2 for x in floors) / n if n > 1 else 0
    std = variance ** 0.5
    sorted_floors = sorted(floors)
    median = sorted_floors[n // 2] if n % 2 == 1 else (sorted_floors[n // 2 - 1] + sorted_floors[n // 2]) / 2
    n_anomalous = sum(1 for r in rows if r["anomalous_recovery"])

    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY -- {n} seeds, N_SAMPLES_PER_CELL={N_SAMPLES_PER_CELL}")
    print("=" * 70)
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(f"  seed={r['seed']:>4}  naive_floor={r['naive_floor_cm']:>5.2f}cm  "
              f"last_sig={r['last_significant_depth_cm']:>5.2f}cm  "
              f"anomalous={r['anomalous_recovery']}")
    print(f"\n  mean={mean:.2f}cm  median={median:.2f}cm  std={std:.2f}cm  "
          f"min={min(floors):.2f}cm  max={max(floors):.2f}cm")
    print(f"  seeds with anomalous late recovery: {n_anomalous}/{n}")
    print(f"\n  RECOMMENDED REPORTABLE RANGE: {mean - std:.1f}cm - {mean + std:.1f}cm "
          f"(mean +/- 1 std)")


if __name__ == "__main__":
    file_exists = os.path.exists(CSV_PATH)
    already_done = set()
    all_rows = []
    if file_exists:
        with open(CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                already_done.add(int(row["seed"]))
                all_rows.append({
                    "seed": int(row["seed"]),
                    "naive_floor_cm": float(row["naive_floor_cm"]),
                    "last_significant_depth_cm": float(row["last_significant_depth_cm"]),
                    "anomalous_recovery": row["anomalous_recovery"] == "True",
                })
        print(f"Found existing results for seeds: {sorted(already_done)}")

    fieldnames = ["seed", "naive_floor_cm", "last_significant_depth_cm", "anomalous_recovery"]
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for seed in SEEDS_TO_RUN:
            if seed in already_done:
                print(f"Seed {seed}: already done, skipping.")
                continue
            t0 = time.time()
            print(f"\n--- Seed {seed} starting ---")
            row = run_one_seed(seed)
            elapsed = time.time() - t0
            print(f"Seed {seed}: naive_floor={row['naive_floor_cm']:.2f}cm  "
                  f"last_sig={row['last_significant_depth_cm']}  "
                  f"anomalous_recovery={row['anomalous_recovery']}  "
                  f"({elapsed:.0f}s)")
            writer.writerow(row)
            f.flush()
            all_rows.append(row)

    print_final_summary(all_rows)
    print(f"\nAll done. Full data in {CSV_PATH}")

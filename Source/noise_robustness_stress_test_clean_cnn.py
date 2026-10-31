# Stage 4+ -- Sensor Noise Robustness Stress Test, CLEAN-TRAINED CNN (200-shuffle BH-FDR)
# File: Source/noise_robustness_stress_test_clean_cnn.py
#
# Companion to noise_robustness_stress_test.py, swapping ONLY the model
# weights: this run loads Source/models/stage4_cnn_model.pt, the clean-trained
# (non-degradation-aware) CNN from floor_characterization_cnn.py, instead of
# the degradation-aware certified seed-121 model. Everything else --
# SNR sweep points, depth grid, 200-shuffle BH-FDR protocol, per-class
# recall tracking -- is identical to noise_robustness_stress_test.py.
#
# WHY: the noise-robustness stress test on the degradation-aware model found
# healthy consistently the MOST robust class near the floor (not the LEAST,
# as Log 14/19 found for the clean-trained CNN under DEPTH stress) -- with
# significant fibrosis consistently the most fragile instead. Two candidate
# explanations: (a) this is a property of degradation-aware training
# specifically, or (b) it's a property of noise-type stress generally, as
# opposed to depth-type stress, independent of which model is tested. This
# script isolates that by running the identical noise-SNR-sweep protocol
# against the OTHER model (clean-trained) -- if the same
# fibrosis-fragile/healthy-robust pattern shows up here too, it points to
# (b); if this model instead reproduces Log 14/19's healthy-fragile pattern
# under noise stress, it points to (a).
#
# DELIBERATE METHODOLOGY FIX vs. noise_robustness_stress_test.py: that
# script's "ordering" check originally sampled the last FDR-significant
# depth (one step before the floor), not the floor depth itself -- a real
# off-by-one relative to what Log 14/19 actually measured (the confusion
# pattern AT the floor). This script checks AT the floor depth directly,
# and reports one row below/at/above it, matching how the finding was
# manually re-verified from noise_robustness_stress_test.py's CSV afterward.
#
# Usage: python3 noise_robustness_stress_test_clean_cnn.py [n_shuffles]
#   Defaults to the full N_SHUFFLES=200 certified run. Pass a small number
#   (e.g. 10) for a quick dry run to sanity-check the pipeline end to end
#   before committing to the full run.

import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows
from constants import (
    SENSOR_SNR_DB, SENSOR_SNR_DB_WORST_CASE, SEED, NOMINAL_FREQ_HZ,
    TRAINING_DEPTHS_CM, BASE_EVAL_DEPTHS_CM, DENSIFIED_DEPTHS_CM,
    BASIC_CNN_CORRECTED_FLOOR_CM,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

TRAIN_DEPTHS_CM = TRAINING_DEPTHS_CM
FREQ = NOMINAL_FREQ_HZ
N_SAMPLES_PER_DEPTH_PER_CLASS = 25
FINE_EVAL_DEPTHS_CM = DENSIFIED_DEPTHS_CM
EVAL_DEPTHS_CM = np.sort(np.unique(np.concatenate([BASE_EVAL_DEPTHS_CM, FINE_EVAL_DEPTHS_CM])))
N_SAMPLES_PER_CELL = 60
N_SHUFFLES = int(sys.argv[1]) if len(sys.argv) > 1 else 200

# Training always uses the standard, primary, currently-certified SNR --
# only the EVALUATION-time SNR varies across the sweep below.
TRAINING_SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)

# Sweep from the primary/current, best-documented SNR down to the weakest of
# the 3 corroborating papers (Fix #1 / Log 18), 5 evenly spaced points.
# Identical sweep to noise_robustness_stress_test.py.
SNR_SWEEP_DB = np.linspace(SENSOR_SNR_DB, SENSOR_SNR_DB_WORST_CASE, 5)

# ONLY DIFFERENCE from noise_robustness_stress_test.py: the clean-trained
# (non-degradation-aware) CNN, not the degradation-aware certified model.
MODEL_WEIGHTS_PATH = "models/stage4_cnn_model.pt"
CSV_PATH = "../outputs/stage4/data/noise_robustness_stress_test_clean_cnn.csv"
FIGURE_PATH = "../outputs/stage4/figures/noise_robustness_stress_test_clean_cnn.png"


def generate_windows(kpa, depth_m, frequency_hz, n_needed, noise_std, jitter_std=0.03):
    windows = []
    while len(windows) < n_needed:
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)
        noisy = attenuated + np.random.normal(0, noise_std, size=attenuated.shape)
        rows = extract_raw_windows(noisy, time, jittered_hr)
        windows.extend(rows)
    return windows[:n_needed]


def build_training_set():
    # Matches degradation_aware_cnn.py / power_law_sensitivity_certified.py /
    # noise_robustness_stress_test.py exactly -- KV attenuation, same
    # depths/counts, standard training SNR. NOTE: this builds a
    # degradation-aware-style training set for the SHUFFLED CONTROLS only
    # (the null distribution) -- the real model being tested (clean-trained
    # CNN) is loaded from disk, never retrained, regardless of what its
    # controls are trained on. The controls' job is just to characterize
    # chance-level performance on this same evaluation data.
    X, y = [], []
    for depth_cm in TRAIN_DEPTHS_CM:
        depth_m = depth_cm / 100
        for name, kpa in stiffness_classes.items():
            windows = generate_windows(kpa, depth_m, FREQ, N_SAMPLES_PER_DEPTH_PER_CLASS, TRAINING_SENSOR_NOISE_STD)
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
    print(f"Loading clean-trained (non-degradation-aware) CNN weights from {MODEL_WEIGHTS_PATH}...")
    model_real = SimpleCNN(n_classes=3)
    model_real.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    model_real.eval()
    print("Clean-trained model loaded (no retraining).\n")

    encoder = LabelEncoder()
    class_kpa_names = sorted(stiffness_classes.keys())
    encoder.fit(class_kpa_names)

    print(f"Rebuilding the standard corrected-noise ({SENSOR_SNR_DB}dB) KV training set to train {N_SHUFFLES} shuffled-label controls...")
    X_train, y_train_names = build_training_set()
    y_train_encoded = encoder.transform(y_train_names)
    print(f"  Total training samples: {len(X_train)}")

    print(f"Training {N_SHUFFLES} shuffled-label controls (identical protocol to Log 13/17)...")
    control_models = []
    for i in range(N_SHUFFLES):
        y_shuffled = np.random.permutation(y_train_encoded)
        control_models.append(train_cnn(X_train, y_shuffled))
        if (i + 1) % 20 == 0 or (i + 1) == N_SHUFFLES:
            print(f"  {i + 1}/{N_SHUFFLES} control models trained")

    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)

    all_results = []
    print(f"\nSNR sweep (dB): {[f'{s:.2f}' for s in SNR_SWEEP_DB]}")
    for snr_db in SNR_SWEEP_DB:
        noise_std = 10 ** (-snr_db / 20)
        print(f"\nEvaluating clean-trained model + all {N_SHUFFLES} controls at SNR={snr_db:.2f}dB across {len(EVAL_DEPTHS_CM)} depths...")
        for depth_cm in EVAL_DEPTHS_CM:
            depth_m = depth_cm / 100
            X_cell, y_cell_names = [], []
            for name in class_kpa_names:
                kpa = stiffness_classes[name]
                windows = generate_windows(kpa, depth_m, FREQ, n_per_class, noise_std)
                X_cell.extend(windows)
                y_cell_names.extend([name] * len(windows))
            X_cell = np.array(X_cell)
            y_cell_encoded = encoder.transform(y_cell_names)
            X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)

            with torch.no_grad():
                preds_real = model_real(X_cell_t).argmax(dim=1).numpy()
            acc_real = np.mean(preds_real == y_cell_encoded)

            per_class_recall = {}
            for cls_idx, cls_name in enumerate(encoder.classes_):
                cls_mask = y_cell_encoded == cls_idx
                per_class_recall[cls_name] = (
                    np.mean(preds_real[cls_mask] == y_cell_encoded[cls_mask]) if cls_mask.sum() > 0 else np.nan
                )

            control_accs = []
            for model_c in control_models:
                with torch.no_grad():
                    preds_c = model_c(X_cell_t).argmax(dim=1).numpy()
                control_accs.append(np.mean(preds_c == y_cell_encoded))
            control_accs = np.array(control_accs)

            p_value = (np.sum(control_accs >= acc_real) + 1) / (N_SHUFFLES + 1)

            all_results.append({
                "snr_db": snr_db, "depth_cm": depth_cm, "acc_real": acc_real,
                "control_mean": control_accs.mean(), "p_value": p_value,
                "recall_healthy": per_class_recall.get("healthy", np.nan),
                "recall_significant_fibrosis": per_class_recall.get("significant_fibrosis", np.nan),
                "recall_cirrhosis": per_class_recall.get("cirrhosis", np.nan),
            })
        print(f"  depth={EVAL_DEPTHS_CM[-1]:5.2f}cm  real={all_results[-1]['acc_real']:.2%}  control={all_results[-1]['control_mean']:.2%}")

    # BH-FDR correction applied PER SNR LEVEL.
    for snr_db in SNR_SWEEP_DB:
        subset = [r for r in all_results if r["snr_db"] == snr_db]
        sig_flags = bh_fdr_correction([r["p_value"] for r in subset])
        for r, sig in zip(subset, sig_flags):
            r["fdr_significant"] = bool(sig)

    print("\n" + "=" * 78)
    print(f"NOISE ROBUSTNESS STRESS TEST -- clean-trained CNN, {N_SHUFFLES}-shuffle BH-FDR")
    print(f"Primary/current SNR: {SENSOR_SNR_DB}dB (Fix #1 / Log 18) -- Worst corroborating paper: {SENSOR_SNR_DB_WORST_CASE}dB")
    print("=" * 78)
    print(f"{'SNR (dB)':>10} {'floor (cm)':>12} {'most fragile class at floor':>30}")

    floors_by_snr = []
    neighbor_tables = []
    for snr_db in SNR_SWEEP_DB:
        subset = sorted([r for r in all_results if r["snr_db"] == snr_db], key=lambda r: r["depth_cm"])
        floor_idx = next((i for i, r in enumerate(subset) if not r["fdr_significant"]), None)
        floor_depth = subset[floor_idx]["depth_cm"] if floor_idx is not None else None
        floors_by_snr.append(floor_depth)

        if floor_idx is not None:
            # Check AT the floor depth itself, not the last-significant depth
            # before it -- matches how Log 14/19 actually measured the
            # confusion pattern, and how this finding was manually
            # re-verified from noise_robustness_stress_test.py's CSV.
            floor_row = subset[floor_idx]
            recalls = {
                "healthy": floor_row["recall_healthy"],
                "significant_fibrosis": floor_row["recall_significant_fibrosis"],
                "cirrhosis": floor_row["recall_cirrhosis"],
            }
            most_fragile = min(recalls, key=recalls.get)
            neighbor_tables.append((snr_db, floor_idx, subset))
        else:
            most_fragile = "n/a (no floor reached)"
            neighbor_tables.append((snr_db, None, subset))

        floor_str = f"{floor_depth:.2f}" if floor_depth is not None else "not reached"
        print(f"{snr_db:10.2f} {floor_str:>12} {most_fragile:>30}")

    valid_floors = [f for f in floors_by_snr if f is not None]
    print(f"\nFloor progression (best to worst SNR): {[f'{f:.2f}' if f is not None else 'none' for f in floors_by_snr]}cm")
    if len(valid_floors) >= 2:
        monotonic = all(valid_floors[i] <= valid_floors[i + 1] for i in range(len(valid_floors) - 1))
        print(f"Floor shrinks or holds monotonically as SNR worsens (physically sensible): {monotonic}")
        if not monotonic:
            print("NON-MONOTONIC -- flag for review.")
    else:
        print("Fewer than 2 SNR levels reached a certified floor -- cannot assess monotonicity.")

    # Neighbor-row report: one depth below / at / above the floor, per SNR.
    print("\n" + "-" * 78)
    print("NEIGHBOR-ROW PER-CLASS RECALL (one depth below / at floor / one above)")
    print("-" * 78)
    for snr_db, floor_idx, subset in neighbor_tables:
        print(f"\nSNR={snr_db:.2f}dB:")
        if floor_idx is None:
            print("  (no floor reached within tested depths)")
            continue
        for i in range(max(0, floor_idx - 1), min(len(subset), floor_idx + 2)):
            r = subset[i]
            tag = "  <-- FLOOR" if i == floor_idx else ("  (below)" if i < floor_idx else "  (above)")
            print(f"  depth={r['depth_cm']:6.2f}cm  healthy={r['recall_healthy']:.2f}  "
                  f"fib={r['recall_significant_fibrosis']:.2f}  cirr={r['recall_cirrhosis']:.2f}  "
                  f"sig={r['fdr_significant']}{tag}")

    fieldnames = ["snr_db", "depth_cm", "acc_real", "control_mean", "p_value", "fdr_significant",
                  "recall_healthy", "recall_significant_fibrosis", "recall_cirrhosis"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved {len(all_results)} rows to {CSV_PATH}")

    # ============================================================
    # Figure: same two-panel layout as noise_robustness_stress_test.py,
    # reference line uses BASIC_CNN_CORRECTED_FLOOR_CM (this model's own
    # certified floor, Log 19) instead of the degradation-aware seed-121
    # floor -- the correct comparison point for THIS model.
    # ============================================================
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.5))

    cmap = plt.get_cmap("viridis")
    for i, snr_db in enumerate(SNR_SWEEP_DB):
        subset = [r for r in all_results if r["snr_db"] == snr_db]
        depths = [r["depth_cm"] for r in subset]
        accs = [r["acc_real"] for r in subset]
        sig = [r["fdr_significant"] for r in subset]
        color = cmap(i / max(len(SNR_SWEEP_DB) - 1, 1))
        ax_a.plot(depths, accs, color=color, linewidth=1.6, zorder=2, label=f"{snr_db:.1f}dB")
        for d, a, s in zip(depths, accs, sig):
            ax_a.scatter([d], [a], s=45, facecolor=color if s else "white", edgecolor=color, linewidth=1.3, zorder=3)

    ax_a.axhspan(0.30, 0.37, color="#999999", alpha=0.18, label="Chance level (~1/3)")
    ax_a.axvline(BASIC_CNN_CORRECTED_FLOOR_CM, color="#111111", linestyle=":", linewidth=1.2, alpha=0.6)
    ax_a.set_xlabel("Depth (cm)", fontsize=10.5, fontweight="medium")
    ax_a.set_ylabel("Certified accuracy", fontsize=10.5, fontweight="medium")
    ax_a.set_title("A.  Certified Accuracy vs. Depth Across SNR\n(clean-trained CNN; filled = FDR-significant, hollow = not)", fontsize=11, fontweight="bold", loc="left")
    ax_a.set_ylim(0.2, 1.05)
    ax_a.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9, ncol=2)

    plotted_floors = [f if f is not None else np.nan for f in floors_by_snr]
    ax_b.plot(SNR_SWEEP_DB, plotted_floors, color="#D95F02", linewidth=2, marker="o", markersize=7, zorder=2)
    ax_b.axhline(BASIC_CNN_CORRECTED_FLOOR_CM, color="#111111", linestyle="--", linewidth=1.2, alpha=0.6,
                 label=f"Primary/current floor ({BASIC_CNN_CORRECTED_FLOOR_CM:.2f}cm @ {SENSOR_SNR_DB:.2f}dB)")
    ax_b.invert_xaxis()
    ax_b.set_xlabel("Sensor SNR (dB) -- degrading left to right", fontsize=10.5, fontweight="medium")
    ax_b.set_ylabel("Certified depth floor (cm)", fontsize=10.5, fontweight="medium")
    ax_b.set_title("B.  Clean-Trained CNN's Floor vs.\nReal Sensor-to-Sensor SNR Variability", fontsize=11, fontweight="bold", loc="left")
    ax_b.legend(loc="best", fontsize=8, frameon=True, framealpha=0.9)

    fig.suptitle("SOMA — Sensor Noise Robustness Stress Test, Clean-Trained CNN (13.58-27.05dB, Fix #1's own corroborating range)",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.text(0.5, -0.05,
             "Clean-trained (non-degradation-aware) CNN, never retrained -- only evaluation-time sensor SNR varies. Same protocol as\n"
             "noise_robustness_stress_test.py, swapping only the model weights, to isolate model-type from noise-type effects.",
             ha="center", fontsize=8, color="#555555")

    plt.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight", pad_inches=0.35)
    print(f"\nSaved figure to {FIGURE_PATH}")

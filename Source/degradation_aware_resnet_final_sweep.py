# SOMA Signal Fidelity Investigation
# Stage 4 -- Capacity-confound retest: ResNet1D through the certified
# degradation-aware pipeline (24-seed sweep), directly comparable to
# degradation_aware_final_sweep.py's SimpleCNN result (8.48cm +/- 1.81cm).
# File: Source/degradation_aware_resnet_final_sweep.py
#
# WHY THIS SCRIPT EXISTS: the certified 8.48cm +/- 1.81cm depth-reliability
# floor was established using SimpleCNN (2 conv layers, no batch norm, no
# residual connections). The open question this script answers: is 8.48cm
# a physical limit of the signal itself, or just the ceiling of a small
# model? ResNet1D (batch norm + residual connections, the same architecture
# Log 08 found statistically indistinguishable from SimpleCNN on CLEAN
# signals, 95% CI -0.03 to +0.08) has never been run through the
# degradation-aware / noise-corrected / 24-seed certification pipeline.
# This script does that, changing ONLY the model class -- everything else
# (training depths, noise, sample counts, shuffle-control protocol, BH-FDR)
# is identical to degradation_aware_final_sweep.py.
#
# TWO DESIGN DECISIONS, made via two pilot dry runs before this full run,
# not assumed or tuned post-hoc against the result:
#
#   1. EPOCH BUDGET = 150, not 60. A first pilot matched SimpleCNN's exact
#      fixed-epoch count (60) and produced near-total training collapse
#      (4/5 shuffled controls never escaped 5 retries; the real model's
#      own "successful" run was weak and non-monotonic, peaking at 51-58%).
#      Matching epoch COUNT to SimpleCNN was an unexamined assumption, not
#      a like-for-like comparison requirement -- a deeper, residual, batch-normed
#      architecture genuinely needs more full-batch update steps to
#      converge than a 2-layer CNN does. A second pilot at 150 epochs
#      produced a clean, monotonically-declining accuracy curve (peak
#      71.67% at depth 0, 0 collapse events, 0 upward jumps >10 points)
#      landing almost exactly on SimpleCNN's own degradation-aware depth-0
#      accuracy (70%, Log 15) -- independent cross-model corroboration
#      this budget is an even, working setup for both architectures, not one
#      favoring either. 150 is locked here BEFORE this full run, decided
#      by that pilot's convergence behavior, not by looking at this run's
#      own results afterward.
#
#   2. COLLAPSE-RETRY SCOPE = real model only, NOT shuffled-label controls.
#      Log 11 built the restart-on-collapse safeguard (retry a fresh init,
#      up to 5 attempts, if post-training accuracy reads as chance) to stop
#      a GOOD, correctly-labeled model from spuriously reading as bad due
#      to an unlucky initialization -- ResNet1D + full-batch gradient
#      descent is known (Log 11) to land in a dead region on a meaningful
#      fraction of random inits. That logic has no legitimate role on a
#      shuffled-label control, which is SUPPOSED to sit near chance:
#      retrying a control until it clears the collapse threshold would
#      push it toward memorizing randomized labels, biasing the null
#      distribution in a direction that can't be cleanly characterized or
#      corrected for afterward. Controls here train ONCE, as-is, whatever
#      the result -- a control landing near chance is a valid, expected
#      null observation, not a failure needing correction.
#
# TRAINING-ACCURACY-VS-TEST-ACCURACY NOTE (confirmed in the second pilot,
# not assumed): shuffled-label controls frequently memorize their own
# training set to fairly high TRAINING accuracy (52-95% observed in the
# pilot, with only 375 training samples and 150 full-batch epochs) --
# that's expected and harmless. What actually matters is TEST-time accuracy
# on the held-out depth-sweep evaluation cells, which sat at chance
# (~30-35%) for every control in the pilot regardless of training-set
# memorization. This script's per-seed sanity check (below) verifies TEST
# accuracy, not training accuracy, for exactly this reason.

import numpy as np
import torch
import torch.nn as nn
import csv
import os
import time
from sklearn.preprocessing import LabelEncoder

from physics_model import stiffness_classes
from resnet_classifier import ResNet1D
from degradation_aware_final_sweep import (
    build_degradation_aware_dataset,
    generate_degraded_windows,
    bh_fdr_correction,
    EVAL_DEPTHS_CM,
    FREQ,
)

N_SAMPLES_PER_CELL = 300   # matches SimpleCNN's final sweep (100/class)
N_SHUFFLES = 200
N_EPOCHS = 150             # REVISED from SimpleCNN's 60 -- see header
COLLAPSE_THRESHOLD = 0.5   # sourced from Log 11 exactly
MAX_RESTARTS = 5           # sourced from Log 11 exactly -- REAL MODEL ONLY

# Loose sanity band around chance (1/3): control test accuracy averaged
# across the whole depth sweep should land close to 33.33%. This is the
# same "control mean sits near chance" check used for every certified
# result in this project (Logs 12, 13, 17, 19, 20, 21) -- flagged per seed,
# not silently assumed.
CONTROL_SANITY_LOW = 0.25
CONTROL_SANITY_HIGH = 0.42

SEEDS_TO_RUN = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
                121, 122, 123, 124]  # 24 seeds, identical to SimpleCNN sweep

CSV_PATH = "logs/degradation_aware_resnet_final_results.csv"
SEED_WEIGHTS_DIR = "seed_weights_resnet"


def train_resnet_plain(X, y_encoded, n_classes=3, n_epochs=N_EPOCHS):
    """Shuffled-label controls only. Trains once, full-batch, fixed
    epochs. No retry -- a control landing near chance is the expected,
    valid null observation, not a failure."""
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_encoded, dtype=torch.long)
    crit = nn.CrossEntropyLoss()

    model = ResNet1D(n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        opt.step()
    model.eval()
    return model


def train_resnet_with_retry(X, y_encoded, n_classes=3, n_epochs=N_EPOCHS):
    """Real (correctly-labeled) model only. Same fixed-epoch, full-batch
    training, plus Log 11's restart-on-collapse safeguard: if final
    training accuracy reads as chance, treat it as a bad initialization
    and retry with a fresh init (up to 5 attempts). Returns the model plus
    the number of collapse events observed and which attempt succeeded."""
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_encoded, dtype=torch.long)
    crit = nn.CrossEntropyLoss()

    collapse_count = 0
    for attempt in range(MAX_RESTARTS):
        model = ResNet1D(n_classes=n_classes)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(n_epochs):
            model.train()
            opt.zero_grad()
            loss = crit(model(X_t), y_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            train_acc_check = (model(X_t).argmax(dim=1) == y_t).float().mean().item()

        if train_acc_check > COLLAPSE_THRESHOLD:
            return model, collapse_count, attempt
        else:
            collapse_count += 1

    # Exhausted retries -- return whatever we have, flagged via collapse_count
    # == MAX_RESTARTS and escaped_on_attempt == -1 (never escaped).
    return model, collapse_count, -1


def run_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train_names = build_degradation_aware_dataset()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)

    model_real, real_collapse_count, real_escaped_on_attempt = train_resnet_with_retry(
        X_train, y_train_encoded
    )

    os.makedirs(SEED_WEIGHTS_DIR, exist_ok=True)
    torch.save(model_real.state_dict(),
               f"{SEED_WEIGHTS_DIR}/degradation_aware_resnet_seed{seed}.pt")

    control_models = []
    for i in range(N_SHUFFLES):
        y_shuffled = np.random.permutation(y_train_encoded)
        control_models.append(train_resnet_plain(X_train, y_shuffled))

    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)

    results = []
    all_control_test_accs = []  # every control's test accuracy at every depth,
                                 # pooled, for the per-seed sanity check
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
        all_control_test_accs.extend(control_accs.tolist())

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

    control_test_mean_overall = float(np.mean(all_control_test_accs))
    control_sanity_ok = CONTROL_SANITY_LOW <= control_test_mean_overall <= CONTROL_SANITY_HIGH

    return {
        "seed": seed,
        "naive_floor_cm": floor_depth,
        "last_significant_depth_cm": last_significant_depth,
        "anomalous_recovery": anomalous_recovery,
        "real_model_collapse_count": real_collapse_count,
        "real_model_escaped_on_attempt": real_escaped_on_attempt,  # -1 = never escaped
        "control_test_mean_overall": control_test_mean_overall,
        "control_sanity_ok": control_sanity_ok,
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
    n_sanity_failed = sum(1 for r in rows if not r["control_sanity_ok"])
    total_collapses = sum(r["real_model_collapse_count"] for r in rows)
    n_never_escaped = sum(1 for r in rows if r["real_model_escaped_on_attempt"] == -1)

    print("\n" + "=" * 70)
    print(f"FINAL SUMMARY (ResNet1D) -- {n} seeds, N_SAMPLES_PER_CELL={N_SAMPLES_PER_CELL}, "
          f"N_EPOCHS={N_EPOCHS}")
    print("=" * 70)
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(f"  seed={r['seed']:>4}  naive_floor={r['naive_floor_cm']:>5.2f}cm  "
              f"last_sig={r['last_significant_depth_cm']:>5.2f}cm  "
              f"anomalous={r['anomalous_recovery']}  "
              f"collapses={r['real_model_collapse_count']}  "
              f"escaped_on_attempt={r['real_model_escaped_on_attempt']}  "
              f"control_mean={r['control_test_mean_overall']:.2%}  "
              f"sanity_ok={r['control_sanity_ok']}")
    print(f"\n  mean={mean:.2f}cm  median={median:.2f}cm  std={std:.2f}cm  "
          f"min={min(floors):.2f}cm  max={max(floors):.2f}cm")
    print(f"  seeds with anomalous late recovery: {n_anomalous}/{n}")
    print(f"  seeds with FAILED control sanity check: {n_sanity_failed}/{n}")
    print(f"  real-model collapse events, total across all {n} seeds: {total_collapses}")
    print(f"  seeds where real model NEVER escaped collapse after {MAX_RESTARTS} retries: {n_never_escaped}")
    print(f"\n  RECOMMENDED REPORTABLE RANGE: {mean - std:.1f}cm - {mean + std:.1f}cm (mean +/- 1 std)")


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
                    "real_model_collapse_count": int(row["real_model_collapse_count"]),
                    "real_model_escaped_on_attempt": int(row["real_model_escaped_on_attempt"]),
                    "control_test_mean_overall": float(row["control_test_mean_overall"]),
                    "control_sanity_ok": row["control_sanity_ok"] == "True",
                })
        print(f"Found existing results for seeds: {sorted(already_done)}")

    fieldnames = ["seed", "naive_floor_cm", "last_significant_depth_cm", "anomalous_recovery",
                  "real_model_collapse_count", "real_model_escaped_on_attempt",
                  "control_test_mean_overall", "control_sanity_ok"]
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for seed in SEEDS_TO_RUN:
            if seed in already_done:
                print(f"Seed {seed}: already done, skipping.")
                continue
            t0 = time.time()
            print(f"\n--- Seed {seed} starting ---", flush=True)
            row = run_one_seed(seed)
            elapsed = time.time() - t0
            sanity_note = "OK" if row["control_sanity_ok"] else "*** SANITY CHECK FAILED ***"
            print(f"Seed {seed}: naive_floor={row['naive_floor_cm']:.2f}cm  "
                  f"last_sig={row['last_significant_depth_cm']}  "
                  f"anomalous_recovery={row['anomalous_recovery']}  "
                  f"collapses={row['real_model_collapse_count']}  "
                  f"escaped_on_attempt={row['real_model_escaped_on_attempt']}  "
                  f"control_test_mean={row['control_test_mean_overall']:.2%} [{sanity_note}]  "
                  f"({elapsed:.0f}s)", flush=True)
            writer.writerow(row)
            f.flush()  # checkpoint immediately -- a crash at seed 19 costs nothing from 1-18
            all_rows.append(row)

    print_final_summary(all_rows)
    print(f"\nAll done. Full data in {CSV_PATH}")

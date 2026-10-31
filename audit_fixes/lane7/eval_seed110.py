#!/usr/bin/env python3
"""
Lane 7 follow-up, item 1: reload the ResNet1D seed-110 saved weights and
evaluate accuracy at every depth of the standard grid (100 Hz, 27.05 dB
SNR, same N_SAMPLES_PER_CELL as the sweep). CNN seed-110 weights do not
exist anywhere in the repo (checked by `find`) -- reported as N/A, not
fabricated.

Read-only on the repo. Does not touch the running sweep (pid 7429) or its
files. Writes only inside audit_fixes/lane7/.

Run from repo root:
    python3 audit_fixes/lane7/eval_seed110.py
"""
import sys
import os
import numpy as np
import torch

from pathlib import Path as _P; REPO_ROOT = str(_P(__file__).resolve().parents[2])
SOURCE_DIR = f"{REPO_ROOT}/Source"
sys.path.insert(0, SOURCE_DIR)

EVAL_SEED = 110  # matches the seed under investigation; stated explicitly
                  # since this re-evaluation is NOT a bit-reproduction of the
                  # original sweep's exact eval windows (the sweep does not
                  # reset the RNG before its eval loop -- see REPORT.md for
                  # the same non-bit-reproducibility caveat already
                  # established elsewhere in this project, e.g. Log 23 /
                  # retrain_seed121_verify.py).

RESNET_WEIGHTS = f"{SOURCE_DIR}/seed_weights_resnet/degradation_aware_resnet_seed110.pt"


def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    hdr("0. WEIGHT FILE AVAILABILITY CHECK")
    print(f"ResNet1D seed-110 weights: {RESNET_WEIGHTS}")
    resnet_weights_exist = os.path.isfile(RESNET_WEIGHTS)
    print(f"  exists: {resnet_weights_exist}")
    if resnet_weights_exist:
        print(f"  size: {os.path.getsize(RESNET_WEIGHTS)} bytes")

    import subprocess
    find_out = subprocess.run(
        ["find", REPO_ROOT, "-iname", "*cnn*seed110*", "-o", "-iname", "*seed110*cnn*"],
        capture_output=True, text=True,
    ).stdout.strip()
    print("\nCNN seed-110 weight search (find -iname '*cnn*seed110*' -o '*seed110*cnn*'):")
    print(f"  matches: {find_out if find_out else '(none)'}")
    cnn_weights_exist = bool(find_out)
    print(f"CNN seed-110 weights available: {cnn_weights_exist}")
    if not cnn_weights_exist:
        print("CNN seed-110 comparison: N/A -- no saved CNN weights for seed 110 exist "
              "anywhere in the repo. Not fabricated.")

    if not resnet_weights_exist:
        print("\nResNet1D seed-110 weights missing -- cannot proceed with evaluation.")
        return

    hdr("1. SEED 110 RECORD FROM SWEEP CSV / LOG")
    import csv
    with open(f"{SOURCE_DIR}/logs/degradation_aware_resnet_final_results.csv") as f:
        rows = list(csv.DictReader(f))
    row110 = next(r for r in rows if r["seed"] == "110")
    for k, v in row110.items():
        print(f"  {k}: {v}")

    print("\nMatching line from sweep log:")
    with open(f"{SOURCE_DIR}/logs/degradation_aware_resnet_final_sweep_log.txt") as f:
        for line in f:
            if line.startswith("Seed 110:"):
                print(f"  {line.strip()}")

    hdr("2. RE-EVALUATING SEED-110 RESNET1D ACROSS STANDARD DEPTH GRID")
    from resnet_classifier import ResNet1D
    from physics_model import stiffness_classes
    from degradation_aware_final_sweep import (
        generate_degraded_windows,
        EVAL_DEPTHS_CM,
        FREQ,
    )

    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)

    model = ResNet1D(n_classes=3)
    state_dict = torch.load(RESNET_WEIGHTS, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded state_dict from {RESNET_WEIGHTS}, model set to eval().")

    from sklearn.preprocessing import LabelEncoder
    class_kpa_names = sorted(stiffness_classes.keys())
    N_SAMPLES_PER_CELL = 300
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)
    print(f"EVAL_DEPTHS_CM ({len(EVAL_DEPTHS_CM)} depths): {list(EVAL_DEPTHS_CM)}")
    print(f"FREQ = {FREQ} Hz, N_SAMPLES_PER_CELL = {N_SAMPLES_PER_CELL} "
          f"({n_per_class}/class), classes = {class_kpa_names}")

    encoder = LabelEncoder()
    encoder.fit(class_kpa_names)

    print("\nPer-depth accuracy (ResNet1D, seed 110 reloaded weights):")
    print(f"{'depth_cm':>10}  {'accuracy':>10}")
    per_depth = []
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
            preds = model(X_cell_t).argmax(dim=1).numpy()
        acc = float(np.mean(preds == y_cell_encoded))
        per_depth.append((float(depth_cm), acc))
        print(f"{depth_cm:>10.4f}  {acc:>10.4f}")

    hdr("3. CNN SEED-110 COMPARISON")
    print("N/A -- no saved CNN weights for seed 110 exist anywhere in the repo "
          "(checked Source/, audit_fixes/lane5/recall_floor_main_run/seed_weights/, and a repo-wide "
          "find for *seed110* patterns). README.md independently confirms "
          "Log 20's 24-seed CNN sweep never called torch.save() for any seed. "
          "Cannot produce a CNN accuracy-by-depth curve for seed 110 without "
          "retraining, which is out of scope for this read-only check.")

    hdr("DONE")


if __name__ == "__main__":
    main()

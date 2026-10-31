#!/usr/bin/env python3
"""
Lane 7 follow-up v2, item 2: empirically verify whether seed i produces
IDENTICAL training and evaluation data in degradation_aware_final_sweep.py
(CNN) and degradation_aware_resnet_final_sweep.py (ResNet1D), by replaying
each script's exact pre-training call sequence (seeding -> build training
set -> N_SHUFFLES label permutations -> eval-window generation for one
depth) and diffing the resulting numpy arrays. No retraining is performed
(that would take hours); only the *data-generation* calls are replayed,
which is all that determines whether the two scripts see the same data for
a given seed.

Read-only. Writes only inside audit_fixes/lane7/.
Run from repo root: python3 audit_fixes/lane7/verify_seed_pairing.py
"""
import sys
import numpy as np

from pathlib import Path as _P; SOURCE_DIR = (str(_P(__file__).resolve().parents[2]) + "/Source")
sys.path.insert(0, SOURCE_DIR)

SEED = 101


def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def replay_cnn_sequence(seed):
    """Exact call order from degradation_aware_final_sweep.py's run_one_seed,
    stopping before model training (torch ops don't affect the numpy stream
    used for data gen, so this is sufficient to check data identity)."""
    import torch
    from degradation_aware_final_sweep import (
        build_degradation_aware_dataset,
        N_SHUFFLES,
        EVAL_DEPTHS_CM,
        generate_degraded_windows,
        FREQ,
    )
    from physics_model import stiffness_classes
    from sklearn.preprocessing import LabelEncoder

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train_names = build_degradation_aware_dataset()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)

    shuffles = [np.random.permutation(y_train_encoded).copy() for _ in range(N_SHUFFLES)]

    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = 300 // len(class_kpa_names)
    depth_cm = EVAL_DEPTHS_CM[0]
    depth_m = depth_cm / 100
    X_cell, y_cell_names = [], []
    for name in class_kpa_names:
        kpa = stiffness_classes[name]
        windows = generate_degraded_windows(kpa, depth_m, FREQ, n_per_class)
        X_cell.extend(windows)
        y_cell_names.extend([name] * len(windows))
    X_cell = np.array(X_cell)

    return {
        "X_train": X_train,
        "y_train_names": y_train_names,
        "shuffles": shuffles,
        "X_cell_depth0": X_cell,
        "y_cell_names_depth0": y_cell_names,
    }


def replay_resnet_sequence(seed):
    """Exact call order from degradation_aware_resnet_final_sweep.py's
    run_one_seed, stopping before model training, for the SAME reason."""
    import torch
    from degradation_aware_resnet_final_sweep import N_SHUFFLES, N_SAMPLES_PER_CELL
    from degradation_aware_final_sweep import (
        build_degradation_aware_dataset,
        EVAL_DEPTHS_CM,
        generate_degraded_windows,
        FREQ,
    )
    from physics_model import stiffness_classes
    from sklearn.preprocessing import LabelEncoder

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, y_train_names = build_degradation_aware_dataset()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)

    shuffles = [np.random.permutation(y_train_encoded).copy() for _ in range(N_SHUFFLES)]

    class_kpa_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_kpa_names)
    depth_cm = EVAL_DEPTHS_CM[0]
    depth_m = depth_cm / 100
    X_cell, y_cell_names = [], []
    for name in class_kpa_names:
        kpa = stiffness_classes[name]
        windows = generate_degraded_windows(kpa, depth_m, FREQ, n_per_class)
        X_cell.extend(windows)
        y_cell_names.extend([name] * len(windows))
    X_cell = np.array(X_cell)

    return {
        "X_train": X_train,
        "y_train_names": y_train_names,
        "shuffles": shuffles,
        "X_cell_depth0": X_cell,
        "y_cell_names_depth0": y_cell_names,
    }


def main():
    hdr("0. QUOTED SEEDING CODE (verify by direct grep, not memory)")
    import subprocess
    for fname, label in [
        ("degradation_aware_final_sweep.py", "CNN sweep"),
        ("degradation_aware_resnet_final_sweep.py", "ResNet1D sweep"),
    ]:
        out = subprocess.run(
            ["grep", "-n", "-A2", "def run_one_seed", f"{SOURCE_DIR}/{fname}"],
            capture_output=True, text=True,
        ).stdout
        print(f"\n--- {label} ({fname}) ---")
        print(out)

    hdr(f"1. REPLAYING DATA-GENERATION CALL SEQUENCE FOR SEED {SEED}")
    print("(Model training itself is NOT replayed -- both scripts' training "
          "loops use only torch's RNG, which is independent of numpy's RNG "
          "used for data generation, so replaying data-gen calls alone is "
          "sufficient to check whether the two scripts see identical data.)")

    cnn_data = replay_cnn_sequence(SEED)
    resnet_data = replay_resnet_sequence(SEED)

    hdr("2. COMPARISON RESULTS")
    x_train_match = np.array_equal(cnn_data["X_train"], resnet_data["X_train"])
    y_train_match = list(cnn_data["y_train_names"]) == list(resnet_data["y_train_names"])
    print(f"X_train identical (np.array_equal): {x_train_match}")
    print(f"y_train_names identical: {y_train_match}")
    print(f"X_train shape: CNN={cnn_data['X_train'].shape}  ResNet={resnet_data['X_train'].shape}")

    shuffles_match = all(
        np.array_equal(a, b) for a, b in zip(cnn_data["shuffles"], resnet_data["shuffles"])
    )
    print(f"All {len(cnn_data['shuffles'])} label-shuffle permutations identical: {shuffles_match}")

    x_cell_match = np.array_equal(cnn_data["X_cell_depth0"], resnet_data["X_cell_depth0"])
    y_cell_match = cnn_data["y_cell_names_depth0"] == resnet_data["y_cell_names_depth0"]
    print(f"Eval-cell windows at depth_cm={EVAL_DEPTHS_CM_FIRST()} identical: {x_cell_match}")
    print(f"Eval-cell labels at that depth identical: {y_cell_match}")

    all_identical = x_train_match and y_train_match and shuffles_match and x_cell_match and y_cell_match
    hdr("3. VERDICT")
    if all_identical:
        print(f"MATCH -- for seed {SEED}, both scripts' data-generation call "
              "sequences produce bit-identical training data, label-shuffle "
              "permutations, and evaluation windows (checked at one depth as "
              "a representative sample; the generation function and call "
              "order are identical for every other depth too, since both "
              "scripts import and call the same generate_degraded_windows "
              "over the same EVAL_DEPTHS_CM list). The paired-seed comparison "
              "in Lane 7's main analysis (Wilcoxon / paired HL) is comparing "
              "the two models on genuinely identical inputs per seed, not "
              "merely index-matched seeds that happen to share a number.")
    else:
        print("MISMATCH -- the two scripts do NOT produce identical data for "
              "matching seed values. The paired test should be described as "
              "index-matching only (same seed number, not verified-identical "
              "underlying data).")


def EVAL_DEPTHS_CM_FIRST():
    from degradation_aware_final_sweep import EVAL_DEPTHS_CM
    return float(EVAL_DEPTHS_CM[0])


if __name__ == "__main__":
    main()

# Lane 5 -- Step 2: load the EXISTING saved seed-121 certified weights
# (Source/models/soma_certified_model_seed121_corrected_noise.pt) and
# reproduce per-class recall at depth=8.48cm, using the SAME method as
# Source/degradation_aware_per_class_recall_floor.py (imported unchanged
# where possible). Read-only w.r.t. Source/.
#
# CHECK VALUES (given): cirrhosis 0.148, fibrosis 0.158, healthy 0.969

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Source"))

from degradation_aware_per_class_recall_floor import (   # noqa: E402
    generate_windows,
    class_names_sorted,
    encoder,
    N_PER_CLASS,
    N_REPLICATES,
    MODEL_PATH,
    FREQ,
)
from physics_model import stiffness_classes   # noqa: E402
from cnn_classifier import SimpleCNN          # noqa: E402

DEPTH_CM = 8.48
CHECK_VALUES = {"cirrhosis": 0.148, "significant_fibrosis": 0.158, "healthy": 0.969}
TOLERANCE = 0.005  # check values given to 3 decimals; allow rounding slack

MODEL_ABS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Source", MODEL_PATH)


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
    return per_class_recall


def main():
    print(f"Loading certified seed-121 weights from {MODEL_ABS_PATH}")
    model = SimpleCNN(n_classes=3)
    model.load_state_dict(torch.load(MODEL_ABS_PATH, map_location="cpu"))
    model.eval()

    recalls_by_class = {name: [] for name in class_names_sorted}
    for rep in range(N_REPLICATES):
        rng = np.random.default_rng(1000 * int(DEPTH_CM * 100) + rep)
        per_class_recall = eval_one_replicate(model, DEPTH_CM, rng)
        for name in class_names_sorted:
            recalls_by_class[name].append(per_class_recall[name])

    print(f"\nPer-class recall at depth={DEPTH_CM}cm ({N_REPLICATES} replicates x {N_PER_CLASS}/class):")
    all_match = True
    for name in class_names_sorted:
        mean_recall = float(np.mean(recalls_by_class[name]))
        check = CHECK_VALUES.get(name)
        match = (check is not None) and (abs(mean_recall - check) < TOLERANCE)
        all_match = all_match and match
        print(f"  {name:>22}: recall={mean_recall:.4f}  check={check}  {'MATCH' if match else 'MISMATCH'}")

    print(f"\nOVERALL: {'MATCH' if all_match else 'MISMATCH'}")


if __name__ == "__main__":
    main()

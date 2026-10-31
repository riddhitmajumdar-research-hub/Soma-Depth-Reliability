# Calibration gap of the 24 degradation-aware networks (seeds 101 to 124),
# evaluation only: no network is trained here.
#
# For each network: 60 windows per depth (20 per class) at the 23 depths of the
# certified sweep, 27.05 dB sensor noise, one fixed evaluation draw shared by
# all networks (EVAL_SEED below). Logits are taken from the saved weights and
# the gap is the binned calibration gap of calibration_analysis.py: 10
# fixed-width confidence bins, mean over non-empty bins of |mean confidence -
# accuracy| (compute_uce).
#
# Scoped temperature scaling, re-implemented from the description in README
# Log 23 (the script that fitted the original is not in the repository): one
# global temperature T per network, fitted by minimising the negative
# log-likelihood on a depth-stratified half of the windows between 0 and
# 5.45 cm (30 of the 60 windows at each of those depths), and the gap is
# computed on the other half before and after dividing the logits by T.
#
# The weights are not in the release (audit_fixes/README.md, "What is not in
# this tree"). Pass the folder that holds degradation_aware_seed101.pt ...
# degradation_aware_seed124.pt with --weights-dir.
#
#   python audit_fixes/proxy_sweeps/calibration_24_models.py --weights-dir <folder with the .pt files>
# Output: audit_fixes/proxy_sweeps/calibration_24_models.csv

import os
import sys
import csv
import argparse
import numpy as np
import torch
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Source"))

from degradation_aware_final_sweep import generate_degraded_windows, EVAL_DEPTHS_CM, FREQ  # noqa: E402
from cnn_classifier import SimpleCNN  # noqa: E402
from physics_model import stiffness_classes  # noqa: E402

EVAL_SEED = 20261004          # fixed evaluation draw, recorded here and in the CSV
FIT_SEED = EVAL_SEED + 1      # seed of the fit/held split
N_PER_CELL = 60
N_BINS = 10
SCOPE_CM = 5.45
CLASS_NAMES = sorted(stiffness_classes.keys())   # alphabetical, the encoder's order
SEEDS = list(range(101, 125))


def fixed_width_gap(conf, correct, n_bins=N_BINS):
    """compute_uce(fixed_width_bins(...)) of calibration_analysis.py."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    gaps = [abs(conf[idx == b].mean() - correct[idx == b].mean()) for b in range(n_bins) if (idx == b).any()]
    return float(np.mean(gaps)) if gaps else float("nan")


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits, y, T):
    p = softmax(logits / T)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def build_eval_set():
    np.random.seed(EVAL_SEED)
    X, y, d = [], [], []
    for depth_cm in EVAL_DEPTHS_CM:
        for j, name in enumerate(CLASS_NAMES):
            w = generate_degraded_windows(stiffness_classes[name], depth_cm / 100, FREQ, N_PER_CELL // len(CLASS_NAMES))
            X.extend(w)
            y.extend([j] * len(w))
            d.extend([float(depth_cm)] * len(w))
    return np.array(X), np.array(y), np.array(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default=os.path.join(REPO, "audit_fixes", "lane5", "recall_floor_main_run", "seed_weights"))
    ap.add_argument("--out", default=os.path.join(HERE, "calibration_24_models.csv"))
    a = ap.parse_args()
    missing = [s for s in SEEDS if not os.path.exists(os.path.join(a.weights_dir, f"degradation_aware_seed{s}.pt"))]
    if missing:
        print(f"The saved weights are not in this folder (they are not part of the release); missing seeds: {missing[:5]}"
              f"{'...' if len(missing) > 5 else ''}. Pass --weights-dir.")
        sys.exit(1)
    torch.set_num_threads(4)
    X, y, depths = build_eval_set()
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    rs = np.random.RandomState(FIT_SEED)
    in_scope = depths <= SCOPE_CM + 1e-9
    fit_mask = np.zeros(len(y), dtype=bool)
    for d in np.unique(depths[in_scope]):
        idx = np.where(depths == d)[0]
        fit_mask[rs.permutation(idx)[: len(idx) // 2]] = True
    held_mask = in_scope & ~fit_mask
    fit_mask &= in_scope
    rows = []
    for seed in SEEDS:
        model = SimpleCNN(n_classes=3)
        model.load_state_dict(torch.load(os.path.join(a.weights_dir, f"degradation_aware_seed{seed}.pt"), map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            logits = model(Xt).numpy()
        probs = softmax(logits)
        conf, correct = probs.max(axis=1), (probs.argmax(axis=1) == y).astype(float)
        gap_all = fixed_width_gap(conf, correct)
        res = minimize_scalar(lambda lt: nll(logits[fit_mask], y[fit_mask], float(np.exp(lt))), bounds=(-8, 4), method="bounded")
        T = float(np.exp(res.x))
        p2 = softmax(logits / T)
        c2, k2 = p2.max(axis=1), (p2.argmax(axis=1) == y).astype(float)
        before = fixed_width_gap(conf[held_mask], correct[held_mask])
        after = fixed_width_gap(c2[held_mask], k2[held_mask])
        rows.append({"seed": seed, "gap_all_depths": gap_all, "T": T, "scoped_gap_before": before, "scoped_gap_after": after,
                     "improved": after < before, "n_fit": int(fit_mask.sum()), "n_held": int(held_mask.sum()),
                     "eval_seed": EVAL_SEED})
        print(f"seed {seed}: gap_all={gap_all:.3f} T={T:.4f} scoped before={before:.3f} after={after:.3f} improved={after < before}")
    ga = np.array([r["gap_all_depths"] for r in rows])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        w.writerow({"seed": "mean", "gap_all_depths": ga.mean(), "scoped_gap_before": np.mean([r["scoped_gap_before"] for r in rows]),
                    "scoped_gap_after": np.mean([r["scoped_gap_after"] for r in rows]), "improved": sum(r["improved"] for r in rows),
                    "eval_seed": EVAL_SEED})
        w.writerow({"seed": "sd", "gap_all_depths": ga.std(ddof=1), "scoped_gap_before": np.std([r["scoped_gap_before"] for r in rows], ddof=1),
                    "scoped_gap_after": np.std([r["scoped_gap_after"] for r in rows], ddof=1), "eval_seed": EVAL_SEED})
    print(f"gap over all depths: mean {ga.mean():.3f}, SD {ga.std(ddof=1):.3f}, min {ga.min():.3f}, max {ga.max():.3f}; "
          f"scoped scaling reduced the held-out gap in {sum(r['improved'] for r in rows)} of {len(rows)} networks")


if __name__ == "__main__":
    main()

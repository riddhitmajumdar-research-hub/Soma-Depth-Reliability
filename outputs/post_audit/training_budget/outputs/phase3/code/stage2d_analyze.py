"""Stage 2D analysis (rule: the stage 2D addendum, section 4). Reads saved files only.
J = 2.0 ms floors: raw_stage2d/j2p0_e200/seed<s>_draw0.npz (seeds 101-108); J = 1.0 ms reference: raw_stage2/e200/seed<s>_draw0.npz. Writes outputs/stage2d/STAGE2D_RESULTS.md.
Paired difference = floor(J=2.0) - floor(J=1.0); exact two-sided binomial sign test, zero differences dropped, n stated. Env STAGE2D_J2_DIR overrides the J=2.0 folder (tests only)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_floors_common import *
seeds = list(range(101, 109)); d2 = os.environ.get("STAGE2D_J2_DIR") or os.path.join(P3, "raw_stage2d", "j2p0_e200")
f2 = np.zeros(len(seeds)); f1 = np.zeros(len(seeds)); c2 = np.zeros(len(seeds), bool)
for i, s in enumerate(seeds):
    p2f = os.path.join(d2, f"seed{s}_draw0.npz")
    if not os.path.exists(p2f): sys.exit(f"MISSING {p2f}: Stage 2D is not complete; nothing computed.")
    f2[i], c2[i] = floor_from_file(p2f); f1[i], _ = floor_from_file(os.path.join(P3, "raw_stage2", "e200", f"seed{s}_draw0.npz"))
d = f2 - f1; up, dn, z, n, p = sign_test(d)
L = ["# Stage 2D results: timing jitter J = 2.0 ms vs J = 1.0 ms, 200 epochs, Phase 3 CNN, seeds 101-108, K = 1, B = 719",
     "J is the half-width of a uniform [-J, +J] distribution (SD = J/sqrt(3): 0.577 ms at J = 1.0, 1.155 ms at J = 2.0). Floors in cm, new rule.", "",
     "| seed | floor J = 1.0 (Stage 2 draw 0) | floor J = 2.0 | difference (2.0 - 1.0) |", "|---|---|---|---|"]
L += [f"| {s} | {f1[i]:.4f} | {f2[i]:.4f}{'*' if c2[i] else ''} | {d[i]:+.4f} |" for i, s in enumerate(seeds)]
L += ["", "(* censored at 20 cm)", "", "## J = 2.0 ms across seeds",
      f"n = {len(f2)}; mean {f2.mean():.3f}; SD (ddof=1) {f2.std(ddof=1):.3f}; median {np.median(f2):.3f}; min {f2.min():.3f}; max {f2.max():.3f}. J = 1.0 reference: mean {f1.mean():.3f}; SD {f1.std(ddof=1):.3f}; median {np.median(f1):.3f}.",
      f"Paired difference: {up} up / {z} equal / {dn} down; exact two-sided binomial sign test, zero differences dropped, n = {n}, p = {p:.4g}." + (" (n = 0: test not defined.)" if n == 0 else ""),
      f"Seeds with floor 0.00 at J = 2.0: {[s for s, x in zip(seeds, f2) if x == 0.0] or 'none'}."]
out = os.path.join(P3, "..", "stage2d", "STAGE2D_RESULTS.md" if not os.environ.get("STAGE2D_J2_DIR") else "SELFTEST_J2_equals_J1.md"); open(out, "w").write("\n".join(L) + "\n"); print("\n".join(L[-4:]))

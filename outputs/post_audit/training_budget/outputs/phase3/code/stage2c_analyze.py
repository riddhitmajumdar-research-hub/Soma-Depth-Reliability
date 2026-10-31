"""Stage 2C analysis (rule: the stage 2C addendum, section 4). No training, no permutations; reads saved draw files only.
Full:  stage2c_analyze.py            draws 0 (raw_stage2/e200) + 1, 2 (raw_stage2c/e200) for seeds 101-124; writes outputs/stage2c/STAGE2C_RESULTS.md
Test:  stage2c_analyze.py --k1   (env STAGE2C_DIR=<folder> reads draws 1 and 2 from there: tests only)
Test:  stage2c_analyze.py --k1       draw 0 only (Stage 2B/Stage 2 K = 1 data); writes outputs/stage2c/K1_reproduction_check.md
Per-seed floor = median of the draws; across seeds mean, SD (ddof=1), median, min, max, P10 (numpy linear), percentile bootstrap CI for P10 over seeds
(10,000 resamples, RNG seed 20261009, as in stage2b_results.py); variance split by one-way random-effects ANOVA (seed vs draw); P10 == min flagged."""
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage2_floors_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--k1", action="store_true"); a = ap.parse_args()
seeds = list(range(101, 125)); K = 1 if a.k1 else 3
fl = np.zeros((len(seeds), K)); cens = np.zeros((len(seeds), K), bool)
for i, s in enumerate(seeds):
    for k in range(K):
        p = os.path.join(P3, "raw_stage2", "e200", f"seed{s}_draw0.npz") if k == 0 else os.path.join(os.environ.get("STAGE2C_DIR") or os.path.join(P3, "raw_stage2c", "e200"), f"seed{s}_draw{k}.npz")
        if not os.path.exists(p): sys.exit(f"MISSING {p}: Stage 2C is not complete; nothing computed.")
        fl[i, k], cens[i, k] = floor_from_file(p)
cert = {int(r["seed"]): float(r["floor_new_median"]) for r in csv.DictReader(open(os.path.join(P3, "analysis_cnn", "cnn_floor_by_seed.csv")))}
v = np.median(fl, axis=1); c = np.array([cert[s] for s in seeds]); g = v - c
p10 = float(np.percentile(v, 10)); rb = np.random.RandomState(20261009); bs = np.percentile(v[rb.randint(0, len(v), (10000, len(v)))], 10, axis=1)
lo, hi = np.percentile(bs, [2.5, 97.5]); up, dn, z, n, ps = sign_test(g)
L = [f"# Stage 2{'C' if not a.k1 else 'B (K = 1 reproduction check of the Stage 2C script)'}: 200 epochs, Phase 3 configuration, N = 24, K = {K}",
     "Per-seed floor = median of the draws (K = 1: the single draw). Floors in cm, new rule, B = 719, BH q = 0.05.", "",
     "| seed | " + " | ".join(f"draw {k}" for k in range(K)) + " | per-seed floor | certified 60-epoch median | difference |", "|---|" + "---|" * (K + 3)]
L += [f"| {s} | " + " | ".join(f"{fl[i,k]:.4f}{'*' if cens[i,k] else ''}" for k in range(K)) + f" | {v[i]:.4f} | {c[i]:.4f} | {g[i]:+.4f} |" for i, s in enumerate(seeds)]
L += ["", "(* = censored at 20 cm)", "", "## Across seeds",
      f"n = {len(v)}; mean {v.mean():.3f}; SD (ddof=1) {v.std(ddof=1):.3f}; median {np.median(v):.3f}; min {v.min():.3f}; max {v.max():.3f}; P10 (linear) {p10:.3f}; 95% bootstrap CI for P10 over seeds [{lo:.3f}, {hi:.3f}].",
      f"P10 equals the minimum floor: {'YES, flagged as a grid coincidence' if abs(p10 - v.min()) < 1e-9 else 'no'}.",
      f"Difference to the certified 60-epoch per-seed medians: {up} up / {z} equal / {dn} down; exact two-sided binomial sign test, zero differences dropped, n = {n}, p = {ps:.4g}."]
if K > 1:
    msw = ((fl - fl.mean(1, keepdims=True)) ** 2).sum() / (len(seeds) * (K - 1)); msb = K * ((fl.mean(1) - fl.mean()) ** 2).sum() / (len(seeds) - 1)
    vs = max((msb - msw) / K, 0.0)
    L += ["", "## Variance split (one-way random-effects ANOVA on single-draw floors)",
          f"Between-draw (within-seed) variance {msw:.3f} (SD {msw**.5:.3f} cm); between-seed variance {vs:.3f} (SD {vs**.5:.3f} cm); share of total variance between seeds {vs/(vs+msw):.1%}. Reference: Phase 3 K = 10 draw-to-draw SD 1.06 cm at 60 epochs (`analysis_cnn/cnn_variance_components_all.csv`)."]
out = os.path.join(P3, "..", "stage2c", "K1_reproduction_check.md" if a.k1 else ("STAGE2C_SELFTEST_triplicated_draw0.md" if os.environ.get("STAGE2C_DIR") else "STAGE2C_RESULTS.md")); open(out, "w").write("\n".join(L) + "\n")
print("\n".join(L[-(4 if K == 1 else 7):]))

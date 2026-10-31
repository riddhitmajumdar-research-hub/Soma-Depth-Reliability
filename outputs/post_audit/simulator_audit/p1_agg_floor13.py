"""Aggregates Step 1.3 data into tables used by diagnosis_1_3.md."""
import csv, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import WORKTREE, gen_windows, stiffness_classes, STAGE_MU_RANGE_KPA, CLASS_NAMES, snr_to_std, SENSOR_SNR_DB
D = os.path.join(WORKTREE, "outputs/phase1/data/floor13")
fl = [r for f in sorted(glob.glob(f"{D}/floors_seed*_*.csv")) for r in csv.DictReader(open(f))]
dt = [r for f in sorted(glob.glob(f"{D}/detail_seed*_*.csv")) for r in csv.DictReader(open(f))]
conds = []
for r in fl:
    if r["condition"] not in conds: conds.append(r["condition"])
seeds = sorted({int(r["seed"]) for r in fl})
out = []
P = out.append
P("A. Floors by condition (8 seeds). C = censored at the 15 cm grid maximum, Z = floor 0.00")
P(f"{'condition':24s} " + " ".join(f"{s:>6d}" for s in seeds) + "  nC nZ  median")
for c in conds:
    v = {int(r["seed"]): (float(r["floor_cm"]), r["censored"] == "True") for r in fl if r["condition"] == c}
    P(f"{c:24s} " + " ".join((f"{v[s][0]:5.1f}C" if v[s][1] else f"{v[s][0]:5.1f} ") + "" for s in seeds) +
      f"  {sum(v[s][1] for s in seeds):2d} {sum(v[s][0]==0 for s in seeds):2d}  {np.median([v[s][0] for s in seeds]):.2f}")
P("")
P("B. Real accuracy at 15 cm and at 0 cm (mean over seeds) per condition")
for c in conds:
    rr = [r for r in fl if r["condition"] == c]
    P(f"{c:24s} acc0={np.mean([float(r['acc_depth0']) for r in rr]):.3f} acc15={np.mean([float(r['acc_depth15']) for r in rr]):.3f} n_sig_mean={np.mean([int(r['n_sig']) for r in rr]):.1f}")
P("")
P("C. Floor-0 cases and the BH tie condition (depth rows per seed/condition)")
by = {}
for r in dt: by.setdefault((int(r["seed"]), r["condition"]), []).append(r)
nz = 0
for (s, c), rows in sorted(by.items()):
    p = np.array([float(x["raw_p"]) for x in rows]); sig = [x["sig"] == "True" for x in rows]
    if not sig[0]:
        nz += 1
        P(f"  seed {s} {c}: acc0={float(rows[0]['real_acc']):.3f} p0={p[0]:.4f} adj0={float(rows[0]['bh_adj_p']):.4f} n_at_min_p={(p == p.min()).sum()} n_sig={sum(sig)} p_min={p.min():.5f}")
P(f"  seeds x conditions with depth 0 non-significant: {nz} of {len(by)}")
P("")
P("D. Null (shuffled-label control) accuracy distribution at depth 0, condition 'base' (200 controls per seed)")
for s in seeds:
    z = np.load(f"{D}/controls_seed{s}_base.npz")
    a = z["base"][0]
    P(f"  seed {s}: min {a.min():.3f} median {np.median(a):.3f} max {a.max():.3f} | frac<=0.40: {np.mean(a<=0.40):.3f} | frac 0.40-0.60: {np.mean((a>0.4)&(a<0.6)):.3f} | frac>=0.60: {np.mean(a>=0.6):.3f} | frac >=0.80: {np.mean(a>=0.8):.3f} | exactly 1/3: {np.mean(np.abs(a-1/3)<1e-9):.3f}")
P("")
P("E. Real model predicted-class fractions and recall, trained on continuous stiffness (range_stage_trrange), depth 0 and 6.5 cm, mean over seeds")
for c in ("range_stage_trrange", "range_lookup_trrange", "range_stage_trbase"):
    for d in (0.0, 6.5, 15.0):
        rr = [r for r in dt if r["condition"] == c and abs(float(r["depth_cm"]) - d) < 1e-6]
        P(f"  {c:22s} d={d:4.1f}: acc={np.mean([float(r['real_acc']) for r in rr]):.3f} recall cirr/heal/fibr={np.mean([float(r['rec_cirr']) for r in rr]):.2f}/{np.mean([float(r['rec_heal']) for r in rr]):.2f}/{np.mean([float(r['rec_fibr']) for r in rr]):.2f} pred_frac(cirr,heal,fibr)={np.mean([float(r['pred_frac_0']) for r in rr]):.2f}/{np.mean([float(r['pred_frac_1']) for r in rr]):.2f}/{np.mean([float(r['pred_frac_2']) for r in rr]):.2f}")
P("")
P("F. Simulator collapse checks for continuous mu in the stage ranges (no model), 400 records per stage per depth, eta_by_stage on")
rng = np.random.RandomState(20261002)
std = snr_to_std(SENSOR_SNR_DB)
for name in CLASS_NAMES:
    lo, hi = STAGE_MU_RANGE_KPA[name]
    for d in (0.0, 6.5, 15.0):
        w, diag = gen_windows(stiffness_classes[name], d / 100, 100, 400, rng, class_name=name, mu_mode="range", eta_mode="stage", return_diag=True)
        X = np.array(w)
        tof = np.array([x["tof_ms"] for x in diag]); k = np.array([x["kpa"] for x in diag]); pk = np.array([x["peak_atten"] for x in diag])
        P(f"  {name:21s} d={d:4.1f} windows={X.shape} finite={np.isfinite(X).all()} all_zero_windows={(np.abs(X).max(1)==0).sum()} mean_window_std={X.std(1).mean():.4f} mu[{k.min():.2f},{k.max():.2f}] tof_ms[{tof.min():.2f},{tof.max():.2f}] median peak/noise={np.median(pk)/std:.2f}")
P("  TOF ranges by stage at the interval edges (10 mm / c): " + ", ".join(f"{n}: {10/ (np.sqrt(lo*1000/3/1060)):.2f} to {10/(np.sqrt(hi*1000/3/1060)):.2f} ms" for n, (lo, hi) in STAGE_MU_RANGE_KPA.items()))
P("  Stage intervals abut at 6.0 and 9.0 kPa, so TOF at the border is identical for the two neighbouring labels.")
open(os.path.join(D, "floor13_summary.txt"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

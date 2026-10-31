"""Addendum section A. Reads data4/calib_by_seed_depth.csv and Phase 3 floors; writes data4/gap_dynamics_*.csv."""
import csv, os, numpy as np
from scipy.stats import spearmanr
P4 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); D4 = os.path.join(P4, "data4"); P3 = os.path.join(P4, "..", "phase3")
R = list(csv.DictReader(open(os.path.join(D4, "calib_by_seed_depth.csv")))); seeds = list(range(101, 125))
fl = {int(r["seed"]): float(r["floor_new_median"]) for r in csv.DictReader(open(os.path.join(P3, "analysis_cnn", "cnn_floor_by_seed.csv")))}
cw = {int(r["seed"]): r for r in csv.DictReader(open(os.path.join(P3, "analysis_cnn", "cnn_classwise_by_seed.csv")))}
cwk = [k for k in next(iter(cw.values())).keys() if "floor" in k.lower()]; print("classwise columns:", cwk)
rows = []
for s in seeds:
    d = sorted((float(r["depth_cm"]), float(r["median_logit_gap"])) for r in R if int(r["seed"]) == s); D = np.array([a for a, _ in d]); g = np.array([b for _, b in d])
    g0 = g[0]; plateau = g[D >= 17.5].mean(); n_pl = int((D >= 17.5).sum())
    i = int(np.argmax(g <= g0 / 2)) if (g <= g0 / 2).any() else None
    if i is None: half = np.nan
    elif i == 0: half = 0.0
    else: half = D[i-1] + (g0 / 2 - g[i-1]) * (D[i] - D[i-1]) / (g[i] - g[i-1])
    m = D <= 5.45; sl, ic = np.polyfit(D[m], np.log(g[m]), 1); pred = sl * D[m] + ic; r2 = 1 - ((np.log(g[m]) - pred) ** 2).sum() / ((np.log(g[m]) - np.log(g[m]).mean()) ** 2).sum()
    hv = np.log(2) / -sl if sl < 0 else np.nan
    ci = int(np.argmax(g <= 1.25 * plateau)) if (g <= 1.25 * plateau).any() else None
    col = D[ci] if ci is not None else np.nan
    rows.append(dict(seed=s, g0=g0, plateau=plateau, n_plateau_depths=n_pl, half_depth=half, slope_ln=sl, halving_distance=hv, r2=r2, collapse_depth=col, floor_new=fl[s], classwise_floor=float(cw[s][cwk[0]]) if cwk else np.nan))
with open(os.path.join(D4, "gap_dynamics_by_seed.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
rb = np.random.RandomState(20261008); out = []
def summ(name, v):
    v = np.array(v, float); ok = v[~np.isnan(v)]; bs = [ok[rb.randint(0, len(ok), len(ok))].mean() for _ in range(10000)]
    out.append([name, len(ok), round(ok.mean(), 3), round(ok.std(ddof=1), 3), round(np.median(ok), 3), round(ok.min(), 3), round(ok.max(), 3), round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)])
for k in ("g0", "plateau", "half_depth", "halving_distance", "r2", "collapse_depth"): summ(k, [r[k] for r in rows])
def rel(name, xs, ys, excl=False):
    x = np.array(xs, float); y = np.array(ys, float); ok = ~np.isnan(x) & ~np.isnan(y)
    if excl: ok &= np.array(seeds) != 104
    x, y = x[ok], y[ok]; r0 = spearmanr(x, y)[0]; bs = []
    for _ in range(10000):
        ix = rb.randint(0, len(x), len(x))
        if np.ptp(x[ix]) and np.ptp(y[ix]): bs.append(spearmanr(x[ix], y[ix])[0])
    out.append([name + (" (without seed 104)" if excl else ""), len(x), round(r0, 3), "", "", "", "", round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)])
for ex in (False, True):
    for a in ("half_depth", "collapse_depth", "halving_distance"):
        rel(f"Spearman {a} vs floor_new", [r[a] for r in rows], [r["floor_new"] for r in rows], ex); rel(f"Spearman {a} vs classwise_floor", [r[a] for r in rows], [r["classwise_floor"] for r in rows], ex)
with open(os.path.join(D4, "gap_dynamics_summary.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["quantity", "n", "mean_or_rho", "sd", "median", "min", "max", "ci_lo", "ci_hi"]); w.writerows(out)
for o in out: print(*o, sep=" | ")
print("seeds with slope>=0 (halving undefined):", [r["seed"] for r in rows if np.isnan(r["halving_distance"])], "; seeds with no half depth:", [r["seed"] for r in rows if np.isnan(r["half_depth"])], "; seeds with no collapse depth:", [r["seed"] for r in rows if np.isnan(r["collapse_depth"])])

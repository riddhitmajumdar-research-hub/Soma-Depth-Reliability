"""Phase 4.2 agreement per PROTOCOL_PHASE4.md. Reads data4/attr_profiles.csv; writes data4/attr_agreement.csv, attr_seed_vs_median.csv, attr_controls.csv."""
import csv, os, numpy as np
from scipy.stats import spearmanr
P4 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); D4 = os.path.join(P4, "data4")
P = {}
for r in csv.reader(open(os.path.join(D4, "attr_profiles.csv"))):
    if r[0] != "net": P[(r[0], r[1], float(r[2]), r[3], int(r[4]))] = np.array(r[5:], float)
seeds = list(range(101, 125)); rb = np.random.RandomState(20261008)
def rho(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0: return np.nan
    return spearmanr(a, b)[0]
def pair_stats(net, meth, d, g):
    M = np.array([P[(net, meth, d, g, s)] for s in seeds]); n = len(seeds); R = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n): R[i, j] = R[j, i] = rho(M[i], M[j])
    iu = np.triu_indices(n, 1); vals = R[iu]; med = float(np.nanmedian(vals)); p5 = float(np.nanpercentile(vals, 5))
    b = []
    for _ in range(10000):
        idx = rb.randint(0, n, n); sub = R[np.ix_(idx, idx)]; b.append(np.nanmedian(sub[idx[:, None] != idx[None, :]]))
    return M, R, med, p5, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(np.isnan(vals).sum())
A = [["net", "method", "depth_cm", "group", "median_pairwise_rho", "ci_lo", "ci_hi", "p5_pairwise_rho", "pairs_undefined", "median_seed_vs_LOO_median_rho", "vs_median_ci_lo", "vs_median_ci_hi", "seeds_rho_gt_0.5_of_24", "replicated_(>=20)", "anomalous_seeds_(3_robust_SD_below_median)", "peak_tertile_mode_share"]]
V = [["method", "depth_cm", "group", "seed", "rho_vs_LOO_median", "peak_sample", "anomaly"]]
for meth in ("ig", "gradcam"):
    for d in (0.0, 4.0, 6.5, 8.0, 10.0):
        for g in ("all", "class0", "class1", "class2"):
            M, R, med, p5, lo, hi, nund = pair_stats("trained", meth, d, g)
            rs = np.array([rho(M[i], np.median(np.delete(M, i, 0), 0)) for i in range(len(seeds))])
            m_ = np.nanmedian(rs); mad = 1.4826 * np.nanmedian(np.abs(rs - m_)); an = [seeds[i] for i in range(len(seeds)) if not np.isnan(rs[i]) and rs[i] < m_ - 3 * mad and mad > 0]
            ok = rs[~np.isnan(rs)]; b = [np.median(ok[rb.randint(0, len(ok), len(ok))]) for _ in range(10000)]
            pk = M.argmax(1); tert = np.minimum(pk // 34, 2); share = float(np.bincount(tert).max() / len(tert))
            for i, s in enumerate(seeds): V.append([meth, d, g, s, None if np.isnan(rs[i]) else round(float(rs[i]), 3), int(pk[i]), int(s in an)])
            A.append(["trained", meth, d, g, round(med, 3), round(lo, 3), round(hi, 3), round(p5, 3), nund, round(float(m_), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3),
                      int((ok > 0.5).sum()), int((ok > 0.5).sum() >= 20), " ".join(map(str, an)) or "none", round(share, 2)])
C = [["method", "depth_cm", "group", "median_rho_trained_vs_abs_input", "median_rho_trained_vs_init_same_seed", "median_pairwise_rho_among_untrained_nets", "median_pairwise_rho_among_trained_nets"]]
for meth in ("ig", "gradcam"):
    for d in (0.0, 4.0, 6.5, 8.0, 10.0):
        for g in ("all", "class0", "class1", "class2"):
            vi = [rho(P[("trained", meth, d, g, s)], P[("input", "abs_input", d, g, s)]) for s in seeds]
            vt = [rho(P[("trained", meth, d, g, s)], P[("init", meth, d, g, s)]) for s in seeds]
            pu = pair_stats("init", meth, d, g)[2] if False else None
            Mu = np.array([P[("init", meth, d, g, s)] for s in seeds]); Ru = [rho(Mu[i], Mu[j]) for i in range(24) for j in range(i + 1, 24)]
            C.append([meth, d, g, round(float(np.nanmedian(vi)), 3), round(float(np.nanmedian(vt)), 3), round(float(np.nanmedian(Ru)), 3), [a for a in A if a[1] == meth and a[2] == d and a[3] == g][0][4]])
for nm, rows in (("attr_agreement.csv", A), ("attr_seed_vs_median.csv", V), ("attr_controls.csv", C)): csv.writer(open(os.path.join(D4, nm), "w", newline="")).writerows(rows)
for r in A: print(*r, sep=" | ")

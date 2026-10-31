"""Phase 3 analysis (3.3 variance decomposition, 3.4 floors, 3.5 classwise). Reads raw/<family> only.
Usage: [P3_RES=<out dir>] p3_analyze.py <family> ...     (single-threaded; run under nice)
BH decisions are recomputed from the archived counts with integer arithmetic (q = 1/20) and must equal the stored flags."""
import csv, glob, hashlib, json, os, sys
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_grid import GRID
WT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
P3 = os.path.join(WT, "outputs", "phase3")
RES = os.environ.get("P3_RES") or os.path.join(P3, "results")
GRID = np.array(GRID); M = len(GRID)
SPEC = {"cnn": dict(seeds=list(range(101, 125)), K=10, B=719), "rf": dict(seeds=list(range(101, 125)), K=10, B=719),
        "resnet": dict(seeds=list(range(101, 111)), K=10, B=719)}
LABEL = os.environ.get("P3_LABEL", "")   # e.g. "partial, N = 4, B = 719": added as the first column of every CSV and as a header line
if os.environ.get("P3_SEEDS"):           # complete seeds only, comma separated (partial snapshots)
    for _f in SPEC: SPEC[_f]["seeds"] = [int(x) for x in os.environ["P3_SEEDS"].split(",")]
SUBSETS = {"all": None, "101-110": set(range(101, 111))}
GRID_VAR = float(np.mean(np.diff(GRID) ** 2) / 12.0)
rng_boot = np.random.RandomState(20261003)


def bh_int(cnt, B, q_num=1, q_den=20):
    """cnt[i] = number of controls with accuracy >= real at depth i. Reject at rank k if (cnt+1)*m*q_den <= k*q_num*(B+1)."""
    cnt = np.asarray(cnt, np.int64); m = len(cnt); o = np.argsort(cnt, kind="stable")
    ok = [(int(cnt[o[k - 1]]) + 1) * m * q_den <= k * q_num * (B + 1) for k in range(1, m + 1)]
    kmax = max([k for k, v in enumerate(ok, 1) if v], default=0)
    sig = np.zeros(m, bool); sig[o[:kmax]] = True
    return sig


def new_floor(sig):
    """Last depth before the first run of 2 consecutive non-significant depths. Returns (floor, censored_at_20, start_nonsignificant)."""
    for i in range(M - 1):
        if (not sig[i]) and (not sig[i + 1]):
            if i == 0:
                return 0.0, False, True
            return float(GRID[i - 1]), False, False
    return float(GRID[-1]), True, False


def old_floor(sig):
    """First non-significant depth; 20 cm (censored) if none."""
    for i in range(M):
        if not sig[i]:
            return float(GRID[i]), False
    return float(GRID[-1]), True


def class_floor(rec):
    """First depth where the lowest class recall is < 0.45 at that depth and at the next. None: censored at 20 cm."""
    mn = rec.min(1)
    for i in range(M - 1):
        if mn[i] < 0.45 and mn[i + 1] < 0.45:
            return float(GRID[i]), False
    return float(GRID[-1]), True


def load_family(fam):
    spec = SPEC[fam]; d = os.path.join(P3, "raw", fam)
    ev, real, gaps = [], {}, []
    for s in spec["seeds"]:
        rp = os.path.join(d, f"seed{s}_real.npz")
        if not os.path.exists(rp):
            gaps.append((s, "real")); continue
        z = np.load(rp); real[s] = dict(counts=z["real_counts"], recall=z["recall"], pred_frac=z["pred_frac"])
        for k in range(spec["K"]):
            dp = os.path.join(d, f"seed{s}_draw{k}.npz")
            if not os.path.exists(dp):
                gaps.append((s, k)); continue
            w = np.load(dp)
            C = w["control_counts"].astype(np.int64); B = int(w["B"])
            assert B == spec["B"], f"B={B} in {fam} {s} {k}"
            cnt = (C >= w["real_counts"].astype(np.int64)[None, :]).sum(0)
            p = (cnt + 1) / (B + 1)
            assert np.allclose(p, w["p"]), f"stored p differs from recomputed p: {fam} {s} {k}"
            sig = bh_int(cnt, B)
            assert np.array_equal(sig, w["sig"].astype(bool)), f"BH differs: {fam} {s} {k}"
            nf, nc, ns = new_floor(sig); of, oc = old_floor(sig)
            ev.append(dict(family=fam, seed=s, draw=k, floor_new=nf, cens_new=nc, start_nonsig=ns, floor_old=of, cens_old=oc,
                           n_sig=int(sig.sum()), p_depth0=float(p[0]), p_min=float(p.min()), n_at_pmin=int((p == p.min()).sum()),
                           depth0_sig=bool(sig[0]), wall_s=float(w["wall_s"]), power=str(w["power_label"]), B=B,
                           sig_string="".join("1" if x else "0" for x in sig)))
    return ev, real, gaps


def summarize(vals):
    v = np.asarray(vals, float)
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else float("nan"), median=float(np.median(v)),
                min=float(v.min()), max=float(v.max()), p5=float(np.percentile(v, 5)), p10=float(np.percentile(v, 10)))


def boot_ci(v, pct, B=10000):
    v = np.asarray(v, float); n = len(v)
    idx = rng_boot.randint(0, n, size=(B, n))
    q = np.percentile(v[idx], pct, axis=1)
    return float(np.percentile(q, 2.5)), float(np.percentile(q, 97.5))


def anova(F):
    """F: list of arrays (draws per seed), possibly unbalanced. Returns (raw seed variance, draw variance)."""
    N = len(F); n_i = np.array([len(f) for f in F]); n = n_i.sum()
    gm = sum(f.sum() for f in F) / n
    ssw = sum(((f - f.mean()) ** 2).sum() for f in F); msw = ssw / (n - N)
    ssb = sum(len(f) * (f.mean() - gm) ** 2 for f in F); msb = ssb / (N - 1)
    n0 = (n - (n_i ** 2).sum() / n) / (N - 1)
    return (msb - msw) / n0, msw


def reml(F):
    y = np.concatenate(F); n = len(y); grp = np.concatenate([[i] * len(f) for i, f in enumerate(F)])
    Z = (grp[:, None] == np.arange(len(F))[None, :]).astype(float); X = np.ones((n, 1)); ZZ = Z @ Z.T

    def nll(th):
        s2a, s2e = th
        with np.errstate(all="ignore"):
            V = s2e * np.eye(n) + s2a * ZZ
            sign, ld = np.linalg.slogdet(V); Vi = np.linalg.inv(V)
            XtViX = X.T @ Vi @ X; mu = np.linalg.solve(XtViX, X.T @ Vi @ y); r = y - X @ mu
            val = 0.5 * (ld + np.linalg.slogdet(XtViX)[1] + r @ Vi @ r)
        return float(val) if np.isfinite(val) else 1e300   # line-search steps into extreme variances are rejected
    best = None
    s0 = np.var(y) if np.var(y) > 0 else 1.0
    for start in ([1.0, 1.0], [0.1, 0.5], [5.0, 0.2]):
        r = minimize(nll, np.array(start) * s0, method="L-BFGS-B", bounds=[(1e-9, None), (1e-9, None)])
        if best is None or r.fun < best.fun:
            best = r
    return float(best.x[0]), float(best.x[1])


def decomp(F, nboot=2000):
    va_raw, ve = anova(F); va = max(va_raw, 0.0)
    ra, re_ = reml(F)
    tot = va + ve + GRID_VAR
    out = dict(seed_raw=va_raw, seed=va, draw=ve, grid=GRID_VAR, total=tot, share_seed=va / tot, share_draw=ve / tot, share_grid=GRID_VAR / tot,
               reml_seed=ra, reml_draw=re_, negative_seed_estimate=bool(va_raw < 0), n_seeds=len(F), n_evals=int(sum(len(f) for f in F)))
    sh = []; N = len(F)
    for _ in range(nboot):
        idx = rng_boot.randint(0, N, N)
        Fb = [F[i] for i in idx]
        try:
            a, e = anova(Fb)
        except ZeroDivisionError:
            continue
        if not np.isfinite(a) or not np.isfinite(e):
            continue
        a = max(a, 0.0); t = a + e + GRID_VAR
        sh.append([a, e, a / t, e / t, GRID_VAR / t])
    sh = np.array(sh)
    for j, nm in enumerate(["seed", "draw", "share_seed", "share_draw", "share_grid"]):
        out[f"{nm}_ci"] = (float(np.percentile(sh[:, j], 2.5)), float(np.percentile(sh[:, j], 97.5)))
    return out


def per_seed(ev, key, seeds):
    out = {}
    cens_key = "cens_new" if key == "floor_new" else "cens_old"
    for s in sorted(seeds):
        v = np.array([e[key] for e in ev if e["seed"] == s])
        if len(v) == 0: continue
        sv = np.sort(v)
        out[s] = dict(floor=float(np.median(v)), tie_flag=bool(len(v) % 2 == 0 and sv[len(v) // 2 - 1] != sv[len(v) // 2]),
                      n_cens=int(sum(e[cens_key] for e in ev if e["seed"] == s)), n=len(v))
    return out


def wcsv(name, header, rows):
    path = os.path.join(RES, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        if LABEL: w.writerow(["label"] + list(header)); w.writerows([[LABEL] + list(r) for r in rows])
        else: w.writerow(header); w.writerows(rows)


def fmt(s, ci5, ci10):
    return (f"n={s['n']} mean {s['mean']:.2f} SD {s['sd']:.2f} median {s['median']:.2f} range {s['min']:.2f}-{s['max']:.2f} "
            f"p5 {s['p5']:.2f} [{ci5[0]:.2f}, {ci5[1]:.2f}] p10 {s['p10']:.2f} [{ci10[0]:.2f}, {ci10[1]:.2f}]")


def main():
    fams = sys.argv[1:] or [f for f in SPEC if glob.glob(os.path.join(P3, "raw", f, "seed*_draw*.npz"))]
    os.makedirs(RES, exist_ok=True)
    summary = {}; L = []; P = L.append
    for fam in fams:
        ev, real, gaps = load_family(fam)
        spec = SPEC[fam]; expected = len(spec["seeds"]) * spec["K"]
        P(f"## {fam}  (N={len(spec['seeds'])}, K={spec['K']}, B={spec['B']})")
        P(f"evaluation matrix: {len(ev)} of {expected} present; gaps: {gaps if gaps else 'none'}")
        keys = [k for k in ev[0].keys()]
        wcsv(f"{fam}_evaluations.csv", keys, [[e[k] for k in keys] for e in ev])
        walls = np.array([e["wall_s"] for e in ev])
        P(f"wall-clock per draw (includes any sleep time): median {np.median(walls):.0f} s, total {walls.sum()/3600:.2f} h; power labels: {sorted({e['power'] for e in ev})}")
        d = {"present": len(ev), "expected": expected}
        for sub, sset in SUBSETS.items():
            seeds = [s for s in spec["seeds"] if sset is None or s in sset]
            evs = [e for e in ev if e["seed"] in seeds]
            for rule, key in (("new", "floor_new"), ("old", "floor_old")):
                ps = per_seed(evs, key, seeds)
                fl = np.array([v["floor"] for v in ps.values()]); s = summarize(fl)
                ci5 = boot_ci(fl, 5); ci10 = boot_ci(fl, 10)
                pooled = np.array([e[key] for e in evs]); sp = summarize(pooled)
                cens_eval = sum(e["cens_new" if rule == "new" else "cens_old"] for e in evs)
                zero_eval = int(sum(e[key] == 0.0 for e in evs))
                n_cens_seeds = int(sum(v["floor"] >= 20.0 for v in ps.values())); n_zero_seeds = int(sum(v["floor"] == 0.0 for v in ps.values()))
                P(f"[{sub} | {rule} rule] per-seed floor (median over draws): {fmt(s, ci5, ci10)}; seeds censored at 20 cm {n_cens_seeds}; seeds at 0.00 cm {n_zero_seeds}; tie-flagged seeds {sum(v['tie_flag'] for v in ps.values())}")
                P(f"[{sub} | {rule} rule] pooled seed x draw: n={sp['n']} mean {sp['mean']:.2f} SD {sp['sd']:.2f} median {sp['median']:.2f} range {sp['min']:.2f}-{sp['max']:.2f} p5 {sp['p5']:.2f} p10 {sp['p10']:.2f}; censored evaluations {cens_eval}; 0.00 floors {zero_eval}")
                d[f"{sub}|{rule}"] = dict(per_seed=s, p5_ci=ci5, p10_ci=ci10, pooled=sp, censored_evals=int(cens_eval), zero_floor_evals=zero_eval,
                                          censored_seeds=n_cens_seeds, zero_seeds=n_zero_seeds, per_seed_values={int(k): v["floor"] for k, v in ps.items()})
            if sub == "all":
                psn = per_seed(evs, "floor_new", seeds); pso = per_seed(evs, "floor_old", seeds)
                wcsv(f"{fam}_floor_by_seed.csv", ["seed", "floor_new_median", "new_tie_flag", "new_n_censored_draws", "floor_old_median", "old_n_censored_draws"],
                     [[s, psn[s]["floor"], psn[s]["tie_flag"], psn[s]["n_cens"], pso[s]["floor"], pso[s]["n_cens"]] for s in sorted(psn)])
        st = sum(e["start_nonsig"] for e in ev)
        P(f"new-rule evaluations where depth 0 starts the run (start_nonsignificant): {st}; evaluations with depth-0 p at the minimum but depth 0 not significant: "
          f"{sum((e['p_depth0'] == 1 / (e['B'] + 1)) and (not e['depth0_sig']) for e in ev)}; evaluations with p_min above the minimum possible: {sum(e['p_min'] > 1 / (e['B'] + 1) + 1e-12 for e in ev)}")
        # variance decomposition, all seeds and seeds 101-110
        for sub, sset in SUBSETS.items():
            seeds = [s for s in spec["seeds"] if sset is None or s in sset]
            Fall = [np.array([e["floor_new"] for e in ev if e["seed"] == s]) for s in seeds if any(e["seed"] == s for e in ev)]
            Fnc = [np.array([e["floor_new"] for e in ev if e["seed"] == s and not e["cens_new"]]) for s in seeds]
            Fnc = [f for f in Fnc if len(f) >= 1]
            cens_frac = float(np.mean([e["cens_new"] for e in ev if e["seed"] in seeds]))
            dc = decomp(Fall)
            P(f"[{sub}] variance components (cm^2), censored floors included at 20 cm (censored fraction {cens_frac:.3f}, {dc['n_seeds']} seeds, {dc['n_evals']} evaluations): "
              f"seed {dc['seed']:.3f} [ANOVA raw {dc['seed_raw']:.3f}, REML {dc['reml_seed']:.3f}] draw {dc['draw']:.3f} [REML {dc['reml_draw']:.3f}] grid {dc['grid']:.3f}; "
              f"shares seed {dc['share_seed']:.3f} draw {dc['share_draw']:.3f} grid {dc['share_grid']:.3f}; bootstrap 95% CI shares seed {dc['share_seed_ci'][0]:.3f}-{dc['share_seed_ci'][1]:.3f}, "
              f"draw {dc['share_draw_ci'][0]:.3f}-{dc['share_draw_ci'][1]:.3f}; negative seed estimate set to 0: {dc['negative_seed_estimate']}")
            dn = None
            if len(Fnc) > 2 and sum(len(f) for f in Fnc) - len(Fnc) > 0:
                dn = decomp(Fnc)
                P(f"[{sub}] variance components, censored evaluations removed (unbalanced, {dn['n_seeds']} seeds, {dn['n_evals']} evaluations): seed {dn['seed']:.3f} [ANOVA raw {dn['seed_raw']:.3f}, REML {dn['reml_seed']:.3f}] "
                  f"draw {dn['draw']:.3f} [REML {dn['reml_draw']:.3f}] shares seed {dn['share_seed']:.3f} draw {dn['share_draw']:.3f} grid {dn['share_grid']:.3f}; "
                  f"CI shares seed {dn['share_seed_ci'][0]:.3f}-{dn['share_seed_ci'][1]:.3f}")
            d[f"variance_{sub}_all"] = dc; d[f"variance_{sub}_noncensored"] = dn
            rows = []
            for lab, r in (("censored_included", dc), ("censored_removed", dn)):
                if r is None: continue
                rows.append([lab, r["n_seeds"], r["n_evals"], r["seed_raw"], r["seed"], r["reml_seed"], r["draw"], r["reml_draw"], r["grid"],
                             r["share_seed"], r["share_draw"], r["share_grid"], *r["share_seed_ci"], *r["share_draw_ci"]])
            wcsv(f"{fam}_variance_components_{sub}.csv", ["variant", "n_seeds", "n_evals", "seed_anova_raw", "seed_anova_clamped", "seed_reml", "draw_anova", "draw_reml", "grid",
                                                         "share_seed", "share_draw", "share_grid", "share_seed_ci_lo", "share_seed_ci_hi", "share_draw_ci_lo", "share_draw_ci_hi"], rows)
        # classwise
        cf = {s: class_floor(r["recall"]) for s, r in real.items()}
        psn = per_seed(ev, "floor_new", spec["seeds"]); pso = per_seed(ev, "floor_old", spec["seeds"])
        for sub, sset in SUBSETS.items():
            seeds = [s for s in cf if sset is None or s in sset]
            v = np.array([cf[s][0] for s in seeds]); sc = summarize(v); ci5 = boot_ci(v, 5); ci10 = boot_ci(v, 10)
            vals, cnts = np.unique(v, return_counts=True)
            P(f"[{sub}] classwise minimum-class safety floor (real model, per seed): {fmt(sc, ci5, ci10)}; censored at 20 cm {sum(cf[s][1] for s in seeds)}; at 0.00 cm {int((v == 0).sum())}; distribution {dict(zip([round(float(x), 2) for x in vals], [int(c) for c in cnts]))}")
            d[f"classwise_{sub}"] = dict(summary=sc, p5_ci=ci5, p10_ci=ci10, censored=int(sum(cf[s][1] for s in seeds)), per_seed={int(s): cf[s][0] for s in seeds})
        wcsv(f"{fam}_classwise_by_seed.csv", ["seed", "classwise_floor_cm", "censored", "new_floor_median", "old_floor_median"],
             [[s, cf[s][0], cf[s][1], psn[s]["floor"] if s in psn else "", pso[s]["floor"] if s in pso else ""] for s in sorted(cf)])
        wcsv(f"{fam}_recall_by_seed_depth.csv", ["seed", "depth_cm", "recall_cirrhosis", "recall_healthy", "recall_significant_fibrosis", "min_class_recall", "accuracy"],
             [[s, round(float(GRID[i]), 4), *[round(float(x), 4) for x in real[s]["recall"][i]], round(float(real[s]["recall"][i].min()), 4), round(float(real[s]["counts"][i]) / 300, 4)]
              for s in sorted(real) for i in range(M)])
        rec = np.mean([r["recall"] for r in real.values()], axis=0)
        wcsv(f"{fam}_mean_recall_by_depth.csv", ["depth_cm", "recall_cirrhosis", "recall_healthy", "recall_significant_fibrosis", "mean_min_class_recall"],
             [[round(float(dd), 4)] + [round(float(x), 4) for x in rec[i]] + [round(float(np.mean([r['recall'][i].min() for r in real.values()])), 4)] for i, dd in enumerate(GRID)])
        summary[fam] = d; P("")
    if LABEL: L.insert(0, f"[{LABEL}]")
    open(os.path.join(RES, "analysis_summary.txt"), "w").write("\n".join(L) + "\n")
    if LABEL: summary["_label"] = LABEL
    json.dump(summary, open(os.path.join(RES, "analysis_summary.json"), "w"), indent=1, default=float)
    print("\n".join(L))


def manifest():
    lines = []
    for fn in sorted(os.listdir(RES)):
        p = os.path.join(RES, fn)
        if os.path.isfile(p) and fn != "MANIFEST.sha256":
            lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {fn}")
    open(os.path.join(RES, "MANIFEST.sha256"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
    manifest()

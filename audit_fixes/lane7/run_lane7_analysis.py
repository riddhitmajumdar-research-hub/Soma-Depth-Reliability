#!/usr/bin/env python3
"""
Lane 7: ResNet1D vs CNN 24-seed floor comparison.
Plan fixed in advance in PREREG.md. Read-only on all inputs.

Run from repo root:
    python3 audit_fixes/lane7/run_lane7_analysis.py
"""
import csv
import sys
import itertools
import numpy as np
from scipy import stats

from pathlib import Path as _P; REPO_ROOT = str(_P(__file__).resolve().parents[2])
CNN_CSV = f"{REPO_ROOT}/Source/logs/degradation_aware_final_results.csv"
RESNET_CSV = f"{REPO_ROOT}/Source/logs/degradation_aware_resnet_final_results.csv"
CONSTANTS_PATH = f"{REPO_ROOT}/Source/constants.py"

RNG_SEED = 20260101
N_BOOTSTRAP = 10000


def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def get_gate_constant():
    with open(CONSTANTS_PATH) as f:
        for line in f:
            if line.strip().startswith("MAX_VALID_DEPTH_CM"):
                val = line.split("=")[1].strip()
                return float(val)
    raise RuntimeError("MAX_VALID_DEPTH_CM not found in constants.py")


def to_bool(s):
    return str(s).strip().lower() == "true"


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def percentiles_three_ways(x, q):
    x = np.asarray(x, dtype=float)
    linear = np.percentile(x, q, method="linear")
    weibull = np.percentile(x, q, method="weibull")
    nearest = np.percentile(x, q, method="closest_observation")
    return linear, weibull, nearest


def bootstrap_hl_ind(x, y, rng, n_boot=N_BOOTSTRAP):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    obs = np.median([xi - yi for xi in x for yi in y])
    boots = np.empty(n_boot)
    nx, ny = len(x), len(y)
    for b in range(n_boot):
        xs = rng.choice(x, size=nx, replace=True)
        ys = rng.choice(y, size=ny, replace=True)
        boots[b] = np.median([xi - yi for xi in xs for yi in ys])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, lo, hi


def bootstrap_hl_paired(diffs, rng, n_boot=N_BOOTSTRAP):
    diffs = np.asarray(diffs, dtype=float)
    obs = np.median(diffs)
    n = len(diffs)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(diffs, size=n, replace=True)
        boots[b] = np.median(s)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return obs, lo, hi


def exact_or_montecarlo_permutation_p(x, y, rng, mc_trials=100000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    n_total = nx + ny
    combined = np.concatenate([x, y])
    obs_stat = np.sum(stats.rankdata(combined)[:nx])  # rank-sum for group x

    from math import comb
    n_combinations = comb(n_total, nx)

    if n_combinations <= 200000:
        ranks = stats.rankdata(combined)
        stat_dist = []
        for combo in itertools.combinations(range(n_total), nx):
            stat_dist.append(np.sum(ranks[list(combo)]))
        stat_dist = np.array(stat_dist)
        two_sided_count = np.sum(
            np.abs(stat_dist - stat_dist.mean()) >= np.abs(obs_stat - stat_dist.mean()) - 1e-9
        )
        p_exact_perm = two_sided_count / len(stat_dist)
        method_used = f"full enumeration ({len(stat_dist)} permutations)"
        return p_exact_perm, method_used
    else:
        ranks = stats.rankdata(combined)
        count = 0
        for _ in range(mc_trials):
            perm = rng.permutation(n_total)
            stat = np.sum(ranks[perm[:nx]])
            if abs(stat - (n_total * (n_total + 1) / 2) * (nx / n_total)) >= abs(
                obs_stat - (n_total * (n_total + 1) / 2) * (nx / n_total)
            ) - 1e-9:
                count += 1
        p_mc = count / mc_trials
        method_used = f"Monte Carlo ({mc_trials} permutations)"
        return p_mc, method_used


def main():
    rng = np.random.default_rng(RNG_SEED)

    hdr("0. FILE / STATUS CHECK")
    print(f"CNN CSV:    {CNN_CSV}")
    print(f"ResNet CSV: {RESNET_CSV}")
    gate = get_gate_constant()
    print(f"MAX_VALID_DEPTH_CM (imported from constants.py): {gate}")

    cnn_rows = load_csv(CNN_CSV)
    resnet_rows = load_csv(RESNET_CSV)
    print(f"CNN rows: {len(cnn_rows)} | columns: {list(cnn_rows[0].keys())}")
    print(f"ResNet rows: {len(resnet_rows)} | columns: {list(resnet_rows[0].keys())}")

    cnn_seeds = [int(r["seed"]) for r in cnn_rows]
    resnet_seeds = [int(r["seed"]) for r in resnet_rows]
    print(f"CNN seeds: {cnn_seeds}")
    print(f"ResNet seeds: {resnet_seeds}")
    full_24 = set(range(101, 125))
    missing_from_resnet = sorted(full_24 - set(resnet_seeds))
    print(f"Seeds missing from ResNet CSV (expected 101-124): {missing_from_resnet}")
    if missing_from_resnet:
        print(
            "STATUS: ResNet1D sweep INCOMPLETE in this pass "
            f"({len(resnet_seeds)}/24 seeds present). "
            "All statistics below computed on the PROVISIONAL n= available now, "
            "NOT the certified n=24 comparison."
        )

    cnn_floor = {int(r["seed"]): float(r["naive_floor_cm"]) for r in cnn_rows}
    resnet_floor = {int(r["seed"]): float(r["naive_floor_cm"]) for r in resnet_rows}
    cnn_anom = {int(r["seed"]): to_bool(r["anomalous_recovery"]) for r in cnn_rows}
    resnet_anom = {int(r["seed"]): to_bool(r["anomalous_recovery"]) for r in resnet_rows}

    matched_seeds = sorted(set(cnn_floor) & set(resnet_floor))
    print(f"\nMatched seeds (present in BOTH files, n={len(matched_seeds)}): {matched_seeds}")

    cnn_floor_matched = [cnn_floor[s] for s in matched_seeds]
    resnet_floor_matched = [resnet_floor[s] for s in matched_seeds]
    cnn_floor_all24 = [cnn_floor[s] for s in sorted(cnn_floor)]

    hdr("1. DESCRIPTIVE STATS — FLOOR SETS")
    for label, data, n_expected in [
        ("CNN (n=24, full)", cnn_floor_all24, 24),
        (f"ResNet1D (n={len(resnet_floor)}, PROVISIONAL — sweep incomplete)"
         if missing_from_resnet else f"ResNet1D (n={len(resnet_floor)})",
         [resnet_floor[s] for s in sorted(resnet_floor)], 24),
    ]:
        arr = np.asarray(data, dtype=float)
        p5_lin, p5_web, p5_near = percentiles_three_ways(arr, 5)
        p10_lin, p10_web, p10_near = percentiles_three_ways(arr, 10)
        print(f"\n{label}")
        print(f"  n      = {len(arr)}")
        print(f"  mean   = {arr.mean():.6f} cm")
        print(f"  SD(ddof=1) = {arr.std(ddof=1):.6f} cm")
        print(f"  median = {np.median(arr):.6f} cm")
        print(f"  5th pct  : linear={p5_lin:.6f}  weibull={p5_web:.6f}  nearest-rank={p5_near:.6f}")
        print(f"  10th pct : linear={p10_lin:.6f}  weibull={p10_web:.6f}  nearest-rank={p10_near:.6f}")
        print(f"  range  = [{arr.min():.6f}, {arr.max():.6f}] cm")

    hdr("2. MANN-WHITNEY U (independent, CNN[n=24] vs ResNet1D[available n])")
    cnn_all = np.array(cnn_floor_all24, dtype=float)
    resnet_all = np.array([resnet_floor[s] for s in sorted(resnet_floor)], dtype=float)
    mw_res = stats.mannwhitneyu(cnn_all, resnet_all, alternative="two-sided", method="auto")
    print(f"scipy mannwhitneyu: U={mw_res.statistic}, p={mw_res.pvalue:.8f}")
    ties_present = len(np.unique(cnn_all)) + len(np.unique(resnet_all)) < len(cnn_all) + len(resnet_all)
    print(f"Ties present in combined pool: {ties_present}")

    perm_p, perm_method = exact_or_montecarlo_permutation_p(cnn_all, resnet_all, rng)
    print(f"Cross-check permutation test: p={perm_p:.8f} ({perm_method})")
    match = "MATCH" if abs(perm_p - mw_res.pvalue) < 0.01 else "MISMATCH"
    print(f"scipy exact/approx p vs permutation p: {match} (|diff|={abs(perm_p - mw_res.pvalue):.6f})")

    hdr("3. HODGES-LEHMANN SHIFT (independent samples, CNN[24] vs ResNet1D[avail])")
    hl_obs, hl_lo, hl_hi = bootstrap_hl_ind(resnet_all, cnn_all, rng)
    print(f"HL shift (ResNet1D - CNN) = {hl_obs:.6f} cm")
    print(f"10,000-resample bootstrap 95% CI: [{hl_lo:.6f}, {hl_hi:.6f}] cm")

    hdr("4. PAIRED WILCOXON SIGNED-RANK (matched seeds only, n="
        f"{len(matched_seeds)}, PROVISIONAL if <24)")
    diffs = np.array([resnet_floor[s] - cnn_floor[s] for s in matched_seeds], dtype=float)
    print(f"Per-seed differences (ResNet1D - CNN):")
    for s, d in zip(matched_seeds, diffs):
        print(f"  seed {s}: ResNet={resnet_floor[s]:.4f}  CNN={cnn_floor[s]:.4f}  diff={d:.4f}")
    n_zero_diff = int(np.sum(diffs == 0))
    print(f"Number of zero differences (Wilcoxon zero_method='wilcox' drops these): {n_zero_diff}")
    if n_zero_diff == len(diffs):
        print("All differences are zero — Wilcoxon test undefined.")
    else:
        try:
            wres = stats.wilcoxon(diffs, zero_method="wilcox", correction=False, mode="auto")
            print(f"Wilcoxon signed-rank: statistic={wres.statistic}, p={wres.pvalue:.8f}")
        except ValueError as e:
            print(f"Wilcoxon test raised: {e}")

    hl_p_obs, hl_p_lo, hl_p_hi = bootstrap_hl_paired(diffs, rng)
    print(f"Paired HL (median of paired diffs) = {hl_p_obs:.6f} cm")
    print(f"10,000-resample bootstrap 95% CI: [{hl_p_lo:.6f}, {hl_p_hi:.6f}] cm")

    hdr(f"5. FRACTION OF SEEDS WITH FLOOR < {gate} cm")
    for label, floors in [
        ("CNN (n=24)", cnn_floor_all24),
        (f"ResNet1D (n={len(resnet_all)}, PROVISIONAL" + (")" if missing_from_resnet else ")"), list(resnet_all)),
    ]:
        arr = np.asarray(floors)
        k = int(np.sum(arr < gate))
        n = len(arr)
        lo, hi = clopper_pearson(k, n)
        print(f"{label}: {k}/{n} = {k/n:.4f}  Clopper-Pearson 95% CI: [{lo:.4f}, {hi:.4f}]")

    hdr("6. ANOMALOUS_RECOVERY COUNTS")
    cnn_anom_count = sum(cnn_anom.values())
    resnet_anom_count = sum(resnet_anom.values())
    print(f"CNN: {cnn_anom_count}/{len(cnn_anom)} = {cnn_anom_count/len(cnn_anom):.4f}")
    print(f"ResNet1D: {resnet_anom_count}/{len(resnet_anom)} = {resnet_anom_count/len(resnet_anom):.4f} "
          f"(n={len(resnet_anom)}, PROVISIONAL if <24: {bool(missing_from_resnet)})")

    hdr("7. RESNET1D RESTART DATA (CNN file has no equivalent columns -> N/A)")
    collapse_counts = {int(r["seed"]): int(r["real_model_collapse_count"]) for r in resnet_rows}
    escaped = {int(r["seed"]): int(r["real_model_escaped_on_attempt"]) for r in resnet_rows}
    print("Per-seed collapse_count / escaped_on_attempt:")
    for s in sorted(collapse_counts):
        print(f"  seed {s}: collapses={collapse_counts[s]}  escaped_on_attempt={escaped[s]}")
    n_with_restart = sum(1 for s in collapse_counts if collapse_counts[s] > 0)
    n_total_resnet = len(collapse_counts)
    lo, hi = clopper_pearson(n_with_restart, n_total_resnet)
    print(f"\nOverall restart rate: {n_with_restart}/{n_total_resnet} = "
          f"{n_with_restart/n_total_resnet:.4f}  Clopper-Pearson 95% CI: [{lo:.4f}, {hi:.4f}]")
    print(f"(PROVISIONAL — based on {n_total_resnet}/24 seeds; sweep incomplete: {bool(missing_from_resnet)})")

    hdr(f"8. RESNET1D 10TH-PERCENTILE GATE CANDIDATE (n={len(resnet_all)}, PROVISIONAL)")
    p10_lin, p10_web, p10_near = percentiles_three_ways(resnet_all, 10)
    print(f"numpy-linear:  {p10_lin:.6f} cm")
    print(f"Weibull:       {p10_web:.6f} cm")
    print(f"nearest-rank:  {p10_near:.6f} cm")
    boot_p10 = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sample = rng.choice(resnet_all, size=len(resnet_all), replace=True)
        boot_p10[b] = np.percentile(sample, 10, method="linear")
    lo, hi = np.percentile(boot_p10, [2.5, 97.5])
    print(f"Bootstrap 95% CI on linear 10th percentile: [{lo:.6f}, {hi:.6f}] cm")
    print(f"Compare to record's certified gate MAX_VALID_DEPTH_CM = {gate} cm "
          f"(that value was derived from the CNN's 24-seed distribution, not ResNet1D's — "
          "stated here for context only, not a like-for-like check).")

    hdr("9. SEEDS WHERE RESNET1D FLOOR EXACTLY EQUALS CNN FLOOR")
    exact_matches = [s for s in matched_seeds if resnet_floor[s] == cnn_floor[s]]
    if exact_matches:
        for s in exact_matches:
            print(f"  seed {s}: both floors = {cnn_floor[s]:.6f} cm")
    else:
        print("  None — no matched seed has an exactly equal floor between models.")

    hdr("10. RAW DATA DUMP FOR REPORT CROSS-CHECK")
    print("CNN seed -> floor:")
    for s in sorted(cnn_floor):
        print(f"  {s}: {cnn_floor[s]}")
    print("ResNet1D seed -> floor:")
    for s in sorted(resnet_floor):
        print(f"  {s}: {resnet_floor[s]}")

    hdr("DONE")


if __name__ == "__main__":
    main()

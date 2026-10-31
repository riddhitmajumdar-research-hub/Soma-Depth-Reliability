#!/usr/bin/env python3
"""
Lane 7 follow-up v2: censoring check, sign-flip permutation test, below-gate
seed lists. Plan fixed in PREREG_v2.md. Read-only on all inputs.

Run from repo root: python3 audit_fixes/lane7/run_lane7_v2_analysis.py
"""
import csv
import sys
import numpy as np
from scipy import stats

from pathlib import Path as _P; REPO_ROOT = str(_P(__file__).resolve().parents[2])
SOURCE_DIR = f"{REPO_ROOT}/Source"
sys.path.insert(0, SOURCE_DIR)
CNN_CSV = f"{SOURCE_DIR}/logs/degradation_aware_final_results.csv"
RESNET_CSV = f"{SOURCE_DIR}/logs/degradation_aware_resnet_final_results.csv"
CONSTANTS_PATH = f"{SOURCE_DIR}/constants.py"

RNG_SEED = 20260101
N_PERM = 100000


def hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get_gate_constant():
    with open(CONSTANTS_PATH) as f:
        for line in f:
            if line.strip().startswith("MAX_VALID_DEPTH_CM"):
                return float(line.split("=")[1].strip())
    raise RuntimeError("MAX_VALID_DEPTH_CM not found")


def gehan_wilcoxon(times, events, group):
    """Hand-rolled Gehan (1965) generalized Wilcoxon test for two-sample
    right-censored data, used because `lifelines` is not installed in this
    environment. For every pair (i, j) with i in group 0 and j in group 1,
    score U_ij = +1 if it's known that i's true time > j's true time
    (i.e. time_i > time_j and either i is uncensored, or i is censored and
    time_i >= time_j... standard Gehan scoring below), -1 if the reverse is
    known, 0 if the comparison is ambiguous given censoring. Sum all U_ij
    for group 0 -> Gehan statistic; test via normal approximation using the
    standard Mann-Whitney-type variance formula restricted to the *known*
    comparisons (Gehan's original permutation variance formula).

    events: 1 = observed (uncensored), 0 = right-censored.
    This is a standard, textbook implementation (Gehan, Biometrika 1965;
    also given in Kalbfleisch & Prentice, 'The Statistical Analysis of
    Failure Time Data', 2nd ed., section 6.3) -- not a novel derivation.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    group = np.asarray(group, dtype=int)
    n = len(times)

    def compare(i, j):
        ti, tj = times[i], times[j]
        ei, ej = events[i], events[j]
        if ei == 1 and ej == 1:
            if ti > tj:
                return 1
            elif ti < tj:
                return -1
            else:
                return 0
        if ei == 1 and ej == 0:
            if ti > tj:
                return 0  # censored j could exceed ti -- ambiguous
            else:
                return -1  # ti <= tj (censored) -> i's true time <= tj <= j's true time
        if ei == 0 and ej == 1:
            if tj > ti:
                return 0
            else:
                return 1
        # both censored -- no valid comparison
        return 0

    U = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            U[i, j] = compare(i, j)

    Wi = U.sum(axis=1)  # sum over j for each i
    W_group0 = Wi[group == 0].sum()

    var_W = (Wi ** 2).sum() * (np.sum(group == 0) * np.sum(group == 1)) / (n * (n - 1))
    if var_W <= 0:
        z = 0.0
    else:
        z = W_group0 / np.sqrt(var_W)
    p_two_sided = 2 * (1 - stats.norm.cdf(abs(z)))
    return W_group0, z, p_two_sided


def logrank_unweighted(times, events, group):
    """Standard unweighted log-rank test, two groups, via the usual
    observed-minus-expected sum at each distinct event time (Mantel-Haenszel
    form). Textbook implementation (Kalbfleisch & Prentice section 1.5)."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    group = np.asarray(group, dtype=int)

    event_times = sorted(set(times[events == 1]))
    O_minus_E = 0.0
    var_sum = 0.0
    for t in event_times:
        at_risk = times >= t
        n_total = at_risk.sum()
        n0 = (at_risk & (group == 0)).sum()
        n1 = (at_risk & (group == 1)).sum()
        d_total = ((times == t) & (events == 1)).sum()
        d0 = ((times == t) & (events == 1) & (group == 0)).sum()
        if n_total <= 1:
            continue
        expected0 = d_total * n0 / n_total
        O_minus_E += d0 - expected0
        var_sum += (d_total * (n0 / n_total) * (n1 / n_total) * (n_total - d_total)) / (n_total - 1)

    if var_sum <= 0:
        chi2 = 0.0
    else:
        chi2 = O_minus_E ** 2 / var_sum
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return O_minus_E, chi2, p


def sign_flip_test(diffs, rng, n_perm=N_PERM):
    diffs = np.asarray(diffs, dtype=float)
    obs_mean = diffs.mean()
    n = len(diffs)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([1.0, -1.0], size=n)
        flipped_mean = (diffs * signs).mean()
        if abs(flipped_mean) >= abs(obs_mean) - 1e-12:
            count += 1
    p = count / n_perm
    return obs_mean, p


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def main():
    rng = np.random.default_rng(RNG_SEED)
    gate = get_gate_constant()

    hdr("0. STATUS CHECK (re-confirmed)")
    cnn_rows = load_csv(CNN_CSV)
    resnet_rows = load_csv(RESNET_CSV)
    print(f"CNN rows: {len(cnn_rows)}  ResNet rows: {len(resnet_rows)}")
    print(f"MAX_VALID_DEPTH_CM = {gate}")

    cnn_floor = {int(r["seed"]): float(r["naive_floor_cm"]) for r in cnn_rows}
    resnet_floor = {int(r["seed"]): float(r["naive_floor_cm"]) for r in resnet_rows}

    from degradation_aware_final_sweep import EVAL_DEPTHS_CM
    grid_max = float(EVAL_DEPTHS_CM[-1])
    print(f"EVAL_DEPTHS_CM[-1] (grid max, imported) = {grid_max}")

    hdr("1. CENSORING")
    print("Quoted floor-assignment code:\n")
    import subprocess
    for fname, label in [
        ("degradation_aware_final_sweep.py", "CNN sweep"),
        ("degradation_aware_resnet_final_sweep.py", "ResNet1D sweep"),
    ]:
        out = subprocess.run(
            ["grep", "-n", "-B1", "-A2", "floor_depth = EVAL_DEPTHS_CM", f"{SOURCE_DIR}/{fname}"],
            capture_output=True, text=True,
        ).stdout
        print(f"--- {label} ({fname}) ---")
        print(out)

    cnn_censored = [s for s, f in cnn_floor.items() if f == grid_max]
    resnet_censored = [s for s, f in resnet_floor.items() if f == grid_max]
    print(f"CNN seeds with floor == grid max ({grid_max}) [right-censored]: {cnn_censored} "
          f"({len(cnn_censored)}/{len(cnn_floor)})")
    print(f"ResNet1D seeds with floor == grid max ({grid_max}) [right-censored]: {resnet_censored} "
          f"({len(resnet_censored)}/{len(resnet_floor)})")

    any_censored = bool(cnn_censored or resnet_censored)
    if any_censored:
        print("\nCensoring present -- running Gehan-Wilcoxon and log-rank as specified.")
        cnn_seeds_sorted = sorted(cnn_floor)
        resnet_seeds_sorted = sorted(resnet_floor)
        times = [cnn_floor[s] for s in cnn_seeds_sorted] + [resnet_floor[s] for s in resnet_seeds_sorted]
        events = ([0 if s in cnn_censored else 1 for s in cnn_seeds_sorted] +
                  [0 if s in resnet_censored else 1 for s in resnet_seeds_sorted])
        group = [0] * len(cnn_seeds_sorted) + [1] * len(resnet_seeds_sorted)

        W, z, p_gehan = gehan_wilcoxon(times, events, group)
        print(f"\nGehan-Wilcoxon (hand-rolled, lifelines not installed): "
              f"W(group0=CNN)={W:.4f}, z={z:.4f}, two-sided p={p_gehan:.8f}")

        OmE, chi2, p_lr = logrank_unweighted(times, events, group)
        print(f"Log-rank (unweighted, hand-rolled): O-E(group0=CNN)={OmE:.4f}, "
              f"chi2={chi2:.4f}, p={p_lr:.8f}")

        print("\n(For comparison, v1's uncensored Mann-Whitney U on the same "
              "floor values: see lane7_output_raw.txt section 2. Both versions "
              "reported here per the pre-registered plan; not reconciled.)")
    else:
        print("\nNo censored seeds in either available set -- Gehan-Wilcoxon / "
              "log-rank not applicable; v1's uncensored Mann-Whitney/HL stand "
              "as the correct comparison.")

    hdr("2. SEEDS BELOW 6.50 cm GATE (full lists)")
    cnn_below = sorted(s for s, f in cnn_floor.items() if f < gate)
    resnet_below = sorted(s for s, f in resnet_floor.items() if f < gate)
    print(f"CNN (n={len(cnn_floor)}): {len(cnn_below)} seeds below {gate} cm: {cnn_below}")
    for s in cnn_below:
        print(f"    seed {s}: {cnn_floor[s]:.6f} cm")
    print(f"\nResNet1D (n={len(resnet_floor)}, PROVISIONAL): {len(resnet_below)} seeds below {gate} cm: {resnet_below}")
    for s in resnet_below:
        print(f"    seed {s}: {resnet_floor[s]:.6f} cm")

    hdr("3. PAIRED SIGN-FLIP PERMUTATION TEST (matched seeds, zero diffs kept)")
    matched_seeds = sorted(set(cnn_floor) & set(resnet_floor))
    diffs = np.array([resnet_floor[s] - cnn_floor[s] for s in matched_seeds], dtype=float)
    n_zero = int(np.sum(diffs == 0))
    print(f"n matched seeds = {len(matched_seeds)}, n zero-differences kept in this test = {n_zero}")
    obs_mean, p_signflip = sign_flip_test(diffs, rng)
    print(f"Observed mean difference (ResNet1D - CNN) = {obs_mean:.6f} cm")
    print(f"Sign-flip permutation test ({N_PERM} flips): two-sided p = {p_signflip:.6f}")
    print("(Compare to v1's Wilcoxon signed-rank p=0.03997423, which drops the "
          "1 zero-difference pair -- see lane7_output_raw.txt section 4d. "
          "Both reported, not reconciled.)")

    hdr("DONE")


if __name__ == "__main__":
    main()

"""
LANE 2 audit script -- repeated-CV significance correction + safety-gate
threshold stability.

READ-ONLY on the rest of the repo. Reads only:
  - Source/logs/log05_log11_grouped_cv_verification.csv
  - Source/logs/log11_grouped_paired_ttests.csv        (check values only)
  - Source/logs/log11_repeated_grouped_cv_raw.csv
  - Source/logs/degradation_aware_final_results.csv
  - Source/constants.py                                 (MAX_VALID_DEPTH_CM)

Writes only inside audit_fixes/lane2/. Does not retrain anything, does not
touch the running ResNet1D sweep or any file outside this lane.

Run from repo root:
    python3 audit_fixes/lane2/analysis.py
"""

import sys
import os
import csv
import json
from itertools import combinations

import numpy as np
from scipy import stats

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_DIR = os.path.join(REPO_ROOT, "Source")
LOGS_DIR = os.path.join(SOURCE_DIR, "logs")
LANE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SOURCE_DIR)
from constants import MAX_VALID_DEPTH_CM  # noqa: E402  (imported, not hardcoded)

OUT = {}
LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


# ---------------------------------------------------------------------------
# Task 1: locate the saved per-fold data
# ---------------------------------------------------------------------------
log("=" * 78)
log("TASK 1: locating saved per-fold accuracies")
log("=" * 78)

single_split_path = os.path.join(LOGS_DIR, "log05_log11_grouped_cv_verification.csv")
repeated_raw_path = os.path.join(LOGS_DIR, "log11_repeated_grouped_cv_raw.csv")
orig_ttest_path = os.path.join(LOGS_DIR, "log11_grouped_paired_ttests.csv")
repeated_summary_path = os.path.join(LOGS_DIR, "log11_repeated_grouped_cv_summary.csv")
floors_path = os.path.join(LOGS_DIR, "degradation_aware_final_results.csv")

for p in [single_split_path, repeated_raw_path, orig_ttest_path,
          repeated_summary_path, floors_path]:
    log(f"  {'FOUND' if os.path.exists(p) else 'MISSING'}: {os.path.relpath(p, REPO_ROOT)}")

log("\n  Per-depth p-values / per-seed significance flags for the 24-seed sweep:")
log("  NOT SAVED -- degradation_aware_final_sweep.py (run_one_seed()) computes a")
log("  per-depth permutation p-value and a BH-FDR significance flag per depth")
log("  internally, but only writes {seed, naive_floor_cm, last_significant_depth_cm,")
log("  anomalous_recovery} to logs/degradation_aware_final_results.csv -- the")
log("  per-depth intermediate values are discarded after each seed. Task 6's")
log("  per-depth BH-FDR-by-seed table therefore cannot be produced without rerunning")
log("  training, which this lane is not permitted to do.")

# Single-split 5-fold accuracies for the grouped comparison (log11_paired_grouped_cv rows)
single_split = {"baseline": [], "cnn": [], "resnet1d": []}
with open(single_split_path) as f:
    for row in csv.DictReader(f):
        if row["item"] == "log11_paired_grouped_cv" and row["fold"] in ("1", "2", "3", "4", "5"):
            single_split["baseline"].append(float(row["baseline_acc"]))
            single_split["cnn"].append(float(row["cnn_acc"]))
            single_split["resnet1d"].append(float(row["resnet1d_acc"]))

log(f"\n  Single-split 5-fold accuracies recovered from log05_log11_grouped_cv_verification.csv:")
for k, v in single_split.items():
    log(f"    {k}: {v}")

# Repeated CV raw (15 reps x 5 folds = 75 rows)
rep_rows = []
with open(repeated_raw_path) as f:
    for row in csv.DictReader(f):
        rep_rows.append({
            "repetition": int(row["repetition"]),
            "fold": int(row["fold"]),
            "baseline_acc": float(row["baseline_acc"]),
            "cnn_acc": float(row["cnn_acc"]),
            "resnet1d_acc": float(row["resnet1d_acc"]),
        })
log(f"\n  Repeated-CV raw rows: {len(rep_rows)} (expect 15 reps x 5 folds = 75)")

# ---------------------------------------------------------------------------
# Task 2: reproduce single-split paired t-tests, check against record values
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("TASK 2: single-split paired t-tests (n=5 folds)")
log("=" * 78)

CHECK_SINGLE = {
    "cnn_vs_baseline": {"p": 0.0175, "d": 1.54},
    "resnet1d_vs_baseline": {"p": 0.0078, "d": 2.76},
    "resnet1d_vs_cnn": {"p": 0.2420, "d": 0.59},
}


def cohens_d_paired(diffs):
    diffs = np.asarray(diffs, dtype=float)
    return diffs.mean() / diffs.std(ddof=1)


single_results = {}
pairs = [("cnn", "baseline", "cnn_vs_baseline"),
         ("resnet1d", "baseline", "resnet1d_vs_baseline"),
         ("resnet1d", "cnn", "resnet1d_vs_cnn")]

for a, b, name in pairs:
    x = np.array(single_split[a])
    y = np.array(single_split[b])
    diffs = x - y
    t_stat, p_val = stats.ttest_rel(x, y)
    d = cohens_d_paired(diffs)
    check = CHECK_SINGLE[name]
    p_match = "MATCH" if abs(p_val - check["p"]) < 5e-3 else "MISMATCH"
    d_match = "MATCH" if abs(d - check["d"]) < 5e-2 else "MISMATCH"
    single_results[name] = {
        "mean_diff": float(diffs.mean()), "t_stat": float(t_stat), "p_value": float(p_val),
        "cohens_d": float(d), "n": len(diffs),
        "check_p": check["p"], "check_p_result": p_match,
        "check_d": check["d"], "check_d_result": d_match,
    }
    log(f"\n{name}: mean_diff={diffs.mean():.4f}  t={t_stat:.3f}  p={p_val:.4f}  d={d:.3f}")
    log(f"  CHECK p={check['p']} -> {p_match} (computed {p_val:.4f})")
    log(f"  CHECK d={check['d']} -> {d_match} (computed {d:.3f})")

# Repeated-CV p=0.0002 check -- this is the sign-flip permutation p-value
# already on disk in log11_repeated_grouped_cv_summary.csv, produced by
# repeated_grouped_cv_log11.py. Reproduced here by re-reading that file
# (not recomputed, since it depends on a specific RNG stream already run).
log("\n--- repeated-CV p=0.0002 check (record's stated repeated-CV significance) ---")
with open(repeated_summary_path) as f:
    for row in csv.DictReader(f):
        p = float(row["permutation_p_value"])
        match = "MATCH" if abs(round(p, 4) - 0.0002) < 1e-4 else "MISMATCH"
        log(f"  {row['comparison']}: permutation_p_value={p:.4f} (rounds to {round(p,4)}) -> {match} vs record's 0.0002")
        log(f"    NOTE: this is a sign-flip PERMUTATION p-value, not a corrected")
        log(f"    resampled t-test p-value -- see Task 3 for the Nadeau-Bengio result,")
        log(f"    which is the statistically appropriate test for non-independent")
        log(f"    repeated-CV folds and is NOT the same number.")

# ---------------------------------------------------------------------------
# Task 3: Nadeau-Bengio corrected resampled t-test
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("TASK 3: Nadeau-Bengio corrected resampled t-test")
log("=" * 78)
log("t = mean / sqrt((1/J + n_test/n_train) * s^2),  n_test/n_train = 1/4")


def nadeau_bengio_test(diffs, n_test_over_n_train=0.25):
    diffs = np.asarray(diffs, dtype=float)
    J = len(diffs)
    mean_diff = diffs.mean()
    s2 = diffs.var(ddof=1)
    denom = np.sqrt((1.0 / J + n_test_over_n_train) * s2)
    t_stat = mean_diff / denom
    df = J - 1
    p_val = 2 * stats.t.sf(abs(t_stat), df)
    t_crit = stats.t.ppf(0.975, df)
    ci_lo = mean_diff - t_crit * denom
    ci_hi = mean_diff + t_crit * denom
    return {
        "J": J, "df": df, "mean_diff": float(mean_diff), "t_stat": float(t_stat),
        "p_value": float(p_val), "ci_95_lo": float(ci_lo), "ci_95_hi": float(ci_hi),
    }


nb_results = {"repeated_75": {}, "single_5": {}}

for a, b, name in pairs:
    diffs75 = np.array([r[f"{a}_acc"] for r in rep_rows]) - np.array([r[f"{b}_acc"] for r in rep_rows])
    res75 = nadeau_bengio_test(diffs75)
    nb_results["repeated_75"][name] = res75
    log(f"\n[75 diffs] {name}: J={res75['J']} df={res75['df']} mean_diff={res75['mean_diff']:.4f}")
    log(f"  Nadeau-Bengio t={res75['t_stat']:.3f}  p={res75['p_value']:.6f}  "
        f"95% CI=[{res75['ci_95_lo']:.4f}, {res75['ci_95_hi']:.4f}]")

    diffs5 = np.array(single_split[a]) - np.array(single_split[b])
    res5 = nadeau_bengio_test(diffs5)
    nb_results["single_5"][name] = res5
    log(f"[5 diffs]  {name}: J={res5['J']} df={res5['df']} mean_diff={res5['mean_diff']:.4f}")
    log(f"  Nadeau-Bengio t={res5['t_stat']:.3f}  p={res5['p_value']:.6f}  "
        f"95% CI=[{res5['ci_95_lo']:.4f}, {res5['ci_95_hi']:.4f}]")

# ---------------------------------------------------------------------------
# Task 4: 24 certified floors
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("TASK 4: 24 certified floors (seeds 101-124)")
log("=" * 78)

seeds, floors, anomalous_flags = [], [], []
with open(floors_path) as f:
    reader = csv.DictReader(f)
    log(f"  Columns found: {reader.fieldnames}")
    for row in reader:
        seeds.append(int(row["seed"]))
        floors.append(float(row["naive_floor_cm"]))
        anomalous_flags.append(row["anomalous_recovery"].strip() == "True")

floors = np.array(floors)
n = len(floors)
mean_f = floors.mean()
sd_f = floors.std(ddof=1)
median_f = np.median(floors)
p10_np = np.percentile(floors, 10, interpolation="linear") if "interpolation" in np.percentile.__doc__ else np.percentile(floors, 10)
p5_np = np.percentile(floors, 5)

CHECK_FLOOR = {"mean": 8.48, "sd": 1.81, "median": 8.00, "p10": 6.50, "p5": 5.65}


def chk(val, ref, tol):
    return "MATCH" if abs(val - ref) < tol else "MISMATCH"


log(f"\n  n={n}")
log(f"  mean={mean_f:.4f}  CHECK 8.48 -> {chk(mean_f, CHECK_FLOOR['mean'], 5e-3)}")
log(f"  SD(ddof=1)={sd_f:.4f}  CHECK 1.81 -> {chk(sd_f, CHECK_FLOOR['sd'], 5e-3)}")
log(f"  median={median_f:.4f}  CHECK 8.00 -> {chk(median_f, CHECK_FLOOR['median'], 1e-9)}")
log(f"  10th pct (numpy linear)={p10_np:.4f}  CHECK 6.50 -> {chk(p10_np, CHECK_FLOOR['p10'], 5e-3)}")
log(f"  5th pct (numpy linear)={p5_np:.4f}  CHECK 5.65 -> {chk(p5_np, CHECK_FLOOR['p5'], 5e-3)}")

# (a) bootstrap 95% CI on 5th and 10th percentiles, 10000 resamples
rng = np.random.default_rng(42)
N_BOOT = 10000
boot_p5 = np.empty(N_BOOT)
boot_p10 = np.empty(N_BOOT)
for i in range(N_BOOT):
    sample = rng.choice(floors, size=n, replace=True)
    boot_p5[i] = np.percentile(sample, 5)
    boot_p10[i] = np.percentile(sample, 10)
ci_p5 = np.percentile(boot_p5, [2.5, 97.5])
ci_p10 = np.percentile(boot_p10, [2.5, 97.5])
log(f"\n  (a) Bootstrap 95% CI, 10,000 resamples of individual floors:")
log(f"      5th percentile:  point={p5_np:.4f}  95% CI=[{ci_p5[0]:.4f}, {ci_p5[1]:.4f}]")
log(f"      10th percentile: point={p10_np:.4f}  95% CI=[{ci_p10[0]:.4f}, {ci_p10[1]:.4f}]")

# (b) leave-one-out range of the 10th percentile
loo_p10 = []
for i in range(n):
    subset = np.delete(floors, i)
    loo_p10.append(np.percentile(subset, 10))
loo_p10 = np.array(loo_p10)
log(f"\n  (b) Leave-one-out 10th percentile (n=23 each, {n} omissions):")
log(f"      range=[{loo_p10.min():.4f}, {loo_p10.max():.4f}]  "
    f"(full-sample value {p10_np:.4f})")
log(f"      values: {sorted(set(round(v, 4) for v in loo_p10))}")

# (c) count of seeds with floor < MAX_VALID_DEPTH_CM (6.50), Clopper-Pearson 95% CI
n_below_gate = int(np.sum(floors < MAX_VALID_DEPTH_CM))
frac_below = n_below_gate / n


def clopper_pearson(k, n, alpha=0.05):
    if k == 0:
        lo = 0.0
    else:
        lo = stats.beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


cp_lo, cp_hi = clopper_pearson(n_below_gate, n)
log(f"\n  (c) Seeds with floor < MAX_VALID_DEPTH_CM ({MAX_VALID_DEPTH_CM}cm, from Source/constants.py):")
log(f"      {n_below_gate}/{n} = {frac_below:.4f}")
log(f"      Clopper-Pearson 95% CI on the fraction: [{cp_lo:.4f}, {cp_hi:.4f}]")

# (d) percentile with 3 interpolation conventions
p10_linear = float(np.percentile(floors, 10, method="linear"))
p10_weibull = float(np.percentile(floors, 10, method="weibull"))
sorted_floors = np.sort(floors)
rank_nearest = int(round(0.10 * (n - 1)))  # nearest-rank via numpy's 'nearest' method for consistency
p10_nearest = float(np.percentile(floors, 10, method="nearest"))

p5_linear = float(np.percentile(floors, 5, method="linear"))
p5_weibull = float(np.percentile(floors, 5, method="weibull"))
p5_nearest = float(np.percentile(floors, 5, method="nearest"))

log(f"\n  (d) Percentile interpolation conventions:")
log(f"      10th pct: numpy-linear={p10_linear:.4f}  weibull(n+1)p={p10_weibull:.4f}  nearest-rank={p10_nearest:.4f}")
log(f"      5th  pct: numpy-linear={p5_linear:.4f}  weibull(n+1)p={p5_weibull:.4f}  nearest-rank={p5_nearest:.4f}")
which_p10 = [name for name, val in [("numpy-linear", p10_linear), ("weibull", p10_weibull), ("nearest-rank", p10_nearest)]
             if abs(val - CHECK_FLOOR["p10"]) < 5e-3]
log(f"      -> convention(s) producing 6.50: {which_p10 if which_p10 else 'NONE (mismatch)'}")

# ---------------------------------------------------------------------------
# Task 5: anomalous_recovery flag
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("TASK 5: anomalous_recovery flag")
log("=" * 78)
log("""
  Definition, from Source/degradation_aware_final_sweep.py (run_one_seed(),
  lines ~155-231): for a given seed, evaluate accuracy at each depth in
  EVAL_DEPTHS_CM for the seed's trained CNN and for 200 label-shuffled
  control CNNs; compute a permutation p-value per depth as
  (#controls >= real accuracy + 1) / 201; apply Benjamini-Hochberg FDR
  (alpha=0.05) across all of that seed's per-depth p-values to get a
  per-depth significance flag; call the first depth (in increasing order)
  that is NOT significant that seed's naive floor; then anomalous_recovery
  is True if any depth STRICTLY DEEPER than that naive floor is still
  flagged significant (i.e. the model appears to regain above-chance,
  FDR-significant accuracy at a depth beyond where it first became
  indistinguishable from chance).
""")
n_anom = sum(anomalous_flags)
log(f"  Seeds with anomalous_recovery=True: {n_anom}/{len(anomalous_flags)}")
log(f"  CHECK 10/24 -> {'MATCH' if n_anom == 10 else 'MISMATCH'}")
anom_seeds = [s for s, f in zip(seeds, anomalous_flags) if f]
log(f"  Seeds: {anom_seeds}")

# ---------------------------------------------------------------------------
# Task 6: per-depth per-seed BH-FDR significance
# ---------------------------------------------------------------------------
log("\n" + "=" * 78)
log("TASK 6: per-depth per-seed significance table")
log("=" * 78)
log("  NOT SAVED. See Task 1 note above -- degradation_aware_final_sweep.py")
log("  discards the per-depth p-values/flags after deriving the three summary")
log("  columns per seed, and this lane is not permitted to rerun training to")
log("  regenerate them. No per-depth table can be produced from existing files.")

# ---------------------------------------------------------------------------
# Save machine-readable outputs
# ---------------------------------------------------------------------------
OUT["single_split_paired_ttests"] = single_results
OUT["nadeau_bengio"] = nb_results
OUT["floor_stats"] = {
    "n": n, "mean": float(mean_f), "sd_ddof1": float(sd_f), "median": float(median_f),
    "p10_numpy_linear": float(p10_np), "p5_numpy_linear": float(p5_np),
    "bootstrap_ci_p5": [float(ci_p5[0]), float(ci_p5[1])],
    "bootstrap_ci_p10": [float(ci_p10[0]), float(ci_p10[1])],
    "loo_p10_range": [float(loo_p10.min()), float(loo_p10.max())],
    "n_below_gate": n_below_gate, "frac_below_gate": float(frac_below),
    "clopper_pearson_95ci": [float(cp_lo), float(cp_hi)],
    "gate_threshold_used_cm": MAX_VALID_DEPTH_CM,
    "p10_conventions": {"numpy_linear": p10_linear, "weibull": p10_weibull, "nearest_rank": p10_nearest},
    "p5_conventions": {"numpy_linear": p5_linear, "weibull": p5_weibull, "nearest_rank": p5_nearest},
}
OUT["anomalous_recovery"] = {"n_true": n_anom, "n_total": len(anomalous_flags), "seeds_true": anom_seeds}

with open(os.path.join(LANE_DIR, "results.json"), "w") as f:
    json.dump(OUT, f, indent=2)

with open(os.path.join(LANE_DIR, "raw_output.txt"), "w") as f:
    f.write("\n".join(LINES) + "\n")

log("\nSaved audit_fixes/lane2/results.json and audit_fixes/lane2/raw_output.txt")

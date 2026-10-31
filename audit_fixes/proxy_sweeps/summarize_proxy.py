# Summary of proxy_floors.csv: per condition the per-seed floors, mean, SD (n - 1),
# median, number of censored seeds and number of seeds at 0.00 cm; and the two
# tests declared in PREREG.md for clean-only against mixed-depth training
# (7_clean_only against 0_baseline): primary the Wilcoxon rank-sum
# (Mann-Whitney U) test, two-sided, unpaired; secondary the paired signed-rank
# test over seeds. All floors are proxy floors, not certified.
#
#   python audit_fixes/proxy_sweeps/summarize_proxy.py [results folder or a floors CSV] [--out summary.csv]
# A folder is merged from proxy_floors_<seed>.csv. After the tests it prints the consistency checks
# declared in PREREG.md (reported, never used to adjust anything).

import os
import sys
import csv
import argparse
import glob
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "cloud", "proxy_sweeps", "results")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ARM_A, ARM_B = "7_clean_only", "0_baseline"


def load(path):
    """path: a proxy_floors.csv file, or the results folder (merges proxy_floors_<seed>.csv and proxy_floors.csv)."""
    files = [path]
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "proxy_floors_*.csv")) + glob.glob(os.path.join(path, "proxy_floors.csv")))
    by = {}
    for p in files:
        for r in csv.DictReader(open(p)):
            by.setdefault(r["condition"], {})[int(r["seed"])] = (float(r["floor_cm"]), r["censored"] == "True")
    return by


def consistency(by, repo):
    """Consistency checks, declared before any result; reported, never used to adjust anything."""
    def ref(path, col, cond_col=None, cond=None):
        out = {}
        for r in csv.DictReader(open(os.path.join(repo, path))):
            if cond_col and r[cond_col] != cond:
                continue
            out[int(r["seed"])] = float(r[col])
        return out
    checks = [
        ("(a) 0_baseline against the robustness battery, condition 1_baseline", "0_baseline",
         {**ref("audit_fixes/cloud/battery/results/floors_101-104.csv", "floor_cm", "condition", "1_baseline"),
          **ref("audit_fixes/cloud/battery/results/floors_105-108.csv", "floor_cm", "condition", "1_baseline")}),
        ("(b) 0_baseline against the certified sweep, naive_floor_cm", "0_baseline",
         ref("Source/logs/degradation_aware_final_results.csv", "naive_floor_cm")),
        ("(c) 7_clean_only against the clean-trained sweep, agg_floor_cm", "7_clean_only",
         ref("audit_fixes/lane5/clean_trained_sweep/clean_sweep/logs/lane5_clean_per_class_results.csv", "agg_floor_cm")),
    ]
    print("\nConsistency checks (declared before any result; reported, never used to adjust anything).")
    print("Control draws and platform differ from the reference runs, so identical floors are not expected.")
    for title, cond, refd in checks:
        if cond not in by:
            print(f"{title}: no {cond} rows yet")
            continue
        seeds = [s for s in sorted(by[cond]) if s in refd and 101 <= s <= 108]
        if not seeds:
            print(f"{title}: no common seeds")
            continue
        x = np.array([by[cond][s][0] for s in seeds])
        y = np.array([refd[s] for s in seeds])
        d = x - y
        corr = float(np.corrcoef(x, y)[0, 1]) if len(seeds) > 2 and x.std() > 0 and y.std() > 0 else float("nan")
        print(f"{title}: seeds {seeds[0]}-{seeds[-1]} (n={len(seeds)})")
        print("  per-seed difference (this run - reference): " + " ".join(f"{s}:{v:+.2f}" for s, v in zip(seeds, d)))
        print(f"  mean difference {d.mean():+.2f}, SD of differences {d.std(ddof=1) if len(d) > 1 else float('nan'):.2f}, correlation {corr:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--reference-root", default=REPO, help="folder holding the reference result files of the consistency checks (default: this repository)")
    a = ap.parse_args()
    by = load(a.path)
    rows = []
    print(f"{'condition':<18} {'n':>2} {'mean':>6} {'SD':>6} {'median':>6} {'censored':>8} {'at 0.00':>7}  floors by seed")
    for cid, d in by.items():
        seeds = sorted(d)
        f = np.array([d[s][0] for s in seeds])
        cens = sum(d[s][1] for s in seeds)
        z = int((f == 0).sum())
        sd = float(f.std(ddof=1)) if len(f) > 1 else float("nan")
        print(f"{cid:<18} {len(f):>2} {f.mean():6.2f} {sd:6.2f} {np.median(f):6.2f} {cens:>8} {z:>7}  " + " ".join(f"{x:.2f}" for x in f))
        rows.append({"condition": cid, "n": len(f), "mean": f.mean(), "sd": sd, "median": float(np.median(f)),
                     "censored": cens, "seeds_at_0": z, "floors": " ".join(f"{s}:{d[s][0]:.2f}" for s in seeds)})
    if ARM_A in by and ARM_B in by:
        a_f = np.array([by[ARM_A][s][0] for s in sorted(by[ARM_A])])
        b_f = np.array([by[ARM_B][s][0] for s in sorted(by[ARM_B])])
        u = mannwhitneyu(a_f, b_f, alternative="two-sided")
        print(f"\nPrimary: Wilcoxon rank-sum (Mann-Whitney U), two-sided, unpaired, {ARM_A} (n={len(a_f)}) against {ARM_B} (n={len(b_f)}): U = {u.statistic:.1f}, p = {u.pvalue:.3f}")
        common = sorted(set(by[ARM_A]) & set(by[ARM_B]))
        pa = np.array([by[ARM_A][s][0] for s in common])
        pb = np.array([by[ARM_B][s][0] for s in common])
        if len(common) >= 2 and np.any(pa != pb):
            w = wilcoxon(pa, pb, alternative="two-sided")
            print(f"Secondary: paired signed-rank over {len(common)} seeds: W = {w.statistic:.1f}, p = {w.pvalue:.3f}")
        else:
            print("Secondary: paired signed-rank not computable (fewer than 2 seeds or all differences zero)")
    consistency(by, os.path.abspath(os.path.expanduser(a.reference_root)))
    if a.out:
        with open(a.out, "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()

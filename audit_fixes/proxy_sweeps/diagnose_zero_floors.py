# Diagnostic for floors of 0.00 cm (floors are not changed). For every (seed, condition) whose floor is 0.00,
# lists real accuracy, raw p and BH-adjusted p at 0 cm and at the next three depths, and classifies the cause
# with a rule fixed before the table was read:
#   (i)  real failure: real accuracy at 0 cm is at most 0.40 (chance is 0.333; 300 windows, SD 0.027);
#   (ii) resolution artifact: real accuracy at 0 cm is above 0.40 and the raw p at 0 cm is at most 2/201
#        (at most one control matches it), yet the depth is not significant after BH, because BH needs at least three
#        depths at the smallest p (1/201) to certify anything with 23 tests;
#   (iii) other: real accuracy above 0.40 and raw p above 2/201.
#   python audit_fixes/proxy_sweeps/diagnose_zero_floors.py [results folder] [--out diagnostic_zero_floors.csv]

import os
import csv
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "cloud", "proxy_sweeps", "results")


def classify(acc0, p0):
    if acc0 <= 0.40:
        return "(i) real failure: accuracy near chance"
    if p0 <= 2 / 201:
        return "(ii) resolution artifact: accuracy above chance, smallest-p resolution"
    return "(iii) other: accuracy above chance, raw p above 2/201"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for fp in sorted(glob.glob(os.path.join(a.path, "proxy_floors_*.csv"))):
        zero = [(int(r["seed"]), r["condition"]) for r in csv.DictReader(open(fp)) if float(r["floor_cm"]) == 0.0]
        for seed, cond in zero:
            d = [r for r in csv.DictReader(open(os.path.join(a.path, f"proxy_depth_detail_{seed}.csv"))) if r["condition"] == cond][:4]
            row = {"seed": seed, "condition": cond}
            for i, r in enumerate(d):
                row[f"depth{i}_cm"] = round(float(r["depth_cm"]), 2)
                row[f"depth{i}_real_acc"] = round(float(r["real_acc"]), 3)
                row[f"depth{i}_raw_p"] = round(float(r["raw_p_value"]), 4)
                row[f"depth{i}_adj_p"] = round(float(r["bh_adjusted_p_value"]), 4)
            row["cause"] = classify(float(d[0]["real_acc"]), float(d[0]["raw_p_value"]))
            rows.append(row)
    out = a.out or os.path.join(a.path, "diagnostic_zero_floors.csv")
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r["seed"], r["condition"], f"acc0={r['depth0_real_acc']} p0={r['depth0_raw_p']} adj0={r['depth0_adj_p']}", r["cause"])


if __name__ == "__main__":
    main()

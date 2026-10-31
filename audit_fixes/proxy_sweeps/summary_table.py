# Per-condition summary table of the proxy sweeps: n, mean, SD (n - 1), median, min, max, number of
# censored seeds (floor = 15.0 cm with every depth significant, a lower bound), seeds at 0.00 cm.
# A mean is marked as a lower bound wherever at least one seed is censored.
#   python audit_fixes/proxy_sweeps/summary_table.py [results folder] [--out summary_by_condition.csv]

import os
import csv
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "cloud", "proxy_sweeps", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=DEFAULT)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    by, label = {}, {}
    for p in sorted(glob.glob(os.path.join(a.path, "proxy_floors_*.csv"))):
        for r in csv.DictReader(open(p)):
            by.setdefault(r["condition"], {})[int(r["seed"])] = (float(r["floor_cm"]), r["censored"] == "True")
            label[r["condition"]] = r["condition_label"]
    rows = []
    for cid in sorted(by, key=lambda c: (int(c.split("_")[0]), c)):
        f = np.array([by[cid][s][0] for s in sorted(by[cid])])
        cens = sum(by[cid][s][1] for s in by[cid])
        rows.append({"condition": cid, "condition_label": label[cid], "n": len(f), "mean": round(float(f.mean()), 2),
                     "sd": round(float(f.std(ddof=1)), 2), "median": round(float(np.median(f)), 2), "min": round(float(f.min()), 2),
                     "max": round(float(f.max()), 2), "censored": cens, "seeds_at_0": int((f == 0).sum()),
                     "mean_is_lower_bound": cens > 0})
    out = a.out or os.path.join(a.path, "summary_by_condition.csv")
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['condition']:<16} mean {r['mean']:6.2f}{'*' if r['mean_is_lower_bound'] else ' '} SD {r['sd']:5.2f} median {r['median']:6.2f} "
              f"min {r['min']:5.2f} max {r['max']:5.2f} censored {r['censored']} at 0.00: {r['seeds_at_0']}")
    print("* lower bound (at least one censored seed)")


if __name__ == "__main__":
    main()

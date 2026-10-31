# Pool the 24 clean-trained confusion matrices (seeds 101-124) by depth.
# Input : audit_fixes/lane5/clean_trained_sweep/clean_sweep/confusion_matrices/seed*_depth*_real_confmat.npy
#         (3x3, rows = true class, columns = predicted class, class order alphabetical:
#          cirrhosis, healthy, significant_fibrosis; see lane5_harness_clean.py:58 and :78-83)
# Output: audit_fixes/lane5/clean_trained_sweep/clean_sweep/pooled_confusion_summary.csv
# Pickle-free: np.load(..., allow_pickle=False).
import csv, glob, re, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "audit_fixes/lane5/clean_trained_sweep/clean_sweep"
CLASSES = ["cirrhosis", "healthy", "significant_fibrosis"]
C, H, F = (CLASSES.index(n) for n in CLASSES)

by = collections.defaultdict(dict)
for f in glob.glob(str(D / "confusion_matrices" / "seed*_depth*_real_confmat.npy")):
    m = re.search(r"seed(\d+)_depth([\d.]+)_real_confmat", f)
    by[float(m[2])][int(m[1])] = np.load(f, allow_pickle=False)

# sanity case: orientation (rows true, columns predicted) and class order
m0 = sum(by[0.0].values())
print("pooled 0.00 cm, rows true / columns predicted, order", CLASSES)
print(m0)
assert m0[C, C] > m0[C, H] and m0[H, H] > m0[H, C], "orientation check failed"
print("sanity: diagonal dominates at 0.00 cm for cirrhosis and healthy rows")

rows = []
for d in sorted(by):
    nets = by[d]
    assert len(nets) == 24, (d, len(nets))
    s = sum(nets.values()).astype(float)
    r = s[H] / s[H].sum() * 100
    n_over = sum(int(m[H, C] / m[H].sum() > 0.5) for m in nets.values())
    rows.append([f"{d:.2f}", len(nets), f"{r[H]:.1f}", f"{r[C]:.1f}", f"{r[F]:.1f}", n_over])

out = D / "pooled_confusion_summary.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["depth_cm", "n_networks", "healthy_to_healthy_pct", "healthy_to_cirrhosis_pct",
                "healthy_to_fibrosis_pct", "n_networks_healthy_to_cirrhosis_over_50pct"])
    w.writerows(rows)
print("wrote", out.relative_to(ROOT), len(rows), "depths")

# One-off recompute of the mean top-minus-second logit gap by depth (not committed).
# Reads outputs/stage4/data/stage4_calibration_data_corrected.csv (seed-42 run, 27.05 dB).
import csv
from collections import defaultdict
import numpy as np

CSV = "outputs/stage4/data/stage4_calibration_data_corrected.csv"
rows = list(csv.DictReader(open(CSV)))
lc = [k for k in rows[0] if k.startswith("logit_")]
by = defaultdict(list)
for r in rows:
    l = sorted(float(r[c]) for c in lc)
    by[float(r["depth_cm"])].append(l[-1] - l[-2])
gap = {d: float(np.mean(v)) for d, v in sorted(by.items())}
for d in (0.0, 1.36, 2.73):
    k = min(gap, key=lambda x: abs(x - d))
    print(f"gap at {k:.2f} cm: {gap[k]:.3f}")
tail = [v for d, v in gap.items() if d >= 8.0]
print(f"gap range for depths >= 8 cm: {min(tail):.3f} to {max(tail):.3f} (n depths = {len(tail)})")

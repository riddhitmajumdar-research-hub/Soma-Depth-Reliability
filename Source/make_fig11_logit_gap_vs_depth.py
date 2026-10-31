# Figure 11: mean gap between the top two class logits vs depth, single trained model (seed 42, 27.05 dB).
# Same computation as Source/logit_gap_recompute.py; adds the plot. Reads only
#   outputs/stage4/data/stage4_calibration_data_corrected.csv   (1380 windows: 23 depths x 60)
# Nothing is typed by hand: the near-zero floor line is the minimum of the mean gaps at depths >= 8 cm.
# The old figure also drew a dashed line "edge of calibration-fit region (5.45 cm)"; no file defines that number, so it is omitted here.
import csv, sys, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path as _P; REPO = str(_P(__file__).resolve().parents[1])
CSV = REPO + "/outputs/stage4/data/stage4_calibration_data_corrected.csv"
OUT = sys.argv[1] if len(sys.argv) > 1 else REPO + "/outputs/figures_final/fig11_logit_gap_vs_depth.png"
rows = list(csv.DictReader(open(CSV)))
lc = [k for k in rows[0] if k.startswith("logit_")]
by = defaultdict(list)
for r in rows:
    l = sorted(float(r[c]) for c in lc); by[float(r["depth_cm"])].append(l[-1] - l[-2])
depths = sorted(by); gap = np.array([np.mean(by[d]) for d in depths])
print(f"n windows {len(rows)}, depths {len(depths)}, windows per depth {sorted({len(v) for v in by.values()})}")
for d in (0.0, 1.36, 2.73):
    k = min(depths, key=lambda x: abs(x - d)); print(f"gap at {k:.2f} cm: {np.mean(by[k]):.3f}")
tail = gap[np.array(depths) >= 8.0]
floor = float(tail.min()); print(f"gap range for depths >= 8 cm: {tail.min():.3f} to {tail.max():.3f} (n depths = {len(tail)})")
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(depths, gap, "o-", color="#1F3A5F", ms=3.5, lw=1.8)
ax.axhline(floor, color="#C07A2C", ls=":", lw=1.8, label=f"Near-zero floor (about {floor:.3f})")
ax.set_xlabel("Depth (cm)"); ax.set_ylabel("Mean top-vs-second logit gap"); ax.set_ylim(bottom=0); ax.legend(frameon=False, loc="upper right")
import csv
with open(OUT.replace(".png", "_values.csv"), "w", newline="") as _f:
    _w = csv.writer(_f); _w.writerow(["depth_cm", "mean_top_minus_second_logit_gap", "n_windows"])
    for _d, _g in zip(depths, gap): _w.writerow([_d, _g, len(by[_d])])
fig.tight_layout(); fig.savefig(OUT, dpi=230); print("saved", OUT)

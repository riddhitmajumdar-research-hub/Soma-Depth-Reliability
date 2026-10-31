# Figure 18: logit gap against depth, configuration B (Log 42)
# Reads only outputs/post_audit/calibration_attribution/calib_by_seed_depth.csv; writes fig18_logit_gap_by_depth.png and fig18_logit_gap_by_depth_values.csv to outputs/figures_final/.
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures_final"
BLUE, ORANGE, GREY, BLACK = "#0072B2", "#E69F00", "#7F7F7F", "#000000"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

rows = list(csv.DictReader(open(ROOT / "outputs/post_audit/calibration_attribution/calib_by_seed_depth.csv")))
depths = sorted({round(float(r["depth_cm"]), 4) for r in rows})
mean, lo, hi = [], [], []
for d in depths:
    v = np.array([float(r["mean_logit_gap"]) for r in rows if abs(float(r["depth_cm"]) - d) < 1e-3])
    mean.append(v.mean()); lo.append(v.min()); hi.append(v.max())
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.fill_between(depths, lo, hi, color=ORANGE, alpha=0.25, label="minimum to maximum of 24 networks")
ax.plot(depths, mean, color=ORANGE, marker="o", ms=3, label="mean of 24 networks")
ax.set_xlabel("depth (cm)")
ax.set_ylabel("mean logit gap (top-1 minus top-2 logit)")
ax.set_title("Configuration B: logit gap against depth, 60 epochs (Log 42)", fontsize=9.5)
ax.legend(frameon=False)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig18_logit_gap_by_depth.png", dpi=150, metadata={"Software": None})
with open(OUT / "fig18_logit_gap_by_depth_values.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["depth_cm", "mean_logit_gap", "min_logit_gap", "max_logit_gap"])
    for a, b, c, d in zip(depths, mean, lo, hi):
        w.writerow([a, round(float(b), 4), round(float(c), 4), round(float(d), 4)])
print("saved", OUT / "fig18_logit_gap_by_depth.png")

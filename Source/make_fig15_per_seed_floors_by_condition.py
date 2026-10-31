# Figure 15: per-seed floors in the sensitivity conditions, configuration A (Log 38)
# Reads only audit_fixes/cloud/proxy_sweeps/results/proxy_floors_101.csv to proxy_floors_108.csv; writes fig15_per_seed_floors_by_condition.png and fig15_per_seed_floors_by_condition_values.csv to outputs/figures_final/.
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

RES = ROOT / "audit_fixes/cloud/proxy_sweeps/results"
by = {}
label = {}
for seed in range(101, 109):
    for r in csv.DictReader(open(RES / ("proxy_floors_%d.csv" % seed))):
        by.setdefault(r["condition"], {})[seed] = float(r["floor_cm"])
order = ["0_baseline", "1_epochs_600", "2_jitter_0.5", "2_jitter_1.0", "3_carrier_50", "3_carrier_150", "4_track_5", "4_track_15", "4_track_20",
         "5_window_50", "5_window_150", "6_stiffness_alt", "7_clean_only"]
names = {"0_baseline": "baseline", "1_epochs_600": "600 epochs", "2_jitter_0.5": "timing jitter 0.5 sample", "2_jitter_1.0": "timing jitter 1.0 sample",
         "3_carrier_50": "pulse carrier 50 Hz", "3_carrier_150": "pulse carrier 150 Hz", "4_track_5": "tracking distance 5 mm", "4_track_15": "tracking distance 15 mm",
         "4_track_20": "tracking distance 20 mm", "5_window_50": "window 50 samples", "5_window_150": "window 150 samples",
         "6_stiffness_alt": "stiffness 5.4 / 7.3 / 11.4 kPa", "7_clean_only": "clean-only training"}
fig, ax = plt.subplots(figsize=(7.2, 5.6))
rows = []
for i, key in enumerate(order):
    y = len(order) - 1 - i
    v = np.array([by[key][s] for s in sorted(by[key])])
    ax.scatter(v, [y] * len(v), s=14, color=GREY, zorder=2)
    ax.scatter([v.mean()], [y], marker="D", s=34, color=BLUE, zorder=3)
    rows.append([names[key]] + [round(x, 3) for x in v] + [round(float(v.mean()), 2)])
ax.set_yticks(range(len(order)))
ax.set_yticklabels([names[k] for k in order][::-1])
ax.set_xlabel("floor (cm); 15.0 cm is the censoring limit")
ax.set_title("Configuration A: floors of eight retrained networks per condition (Log 38)", fontsize=9.5)
ax.scatter([], [], s=14, color=GREY, label="one network")
ax.scatter([], [], marker="D", s=34, color=BLUE, label="mean of 8 (lower bound where censored)")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig15_per_seed_floors_by_condition.png", dpi=150, metadata={"Software": None})
with open(OUT / "fig15_per_seed_floors_by_condition_values.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["condition"] + ["floor_seed%d_cm" % s for s in range(101, 109)] + ["mean_cm"]); w.writerows(rows)
print("saved", OUT / "fig15_per_seed_floors_by_condition.png")

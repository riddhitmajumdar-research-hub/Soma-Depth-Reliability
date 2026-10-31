# Figure 16: median floor against the cirrhosis viscosity, configuration A (Log 39)
# Reads only outputs/post_audit/simulator_audit/eta3_summary.csv; writes fig16_median_floor_vs_cirrhosis_viscosity.png and fig16_median_floor_vs_cirrhosis_viscosity_values.csv to outputs/figures_final/.
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

rows = list(csv.DictReader(open(ROOT / "outputs/post_audit/simulator_audit/eta3_summary.csv")))
eta = np.array([float(r["eta3_pa_s"]) for r in rows])
med = np.array([float(r["floor_median_cm"]) for r in rows])
lo = np.array([float(r["floor_min_cm"]) for r in rows])
hi = np.array([float(r["floor_max_cm"]) for r in rows])
fig, ax = plt.subplots(figsize=(6.4, 4.0))
sweep = eta < 1.5
ax.fill_between(eta[sweep], lo[sweep], hi[sweep], color=BLUE, alpha=0.18, label="minimum to maximum of 12 networks")
ax.plot(eta[sweep], med[sweep], color=BLUE, marker="o", ms=4, label="median of 12 networks")
ax.vlines(eta[~sweep], lo[~sweep], hi[~sweep], color=BLUE, alpha=0.4, lw=3)
ax.scatter(eta[~sweep], med[~sweep], color=BLUE, s=30, zorder=3)
ax.axhline(15.0, color=GREY, lw=0.8, ls="--")
ax.text(0.2, 15.3, "15.0 cm: censoring limit", color=GREY, fontsize=8)
ax.axvline(1.77, color=BLACK, lw=0.8, ls=":")
ax.text(1.74, 2.0, "training value, 1.77 Pa.s", rotation=90, ha="right", fontsize=8)
ax.set_xlabel("cirrhosis viscosity at test (Pa.s)")
ax.set_ylabel("floor (cm)")
ax.set_ylim(-0.5, 17)
ax.set_title("Configuration A: median floor against the cirrhosis viscosity (Log 39)", fontsize=9.5)
ax.legend(loc="lower right", bbox_to_anchor=(0.86, 0.02), frameon=False)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig16_median_floor_vs_cirrhosis_viscosity.png", dpi=150, metadata={"Software": None})
with open(OUT / "fig16_median_floor_vs_cirrhosis_viscosity_values.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cirrhosis_viscosity_pa_s", "median_floor_cm", "min_floor_cm", "max_floor_cm"])
    for a, b, c, d in zip(eta, med, lo, hi):
        w.writerow([a, round(float(b), 3), round(float(c), 3), round(float(d), 3)])
print("saved", OUT / "fig16_median_floor_vs_cirrhosis_viscosity.png")

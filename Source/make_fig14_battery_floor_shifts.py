# Figure 14: median shift of the floor in each battery condition, configuration A (Log 28)
# Reads only audit_fixes/cloud/battery/results/floors_101-104.csv; writes fig14_battery_floor_shifts.png and fig14_battery_floor_shifts_values.csv to outputs/figures_final/.
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

rows = list(csv.DictReader(open(ROOT / "audit_fixes/cloud/battery/results/floors_101-104.csv")))
by = {}
for r in rows:
    by.setdefault(r["condition"], {})[int(r["seed"])] = float(r["floor_cm"])
base = by["1_baseline"]
# conditions of Log 28 with a usable shift; 4a and 4b split between 15 cm and 0 cm and are left out,
# the amplitude condition is the corrected rerun (2b, Log 29)
labels = [
    ("5a_snr20", "SNR 20 dB"), ("5b_snr13.58", "SNR 13.58 dB"), ("5c_snr10", "SNR 10 dB"), ("5d_snr5", "SNR 5 dB"), ("5e_snr0", "SNR 0 dB"),
    ("3a_spreading_p0.5", "spreading p = 0.5"), ("3b_spreading_p1.0", "spreading p = 1.0"),
    ("6a_exact_atten", "exact attenuation"), ("6b_exact_atten_tof", "exact attenuation + time of flight"),
    ("2b_amp_randomization_linear", "amplitude randomisation (corrected)"),
    ("7a_viscosity_healthy_fibrosis_1.4x", "healthy/fibrosis viscosity x1.4"), ("7b_viscosity_healthy_fibrosis_1.8x", "healthy/fibrosis viscosity x1.8"),
    ("7c_viscosity_healthy_fibrosis_2.6x", "healthy/fibrosis viscosity x2.6"),
    ("7d_viscosity_cirrhosis_1.2", "cirrhosis viscosity 1.2 Pa.s"), ("7e_viscosity_cirrhosis_2.18", "cirrhosis viscosity 2.18 Pa.s"),
    ("7f_viscosity_cirrhosis_2.5", "cirrhosis viscosity 2.5 Pa.s"),
]
seeds = sorted(base)
table = []
for key, lab in labels:
    sh = [by[key][s] - base[s] for s in seeds]
    table.append((lab, sh, float(np.median(sh))))
fig, ax = plt.subplots(figsize=(7.2, 5.6))
for i, (lab, sh, med) in enumerate(table):
    y = len(table) - 1 - i
    ax.scatter(sh, [y] * len(sh), s=14, color=GREY, zorder=2)
    ax.scatter([med], [y], marker="D", s=34, color=BLUE, zorder=3)
ax.axvline(0, color=BLACK, lw=0.8)
ax.set_yticks(range(len(table)))
ax.set_yticklabels([t[0] for t in table][::-1])
ax.set_xlabel("shift of the floor against the seed's own baseline (cm)")
ax.set_title("Configuration A: floor shift per condition (Log 28)", fontsize=9.5)
ax.scatter([], [], s=14, color=GREY, label="one seed")
ax.scatter([], [], marker="D", s=34, color=BLUE, label="median of 4 seeds")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig14_battery_floor_shifts.png", dpi=150, metadata={"Software": None})
with open(OUT / "fig14_battery_floor_shifts_values.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["condition"] + ["shift_seed%d_cm" % s for s in seeds] + ["median_shift_cm"])
    for lab, sh, med in table:
        w.writerow([lab] + [round(v, 3) for v in sh] + [round(med, 3)])
print("saved", OUT / "fig14_battery_floor_shifts.png")

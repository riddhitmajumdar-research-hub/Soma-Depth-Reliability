# Progression figure at 100 Hz (27.05 dB), drawn only from CSV files that are in the repository:
#   outputs/stage4/data/stage4_progression_baseline_corrected.csv  (corrected baseline; source in its header)
#   outputs/stage4/data/stage4_progression_clean_cnn.csv           (clean-trained CNN, one saved run)
#   outputs/stage4/data/stage4_progression_degaware_seed42.csv     (degradation-aware CNN, one saved run, accuracy only)
#   Source/logs/degradation_aware_final_results.csv                (24-network mean floor, column naive_floor_cm)
# Floor lines come from the floor_cm column of the first two CSVs and from the 24-network mean. The saved run of the
# degradation-aware CNN has no controls or p-values in the repository, so it has no floor line of its own.
# The plotted points are written to outputs/stage4/data/stage4_progression_plotted_values.csv.
# The first version of this figure is kept as outputs/stage4/figures/stage4_full_progression_corrected_first_version.png;
# its baseline and degradation-aware curves came from result files written before the noise correction (see Log 35).
import csv, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path as _P; ROOT = str(_P(__file__).resolve().parents[1])
D = f"{ROOT}/outputs/stage4/data"

def read(name):
    with open(f"{D}/{name}", newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))

baseline = read("stage4_progression_baseline_corrected.csv")
clean = read("stage4_progression_clean_cnn.csv")
aware = read("stage4_progression_degaware_seed42.csv")
floor_base = float(baseline[0]["floor_cm"])
floor_clean = float(clean[0]["floor_cm"])
with open(f"{ROOT}/Source/logs/degradation_aware_final_results.csv", newline="") as f:
    floors24 = [float(r["naive_floor_cm"]) for r in csv.DictReader(f)]
mean24 = statistics.mean(floors24)

fig, ax = plt.subplots(figsize=(12, 7))
series = ((baseline, "#993C1D", "Baseline (hand-picked features), corrected-noise run"),
          (clean, "#BA7517", "CNN (clean-trained only), one saved run"),
          (aware, "#0F6E56", "CNN (degradation-aware), one saved run, accuracy only (seed 42)"))
plotted = []
for rows, c, lab in series:
    xs = [float(r["depth_cm"]) for r in rows]; ys = [float(r["accuracy"]) for r in rows]
    ax.plot(xs, ys, marker="o", color=c, label=lab, linewidth=2)
    plotted += [(lab, x, y) for x, y in zip(xs, ys)]

ax.axvline(floor_base, color="#993C1D", linestyle=":", alpha=0.6)
ax.axvline(floor_clean, color="#BA7517", linestyle=":", alpha=0.6)
ax.axvline(mean24, color="#0F6E56", linestyle=":", alpha=0.8)
ax.axhline(y=1/3, color="gray", linestyle="--", linewidth=1, label="chance (33.3%)")
ax.text(floor_base, 1.01, f"baseline floor {floor_base:.2f} cm", color="#993C1D", fontsize=9, ha="left")
ax.text(floor_clean, 1.01, f"clean-trained {floor_clean:.2f} cm", color="#BA7517", fontsize=9, ha="left")
ax.text(mean24, 1.01, f"24-network mean {mean24:.2f} cm", color="#0F6E56", fontsize=9, ha="left")
ax.set_xlabel("Depth (cm)")
ax.set_ylabel("Classification accuracy")
ax.set_title(f"Stage 4 -- progression at 100 Hz (27.05 dB): floors {floor_base:.2f} cm, {floor_clean:.2f} cm and {mean24:.2f} cm (24-network mean)", pad=22)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{ROOT}/outputs/stage4/figures/stage4_full_progression_corrected.png", dpi=200, bbox_inches="tight")

with open(f"{D}/stage4_progression_plotted_values.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["curve", "depth_cm", "accuracy"]); w.writerows(plotted)

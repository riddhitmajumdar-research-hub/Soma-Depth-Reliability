# Figure 19: floors at 60, 200 and 600 epochs with the accuracy null band, configuration B (Log 43)
# Reads only outputs/post_audit/training_budget/derived/floor_by_budget_8_seeds.csv and accuracy_null_band.csv; writes fig19_training_budget_floors.png and fig19_training_budget_floors_values.csv to outputs/figures_final/.
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

DER = ROOT / "outputs/post_audit/training_budget/derived"
fl = list(csv.DictReader(open(DER / "floor_by_budget_8_seeds.csv")))
band = list(csv.DictReader(open(DER / "accuracy_null_band.csv")))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
cols = ["floor_60_K10_median_cm", "floor_200_K1_cm", "floor_600_K1_cm"]
for r in fl:
    a1.plot([60, 200, 600], [float(r[c]) for c in cols], color=GREY, lw=0.9, marker="o", ms=3, zorder=2)
means = [np.mean([float(r[c]) for r in fl]) for c in cols]
a1.plot([60, 200, 600], means, color=BLACK, lw=2.2, marker="D", ms=6, zorder=3, label="mean (600 epochs: lower bound)")
a1.set_xscale("log")
a1.set_xticks([60, 200, 600]); a1.set_xticklabels(["60", "200", "600"])
a1.set_xlabel("training epochs"); a1.set_ylabel("floor (cm)")
a1.set_title("Floors of seeds 101 to 108", fontsize=9)
a1.legend(frameon=False, loc="lower right")
GREEN, VERMILLION = "#009E73", "#D55E00"
for ep, col in ((60, GREEN), (200, VERMILLION)):
    b = [r for r in band if int(r["epochs"]) == ep]
    d = np.array([float(r["depth_cm"]) for r in b])
    a2.plot(d, [float(r["median_accuracy"]) for r in b], color=col, lw=1.8, label="accuracy, %d epochs" % ep)
    a2.plot(d, [float(r["null_p97.5_median_across_seeds"]) for r in b], color=col, lw=1.2, ls="--", label="upper end of null band, %d epochs" % ep)
a2.set_xlabel("depth (cm)"); a2.set_ylabel("accuracy (median over 24 seeds)")
a2.set_title("Accuracy against the shuffled-label null band", fontsize=9)
a2.legend(frameon=False, fontsize=7.5)
fig.suptitle("Configuration B: training budget (Log 43)", fontsize=10)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig19_training_budget_floors.png", dpi=150, metadata={"Software": None})
with open(OUT / "fig19_training_budget_floors_values.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["seed", "floor_60_epochs_cm", "floor_200_epochs_cm", "floor_600_epochs_cm"])
    for r in fl:
        w.writerow([r["seed"]] + [round(float(r[c]), 3) for c in cols])
    w.writerow([])
    w.writerow(["epochs", "depth_cm", "median_accuracy", "upper_end_of_null_band"])
    for r in band:
        if int(r["epochs"]) in (60, 200):
            w.writerow([r["epochs"], r["depth_cm"], r["median_accuracy"], r["null_p97.5_median_across_seeds"]])
print("saved", OUT / "fig19_training_budget_floors.png")

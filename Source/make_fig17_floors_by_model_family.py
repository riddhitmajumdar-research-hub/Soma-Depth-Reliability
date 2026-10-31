# Figure 17: floors of three model families, configuration B (Log 41)
# Reads only outputs/post_audit/family_floors/cnn_floor_by_seed.csv, rf_floor_by_seed.csv and resnet_floor_by_seed.csv; writes fig17_floors_by_model_family.png and fig17_floors_by_model_family_values.csv to outputs/figures_final/.
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

FAM = ROOT / "outputs/post_audit/family_floors"
data = []
for name, fn in (("CNN, 60 epochs (24 networks)", "cnn_floor_by_seed.csv"), ("Random forest (24 networks)", "rf_floor_by_seed.csv"),
                 ("ResNet1D (10 networks)", "resnet_floor_by_seed.csv")):
    v = np.array([float(r["floor_new_median"]) for r in csv.DictReader(open(FAM / fn))])
    data.append((name, v))
fig, ax = plt.subplots(figsize=(6.4, 4.0))
rng = np.random.RandomState(0)
for i, (name, v) in enumerate(data):
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=16, color=ORANGE, zorder=2)
    ax.scatter([i], [v.mean()], marker="D", s=40, color=BLACK, zorder=3)
ax.set_xticks(range(len(data)))
ax.set_xticklabels([d[0] for d in data], fontsize=8)
ax.set_ylabel("floor, median of ten draws (cm)")
ax.set_title("Configuration B: floor of each network by model family (Log 41)", fontsize=9.5)
ax.scatter([], [], s=16, color=ORANGE, label="one network")
ax.scatter([], [], marker="D", s=40, color=BLACK, label="mean")
ax.legend(loc="lower center", frameon=False)
fig.tight_layout()
OUT.mkdir(exist_ok=True)
fig.savefig(OUT / "fig17_floors_by_model_family.png", dpi=150, metadata={"Software": None})
with open(OUT / "fig17_floors_by_model_family_values.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["family", "n", "mean_cm", "sd_cm", "median_cm"])
    for name, v in data:
        w.writerow([name, len(v), round(float(v.mean()), 2), round(float(v.std(ddof=1)), 2), round(float(np.median(v)), 2)])
print("saved", OUT / "fig17_floors_by_model_family.png")

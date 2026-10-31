# Figure 8(a), 2 x 2 layout (the layout of make_fig10_pooled_confusion.py is a single row; that script is not modified).
# Same inputs as make_fig10_pooled_confusion.py: 96 saved files
#   audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/confusion_matrices/seed{101..124}_depth{0.00,2.73,4.30,6.82}_real_confmat.npy
# Same class order (saved order cirrhosis, healthy, significant_fibrosis), same display labels, same colormap (YlOrRd). No embedded title.
# Native size is 4.92 in (12.5 cm) wide at 300 dpi, so text sizes are the printed sizes (counts 8 pt, ticks 8 pt).
import sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path as _P; REPO = str(_P(__file__).resolve().parents[1])
SRC = REPO + "/audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/confusion_matrices"
OUT = sys.argv[1] if len(sys.argv) > 1 else REPO + "/outputs/figures_final/fig08a_pooled_confusion_2x2.png"
DEPTHS = ["0.00", "2.73", "4.30", "6.82"]
LABELS = ["Cirrhosis", "Healthy", "Fibrosis"]
SEEDS = range(101, 125)
plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(2, 2, figsize=(4.92, 4.9))
tot = {}
for k, (ax, d) in enumerate(zip(axes.ravel(), DEPTHS)):
    cm = sum(np.load(f"{SRC}/seed{s}_depth{d}_real_confmat.npy") for s in SEEDS)
    assert (cm.sum(1) == 2400).all(); tot[d] = cm
    ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS, fontsize=8); ax.set_yticklabels(LABELS if k % 2 == 0 else [], fontsize=8)
    ax.set_title(f"Depth = {d} cm", fontsize=9)
    if k // 2 == 1: ax.set_xlabel("Predicted", fontsize=8)
    if k % 2 == 0: ax.set_ylabel("True", fontsize=8)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8, color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.tight_layout(h_pad=0.8, w_pad=0.4)
fig.savefig(OUT, dpi=300); print("saved", OUT)
for d in DEPTHS: print(d, tot[d].tolist())
c = tot["6.82"]
print("6.82 cm: cirrhosis->healthy %d/2400 = %.1f%%; fibrosis->healthy %d/2400 = %.1f%%; healthy->cirrhosis %d/2400 = %.2f%%; healthy kept %d/2400 = %.2f%%" % (c[0,1], 100*c[0,1]/2400, c[2,1], 100*c[2,1]/2400, c[1,0], 100*c[1,0]/2400, c[1,1], 100*c[1,1]/2400))
import csv
with open(OUT.replace(".png", "_counts.csv"), "w", newline="") as _f:
    _w = csv.writer(_f); _w.writerow(["depth_cm", "true_class", "pred_cirrhosis", "pred_healthy", "pred_fibrosis"])
    for _d in DEPTHS:
        for _i, _n in enumerate(["cirrhosis", "healthy", "significant_fibrosis"]): _w.writerow([_d, _n] + [int(x) for x in tot[_d][_i]])

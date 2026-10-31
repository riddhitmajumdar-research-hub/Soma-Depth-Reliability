# Fig. 10: multi-depth confusion matrices, certified degradation-aware CNN,
# pooled over the 24 seeds (101-124). Sums saved counts only; nothing is re-evaluated.
# Layout and style follow Source/stage4_depth_diagnostics.py.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/confusion_matrices"
DEPTHS = ["0.00", "2.73", "4.30", "6.82"]
NAMES = ["cirrhosis", "healthy", "significant_fibrosis"]  # saved order (sorted class names)
LABELS = ["Cirrhosis", "Healthy", "Fibrosis"]               # display names, same order
SEEDS = range(101, 125)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, d in zip(axes, DEPTHS):
    cm = sum(np.load(f"{SRC}/seed{s}_depth{d}_real_confmat.npy") for s in SEEDS)
    assert (cm.sum(1) == 2400).all()
    ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_yticklabels(LABELS, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Depth = {d} cm")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    print(d, cm.tolist(), "healthy->cirrhosis %.2f%%" % (100 * cm[1, 0] / 2400), "healthy->healthy %.2f%%" % (100 * cm[1, 1] / 2400))
fig.suptitle("Confusion matrices across depth (100 Hz, 27.05 dB), mixed-depth CNN, pooled over 24 networks (seeds 101-124); 2400 windows per class per depth")
fig.tight_layout()
import os
os.makedirs("outputs/figures_final", exist_ok=True)
fig.savefig("outputs/figures_final/fig08a_pooled_confusion.png", dpi=200, bbox_inches="tight")

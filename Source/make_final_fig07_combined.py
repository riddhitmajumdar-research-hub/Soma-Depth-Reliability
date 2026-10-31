# Combined figure: (a) pooled confusion matrices of 24 networks at 4 depths (2x2) and (b) share of windows predicted healthy vs depth.
# Same saved inputs and calculations as make_fig10_pooled_confusion.py (panel a) and make_final_fig08b_share_healthy.py (panel b); nothing is re-evaluated.
#   panel a: audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/confusion_matrices/seed{101..124}_depth{0.00,2.73,4.30,6.82}_real_confmat.npy
#   panel b: the same folder, all 23 depths; audit_fixes/cloud/battery/results/noise_probe_101-104.csv and noise_probe_105-108.csv
import csv, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CM = f"{ROOT}/audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/confusion_matrices"
NOISE = sorted(glob.glob(f"{ROOT}/audit_fixes/cloud/battery/results/noise_probe_*.csv"))
OUT = f"{ROOT}/outputs/figures_final/fig07_pooled_confusion_and_share_healthy.png"
DEPTHS = ["0.00", "2.73", "4.30", "6.82"]
LABELS = ["Cirrhosis", "Healthy", "Fibrosis"]       # saved order: cirrhosis, healthy, significant_fibrosis
SEEDS = range(101, 125)
GATE_CM = 6.50

fig = plt.figure(figsize=(13, 5.6))
outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)
left = outer[0].subgridspec(2, 2, hspace=0.45, wspace=0.12)
for k, d in enumerate(DEPTHS):
    ax = fig.add_subplot(left[k // 2, k % 2])
    cm = sum(np.load(f"{CM}/seed{s}_depth{d}_real_confmat.npy") for s in SEEDS)
    assert (cm.sum(1) == 2400).all()
    ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS, fontsize=9); ax.set_yticklabels(LABELS if k % 2 == 0 else [], fontsize=9)
    ax.set_title(f"{d} cm", fontsize=11)
    if k // 2 == 1: ax.set_xlabel("Predicted", fontsize=10)
    if k % 2 == 0: ax.set_ylabel("True", fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    print(d, cm.tolist())
fig.text(0.04, 0.97, "(a)", fontsize=14, fontweight="bold", va="top")
fig.text(0.095, 0.975, "Pooled over 24 networks, 2400 windows per class per depth", fontsize=10, va="top")

share = {}
for f in glob.glob(f"{CM}/seed*_depth*_real_confmat.npy"):
    m = re.search(r"seed(\d+)_depth([\d.]+)_real_confmat\.npy", os.path.basename(f))
    cm = np.load(f)
    share.setdefault(float(m.group(2)), {})[int(m.group(1))] = cm[:, 1].sum() / cm.sum()
depths = sorted(share)
assert len(depths) == 23 and all(len(v) == 24 for v in share.values())
M = np.array([[share[d][s] for s in SEEDS] for d in depths])
med, q1, q3 = np.median(M, 1), np.percentile(M, 25, 1), np.percentile(M, 75, 1)
noise = [float(r["fraction"]) for f in NOISE for r in csv.DictReader(open(f)) if r["source"] == "real" and r["class"] == "healthy"]
assert len(noise) == 8
ns = float(np.mean(noise))
print(f"median share healthy: 6.50 cm {med[depths.index(6.5)]:.3f}; 15 cm {med[depths.index(15.0)]:.3f}; pure-noise {ns:.3f} range {min(noise):.2f}-{max(noise):.2f}")

ax = fig.add_subplot(outer[1])
ax.fill_between(depths, q1, q3, color="#1F3A5F", alpha=0.18, linewidth=0, label="Interquartile range")
ax.plot(depths, med, "o-", color="#1F3A5F", ms=3.5, lw=1.8, label="Median over 24 networks")
ax.axhline(1 / 3, color="gray", linestyle=":", lw=1.4, label="Chance (33%)")
ax.axhline(ns, color="#C07A2C", linestyle="--", lw=1.6, label=f"Pure-noise input, 8 networks (about {ns*100:.0f}%)")
ax.axvline(GATE_CM, color="black", linestyle="--", lw=1.3)
ax.text(GATE_CM - 0.12, 0.97, "Safety gate 6.50 cm", rotation=90, va="top", ha="right", fontsize=9)
ax.set_xlabel("Depth (cm)", fontsize=11); ax.set_ylabel("Share of windows predicted healthy", fontsize=11)
ax.set_ylim(0, 1.0); ax.set_xlim(-0.3, 15.3)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
pos = ax.get_position()
fig.text(pos.x0 - 0.075, 0.97, "(b)", fontsize=14, fontweight="bold", va="top")
fig.text(pos.x0 - 0.02, 0.975, "Share of windows predicted healthy vs depth", fontsize=10, va="top")
fig.suptitle("100 Hz, 27.05 dB noise; 24 networks (seeds 101-124)", fontsize=10, y=0.02, style="italic")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", os.path.abspath(OUT))

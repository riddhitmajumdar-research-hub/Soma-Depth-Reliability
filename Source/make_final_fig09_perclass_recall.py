# Final Fig. 9: per-class recall vs depth, median and interquartile band over the 24 networks (seeds 101-124).
# Reads saved per-network files only: audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/logs/perclass_balacc_depth_detail_seed*.csv
# (column recall_real, 23 depths 0-15 cm, one file per network). Nothing is re-evaluated.
# These are the 24 RETRAINED re-evaluated networks, not the original certified weights.
import csv, glob, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = f"{ROOT}/audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/logs"
OUT = f"{ROOT}/outputs/figures_final/fig09_per_class_recall.png"
GATE_CM, LINE = 6.50, 0.45

rec = collections.defaultdict(dict)       # (class, depth) -> {seed: recall}
files = sorted(glob.glob(f"{SRC}/perclass_balacc_depth_detail_seed*.csv"))
assert len(files) == 24
for f in files:
    for r in csv.DictReader(open(f)):
        rec[(r["class"], round(float(r["depth_cm"]), 2))][int(r["seed"])] = float(r["recall_real"])
classes = [("cirrhosis", "Cirrhosis", "#A23B3B"), ("healthy", "Healthy", "#2E7D4F"), ("significant_fibrosis", "Fibrosis", "#1F3A5F")]
depths = sorted({k[1] for k in rec})
assert all(len(v) == 24 for v in rec.values()) and len(depths) == 23

fig, ax = plt.subplots(figsize=(9, 5.2))
for key, name, color in classes:
    M = np.array([[rec[(key, d)][s] for s in range(101, 125)] for d in depths])      # depth x network
    med, q1, q3 = np.median(M, 1), np.percentile(M, 25, 1), np.percentile(M, 75, 1)
    ax.fill_between(depths, q1, q3, color=color, alpha=0.15, linewidth=0)
    ax.plot(depths, med, "o-", color=color, ms=3.5, lw=1.8, label=name)
    i818 = depths.index(8.18)
    print(f"{name}: median recall at 8.18 cm = {med[i818]:.3f}")
    # first depth with recall < 0.45 in each network (FV check)
    cross = [next((d for j, d in enumerate(depths) if M[j, n] < LINE), None) for n in range(24)]
    got = [c for c in cross if c is not None]
    print(f"   networks crossing {LINE}: {len(got)} of 24; median crossing depth {np.median(got):.2f} cm; "
          f"crossings before {GATE_CM} cm strictly: {sum(c < GATE_CM for c in got)}")
ax.axvline(GATE_CM, color="black", linestyle="--", lw=1.3)
ax.text(GATE_CM - 0.12, 0.03, "Safety gate 6.50 cm", rotation=90, ha="right", va="bottom", fontsize=9)
ax.axhline(LINE, color="gray", linestyle=":", lw=1.3)
ax.text(14.9, LINE + 0.015, "Recall 0.45", ha="right", fontsize=8.5, color="gray")
ax.set_xlabel("Depth (cm)")
ax.set_ylabel("Recall (median, interquartile band)")
ax.set_ylim(0, 1.05); ax.set_xlim(-0.3, 15.3)
ax.set_title("Per-class recall vs depth, 24 networks (seeds 101-124), 100 Hz, 27.05 dB noise")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
fig.text(0.5, -0.01, "Median and interquartile range (25th-75th percentile) over 24 retrained, re-evaluated networks at each of 23 depths.", ha="center", fontsize=8, style="italic")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", os.path.abspath(OUT))

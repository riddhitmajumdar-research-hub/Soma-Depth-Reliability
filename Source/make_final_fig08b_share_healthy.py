# Final Fig. 8b: share of windows predicted healthy vs depth, median and interquartile band over 24 networks (seeds 101-124).
# Reads saved files only: the 24 x 23 `seed{S}_depth{D}_real_confmat.npy` matrices in
# audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/confusion_matrices/ (rows = true class, columns = predicted;
# saved class order cirrhosis, healthy, significant_fibrosis) and the pure-noise probe files
# audit_fixes/cloud/battery/results/noise_probe_101-104.csv and noise_probe_105-108.csv (source=real, class=healthy).
import csv, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CM = f"{ROOT}/audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/confusion_matrices"
NOISE = sorted(glob.glob(f"{ROOT}/audit_fixes/cloud/battery/results/noise_probe_*.csv"))
OUT = f"{ROOT}/outputs/figures_final/fig08b_share_predicted_healthy.png"
GATE_CM = 6.50
HEALTHY_COL = 1

share = {}                                   # depth -> list over networks
for f in glob.glob(f"{CM}/seed*_depth*_real_confmat.npy"):
    m = re.search(r"seed(\d+)_depth([\d.]+)_real_confmat\.npy", os.path.basename(f))
    s, d = int(m.group(1)), float(m.group(2))
    cm = np.load(f)
    share.setdefault(d, {})[s] = cm[:, HEALTHY_COL].sum() / cm.sum()
depths = sorted(share)
assert len(depths) == 23 and all(len(v) == 24 for v in share.values()), (len(depths), {d: len(v) for d, v in share.items() if len(v) != 24})
M = np.array([[share[d][s] for s in range(101, 125)] for d in depths])
med, q1, q3 = np.median(M, 1), np.percentile(M, 25, 1), np.percentile(M, 75, 1)

noise = []
for f in NOISE:
    for r in csv.DictReader(open(f)):
        if r["source"] == "real" and r["class"] == "healthy":
            noise.append(float(r["fraction"]))
assert len(noise) == 8
noise_share = float(np.mean(noise))

i65, i15 = depths.index(6.5), depths.index(15.0)
print(f"median share predicted healthy: 6.50 cm = {med[i65]:.3f}; 15 cm = {med[i15]:.3f}; pure-noise probe (8 networks, seeds 101-108) pooled mean = {noise_share:.3f}, range {min(noise):.2f}-{max(noise):.2f}")

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.fill_between(depths, q1, q3, color="#1F3A5F", alpha=0.18, linewidth=0, label="Interquartile range")
ax.plot(depths, med, "o-", color="#1F3A5F", ms=3.5, lw=1.8, label="Median over 24 networks")
ax.axhline(1 / 3, color="gray", linestyle=":", lw=1.4, label="Chance (33%)")
ax.axhline(noise_share, color="#C07A2C", linestyle="--", lw=1.6, label=f"Pure-noise input, 8 networks (about {noise_share*100:.0f}%)")
ax.axvline(GATE_CM, color="black", linestyle="--", lw=1.3)
ax.text(GATE_CM + 0.12, 0.03, "Safety gate 6.50 cm", rotation=90, va="bottom", ha="left", fontsize=9)
ax.set_xlabel("Depth (cm)"); ax.set_ylabel("Share of windows predicted healthy")
ax.set_ylim(0, 1.0); ax.set_xlim(-0.3, 15.3)
ax.set_title("Share of windows predicted healthy vs depth, 24 networks (seeds 101-124), 27.05 dB")
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.33), fontsize=8.5)
ax.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", os.path.abspath(OUT))

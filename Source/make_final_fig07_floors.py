# Final Fig. 7: certified floor distribution over 24 networks (seeds 101-124), mixed-depth vs clean-only training,
# with the 6.50 cm safety gate. Reads saved CSVs only; nothing is re-evaluated.
#   mixed-depth : Source/logs/degradation_aware_final_results.csv            (naive_floor_cm)
#   clean-only  : audit_fixes/lane5/clean_trained_sweep/clean_sweep/logs/lane5_clean_per_class_results.csv (agg_floor_cm)
# Values at the 15 cm grid maximum are censored (the floor is at least 15 cm) and are drawn as a separate hatched bar.
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MIXED = f"{ROOT}/Source/logs/degradation_aware_final_results.csv"
CLEAN = f"{ROOT}/audit_fixes/lane5/clean_trained_sweep/clean_sweep/logs/lane5_clean_per_class_results.csv"
OUT = f"{ROOT}/outputs/figures_final/fig07_floor_distribution.png"
GATE_CM = 6.50       # Source/constants.py MAX_VALID_DEPTH_CM
CENSOR_CM = 15.0     # grid maximum

def col(path, name):
    rows = list(csv.DictReader(open(path)))
    return np.array([int(r["seed"]) for r in rows]), np.array([float(r[name]) for r in rows])

s_mix, mixed = col(MIXED, "naive_floor_cm")
s_cln, clean = col(CLEAN, "agg_floor_cm")
assert len(mixed) == 24 and len(clean) == 24 and sorted(s_mix) == list(range(101, 125)) == sorted(s_cln)

def stats(v):
    return v.mean(), v.std(ddof=1), np.median(v), v.min(), v.max()

edges = np.arange(0, 15.01, 1.5)            # 10 bins covering 0 to 15 (finite floors; 15.0 handled separately)
centers = (edges[:-1] + edges[1:]) / 2
w = 0.6
fig, ax = plt.subplots(figsize=(9, 5.2))
for v, off, color, lab in ((clean, -w / 2, "#C07A2C", "Clean-only training"), (mixed, w / 2, "#1F3A5F", "Mixed-depth training")):
    m, sd, med, lo, hi = stats(v)
    finite = v[v < CENSOR_CM]
    h, _ = np.histogram(finite, bins=edges)
    n_cens = int((v >= CENSOR_CM).sum())
    ax.bar(centers + off, h, width=w, color=color,
           label=f"{lab} (n=24 networks): mean {m:.2f} cm, SD {sd:.2f}, median {med:.2f}, range {lo:.2f}-{hi:.2f}")
    if n_cens:
        ax.bar(15.9 + off, n_cens, width=w, color=color, hatch="//", edgecolor="white")
        ax.text(15.9 + off, n_cens + 0.15, f"{n_cens}", ha="center", fontsize=9)
    print(f"{lab}: n={len(v)} mean={m:.4f} SD(n-1)={sd:.4f} median={med:.4f} min={lo:.4f} max={hi:.4f} below gate={int((v < GATE_CM).sum())} at 15 cm={n_cens}")

ax.axvline(GATE_CM, color="black", linestyle="--", linewidth=1.4)
ax.text(GATE_CM - 0.12, ax.get_ylim()[1] * 0.97, "Safety gate 6.50 cm", rotation=90, va="top", ha="right", fontsize=9)
# network 111 (clean-only): floor of 0.00 cm, a correction artifact of the certification rule, not a real floor
i111 = int(np.where(s_cln == 111)[0][0])
assert clean[i111] == 0.0
ax.annotate("network 111:\nBH artifact,\n0.00 cm floor", xy=(0.75 - w / 2, 0.15), xytext=(1.2, 3.2), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", lw=0.8))
ax.set_xlim(-0.3, 16.9)
ax.set_xticks(list(range(0, 13, 3)) + [15.9])
ax.set_xticklabels([str(t) for t in range(0, 13, 3)] + ["15\n(censored)"])
ax.set_xlabel("Certified floor depth (cm); bin width 1.5 cm")
ax.set_ylabel("Number of networks")
ax.set_title("Certified floor over 24 networks (seeds 101-124), 100 Hz, 27.05 dB noise")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
ax.grid(alpha=0.3, axis="y")
fig.text(0.5, -0.12, "Hatched bars: floor at the 15 cm grid maximum (censored; true floor is at least 15 cm). Means count these as 15 cm. SD uses n-1.",
         ha="center", fontsize=8, style="italic")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", os.path.abspath(OUT))

# Final Fig. 12: certified floor vs cirrhosis viscosity, 8-network proxy (seeds 101-108).
# Reads saved files only: audit_fixes/cloud/battery/results/floors_101-104.csv and floors_105-108.csv
# (conditions 7d = 1.2, 1_baseline = 1.77, 7e = 2.18, 7f = 2.5 Pa.s cirrhosis viscosity; healthy/fibrosis fixed at 0.5 Pa.s).
# Censor rule used here: floor >= 15.0 cm (the grid maximum) is a lower bound. NOTE: the CSV 'censored' column flags only 4 of the
# 5 networks that sit at 15.0 at 1.2 Pa.s (seed 101 has floor 15.0 but censored=False); this script uses the value, not the flag.
import csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = sorted(glob.glob(f"{ROOT}/audit_fixes/cloud/battery/results/floors_*.csv"))
OUT = f"{ROOT}/outputs/figures_final/fig12_floor_vs_cirrhosis_viscosity.png"
CONDS = [("7d_viscosity_cirrhosis_1.2", 1.2), ("1_baseline", 1.77), ("7e_viscosity_cirrhosis_2.18", 2.18), ("7f_viscosity_cirrhosis_2.5", 2.5)]
CENSOR_CM = 15.0

rows = []
for f in SRC:
    rows += list(csv.DictReader(open(f)))
fig, ax = plt.subplots(figsize=(8, 5.2))
means, xs = [], []
rng = np.random.default_rng(0)       # horizontal jitter only (display)
first = {"c": True, "n": True}
for cond, eta in CONDS:
    sel = [(int(r["seed"]), float(r["floor_cm"]), r["censored"]) for r in rows if r["condition"] == cond]
    assert len(sel) == 8 and sorted(s for s, _, _ in sel) == list(range(101, 109))
    v = np.array([f for _, f, _ in sel])
    cens = v >= CENSOR_CM
    jit = rng.uniform(-0.06, 0.06, 8)
    ax.scatter(eta + jit[~cens], v[~cens], color="#1F3A5F", s=36, zorder=3, label="Certified floor" if first["n"] else None); first["n"] = False
    if cens.any():
        ax.scatter(eta + jit[cens], v[cens], color="#C07A2C", s=36, zorder=3, marker="^",
                   label="At the 15 cm grid maximum (lower bound)" if first["c"] else None); first["c"] = False
    means.append(v.mean()); xs.append(eta)
    flagged = sum(c == "True" for _, _, c in sel)
    print(f"eta {eta}: mean {v.mean():.2f} cm (censored values counted as 15), values at >=15.0: {int(cens.sum())} of 8, CSV censored flag: {flagged}")
ax.plot(xs, means, "-", color="black", lw=1.2, zorder=2, label="Mean over 8 networks (lower bound at 1.2 Pa·s)")
ax.set_xticks([e for _, e in CONDS])
ax.set_xlabel("Cirrhosis viscosity (Pa·s); healthy and fibrosis fixed at 0.5 Pa·s")
ax.set_ylabel("Certified floor (cm)")
ax.set_ylim(0, 16.5)
ax.set_title("Floor vs cirrhosis viscosity: 8-network proxy (seeds 101-108), 100 Hz, 27.05 dB")
ax.legend(fontsize=8.5, loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", os.path.abspath(OUT))

"""Aggregates Step 1.2 data: per-eta3 summary table and the sensitivity map figure."""
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import WORKTREE, attenuation_coefficient, stiffness_classes  # noqa: E402

D = os.path.join(WORKTREE, "outputs", "phase1", "data", "eta3")
F = os.path.join(WORKTREE, "outputs", "phase1", "figures")
floors = [r for fn in sorted(glob.glob(os.path.join(D, "floors_seed*.csv"))) for r in csv.DictReader(open(fn))]
detail = [r for fn in sorted(glob.glob(os.path.join(D, "detail_seed*.csv"))) for r in csv.DictReader(open(fn))]
seeds = sorted({int(r["seed"]) for r in floors})
conds = ["eta3_0.2", "eta3_0.3", "eta3_0.4", "eta3_0.5", "eta3_0.6", "eta3_0.7", "eta3_0.8", "eta3_0.9", "eta3_1.0", "eta3_1.77_ref"]
etas = [float(c.split("_")[1]) for c in conds]
fl = np.full((len(seeds), len(conds)), np.nan)
ce = np.zeros((len(seeds), len(conds)), bool)
for r in floors:
    i, j = seeds.index(int(r["seed"])), conds.index(r["condition"])
    fl[i, j] = float(r["floor_cm"]); ce[i, j] = r["censored"] == "True"

depths = sorted({float(r["depth_cm"]) for r in detail})
acc = np.zeros((len(conds), len(depths)))
cnt = np.zeros_like(acc)
for r in detail:
    j, k = conds.index(r["condition"]), depths.index(float(r["depth_cm"]))
    acc[j, k] += float(r["real_acc"]); cnt[j, k] += 1
acc /= cnt

a_h = attenuation_coefficient(stiffness_classes["healthy"], 100)
rows = []
for j, (c, e) in enumerate(zip(conds, etas)):
    a_c = attenuation_coefficient(stiffness_classes["cirrhosis"], 100, eta=e)
    f = fl[:, j]
    rows.append(dict(condition=c, eta3_pa_s=e, n_seeds=len(seeds), n_censored_at_15cm=int(ce[:, j].sum()),
                     n_floor_zero=int((f == 0).sum()), floor_median_cm=np.median(f), floor_min_cm=f.min(),
                     floor_max_cm=f.max(), floor_mean_cm=f.mean(),
                     mean_real_acc_at_15cm=acc[j, -1], mean_real_acc_at_0cm=acc[j, 0],
                     alpha_cirrhosis_per_m=a_c, alpha_ratio_healthy_over_cirrhosis=a_h / a_c))
with open(os.path.join(D, "eta3_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
with open(os.path.join(D, "eta3_floors_by_seed.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["seed"] + conds + [c + "_censored" for c in conds])
    for i, s in enumerate(seeds):
        w.writerow([s] + list(fl[i]) + list(ce[i]))

fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1.05, 1.2, 1.25]))
im = ax[0].imshow(fl, aspect="auto", cmap="viridis", vmin=0, vmax=15)
ax[0].set_xticks(range(len(conds))); ax[0].set_xticklabels([f"{e:g}" for e in etas], fontsize=8)
ax[0].set_yticks(range(len(seeds))); ax[0].set_yticklabels(seeds, fontsize=8)
for i in range(len(seeds)):
    for j in range(len(conds)):
        ax[0].text(j, i, ("≥15" if ce[i, j] else f"{fl[i, j]:.1f}"), ha="center", va="center", fontsize=6.5,
                   color="white" if fl[i, j] < 9 else "black")
ax[0].set_xlabel("cirrhosis η (Pa·s); last column = 1.77 reference"); ax[0].set_ylabel("seed")
ax[0].set_title("Certified floor (cm); ≥15 = every depth significant")
plt.colorbar(im, ax=ax[0], fraction=0.046)

xs = np.arange(len(conds))
for i in range(len(seeds)):
    ax[1].scatter(xs + (i - len(seeds) / 2) * 0.03, fl[i], s=14, c=np.where(ce[i], "#C0392B", "#34495E"), alpha=0.8, zorder=3)
ax[1].plot(xs, np.median(fl, axis=0), color="#0F6E56", lw=2, label="median over seeds", zorder=2)
ax[1].fill_between(xs, np.percentile(fl, 25, axis=0), np.percentile(fl, 75, axis=0), color="#0F6E56", alpha=0.18, label="IQR")
ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{e:g}" for e in etas], fontsize=8)
ax[1].set_xlabel("cirrhosis η (Pa·s)"); ax[1].set_ylabel("floor (cm)")
ax[1].set_title(f"{len(seeds)} seeds; red = censored at the 15 cm grid maximum")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

im2 = ax[2].imshow(acc, aspect="auto", cmap="magma", vmin=0.25, vmax=0.95, origin="upper")
ax[2].set_yticks(range(len(conds))); ax[2].set_yticklabels([f"{e:g}" for e in etas], fontsize=8)
tk = list(range(0, len(depths), 3))
ax[2].set_xticks(tk); ax[2].set_xticklabels([f"{depths[k]:.1f}" for k in tk], fontsize=8)
ax[2].set_xlabel("depth (cm)"); ax[2].set_ylabel("cirrhosis η (Pa·s)")
ax[2].set_title("Mean real-model accuracy over seeds")
plt.colorbar(im2, ax=ax[2], fraction=0.046)
fig.suptitle("Step 1.2: local sensitivity of the certified floor to cirrhosis viscosity (model trained at η=1.77, evaluated at each η)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(F, "p1_2_eta3_sensitivity_map.png"), dpi=150)
print(open(os.path.join(D, "eta3_summary.csv")).read())

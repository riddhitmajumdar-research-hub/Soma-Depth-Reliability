"""Step 1.2 second sensitivity map: per-class recall and one-vs-rest accuracy at fixed depths,
computed from the saved eta3 detail CSVs (no retraining). The 23-depth grid has no 10.0 cm point;
9.55 and 10.91 cm are shown instead. One-vs-rest accuracy for class k with 100 test windows per
class: TP = recall_k*100, FP = pred_frac_k*300 - TP, TN = 200 - FP, acc = (TP+TN)/300."""
import csv, glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import WORKTREE
D = os.path.join(WORKTREE, "outputs/phase1/data/eta3")
rows = [r for fn in sorted(glob.glob(os.path.join(D, "detail_seed*.csv"))) for r in csv.DictReader(open(fn))]
DEPTHS = [0.0, 6.5, 8.0, 9.545454545454545, 10.909090909090908]
conds = ["eta3_0.2","eta3_0.3","eta3_0.4","eta3_0.5","eta3_0.6","eta3_0.7","eta3_0.8","eta3_0.9","eta3_1.0","eta3_1.77_ref"]
etas = [c.split("_")[1] for c in conds]
names = ["cirrhosis", "healthy", "significant_fibrosis"]
rec_key = ["rec_cirr", "rec_heal", "rec_fibr"]
acc = {}
for r in rows:
    d = float(r["depth_cm"])
    di = [i for i, x in enumerate(DEPTHS) if abs(x - d) < 1e-6]
    if not di or r["condition"] not in conds:
        continue
    j = conds.index(r["condition"])
    for k in range(3):
        rec = float(r[rec_key[k]]); pf = float(r[f"pred_frac_{k}"])
        tp = rec * 100; fp = pf * 300 - tp
        acc.setdefault((k, "rec", j, di[0]), []).append(rec)
        acc.setdefault((k, "ovr", j, di[0]), []).append((tp + 200 - fp) / 300)
        acc.setdefault((k, "n", j, di[0]), []).append(1)
M = {key: np.mean(v) for key, v in acc.items() if key[1] != "n"}
nseed = len(acc[(0, "n", 0, 0)])
out = []
for k in range(3):
    for j, c in enumerate(conds):
        for di, d in enumerate(DEPTHS):
            out.append(dict(cls=names[k], eta3=etas[j], depth_cm=round(d, 2), n_seeds=nseed,
                            recall_mean=M[(k, "rec", j, di)], ovr_accuracy_mean=M[(k, "ovr", j, di)],
                            recall_sd=float(np.std(acc[(k, "rec", j, di)]))))
with open(os.path.join(D, "eta3_perclass_fixed_depths.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)

fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
for row, (kind, title, lo) in enumerate((("rec", "recall", 0.0), ("ovr", "one-vs-rest accuracy", 0.4))):
    for k in range(3):
        A = np.array([[M[(k, kind, j, di)] for di in range(len(DEPTHS))] for j in range(len(conds))])
        im = ax[row, k].imshow(A, aspect="auto", cmap="viridis", vmin=lo, vmax=1.0)
        for j in range(len(conds)):
            for di in range(len(DEPTHS)):
                ax[row, k].text(di, j, f"{A[j, di]:.2f}", ha="center", va="center", fontsize=7, color="white" if A[j, di] < (lo + 1) / 2 else "black")
        ax[row, k].set_xticks(range(len(DEPTHS))); ax[row, k].set_xticklabels([f"{d:.2f}".rstrip("0").rstrip(".") for d in DEPTHS], fontsize=8)
        ax[row, k].set_yticks(range(len(conds))); ax[row, k].set_yticklabels(etas, fontsize=8)
        ax[row, k].set_title(f"{names[k]}: {title}", fontsize=9)
        if row == 1: ax[row, k].set_xlabel("depth (cm)")
        if k == 0: ax[row, k].set_ylabel("cirrhosis η (Pa·s); 1.77 = reference")
        plt.colorbar(im, ax=ax[row, k], fraction=0.046)
fig.suptitle(f"Step 1.2, second map: per-class recall and one-vs-rest accuracy at fixed depths, mean over {nseed} seeds (no certification statistic)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(WORKTREE, "outputs/phase1/figures/p1_2_eta3_perclass_map.png"), dpi=150)
for k in (0,):
    print("cirrhosis recall by eta3 at depths", DEPTHS)
    for j, c in enumerate(etas):
        print(c, [round(M[(k, "rec", j, di)], 2) for di in range(len(DEPTHS))])
print("n seeds", nseed)

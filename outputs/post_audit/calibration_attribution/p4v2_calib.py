"""Phase 4.1 per PROTOCOL_PHASE4.md. Outputs data4/calib_*.csv."""
import csv, os, sys
import numpy as np, torch
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata, wilcoxon
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); P4 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(P4, "..", "phase3", "code"))
from p3_common import *  # noqa
from p3_grid import GRID
from p3_run import eval_cells
from cnn_classifier import SimpleCNN
G = np.array(GRID); D4 = os.path.join(P4, "data4"); NB = 15
rb = np.random.RandomState(20261008)
def sm(l): e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
def bins(conf, cor):
    b = np.minimum((conf * NB).astype(int), NB - 1); ece = 0.0; gaps = []
    for k in range(NB):
        m = b == k
        if m.any(): d = abs(cor[m].mean() - conf[m].mean()); ece += m.mean() * d; gaps.append(d)
    return float(ece), float(np.mean(gaps))
def auroc(score, cor):
    pos, neg = score[cor == 1], score[cor == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = rankdata(np.concatenate([pos, neg])); return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
def nll(l, y, T): p = sm(l / T); return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())
fl = {int(r["seed"]): float(r["floor_new_median"]) for r in csv.DictReader(open(os.path.join(P3, "analysis_cnn", "cnn_floor_by_seed.csv")))}
rows = []; trows = []
for seed in range(101, 125):
    ds, _, _ = p2.get_dataset("p2_primary_v1", seed)
    m = SimpleCNN(n_classes=3); m.load_state_dict(torch.load(os.path.join(WORKTREE, f"outputs/phase2/checkpoints/cnn/cnn_p2_primary_v1_seed{seed}.pt"))); m.eval()
    lg = lambda X: m(torch.tensor(X, dtype=torch.float32).unsqueeze(1)).detach().numpy()
    # validation windows at 0 cm only: train_depths [0,3,6,9,12], windows ordered depth-outer, class-inner, 25 per (depth, class) -> first 75
    Xv, yv = ds["val"]; Xv0, yv0 = Xv[:75], yv[:75]; assert len(yv0) == 75 and np.bincount(yv0).tolist() == [25, 25, 25]
    lv = lg(Xv0)
    T = float(np.exp(minimize_scalar(lambda t: nll(lv, yv0, np.exp(t)), bounds=(np.log(0.05), np.log(20)), method="bounded").x))
    trows.append([seed, T, nll(lv, yv0, 1.0), nll(lv, yv0, T), fl[seed], float(G[np.argmin(np.abs(G - fl[seed]))])])
    cx, cy = eval_cells(seed)
    for i, (x, y) in enumerate(zip(cx, cy)):
        l = lg(x); p = sm(l); conf = p.max(1); cor = (p.argmax(1) == y).astype(float); cT = sm(l / T).max(1)
        t2 = np.sort(l, 1)[:, -2:]; lgap = t2[:, 1] - t2[:, 0]
        e1, g1 = bins(conf, cor); e2, g2 = bins(cT, cor)
        rows.append([seed, float(G[i]), cor.mean(), conf.mean(), e1, g1, cT.mean(), e2, g2, lgap.mean(), float(np.median(lgap)), auroc(conf, cor), auroc(lgap, cor), int(cor.sum())])
HDR = ["seed", "depth_cm", "accuracy", "mean_max_softmax", "ECE_15bins", "binned_gap_15bins", "mean_max_softmax_T", "ECE_15bins_T", "binned_gap_15bins_T", "mean_logit_gap", "median_logit_gap", "AUROC_max_softmax", "AUROC_logit_gap", "n_correct"]
with open(os.path.join(D4, "calib_by_seed_depth.csv"), "w", newline="") as f: w = csv.writer(f); w.writerow(HDR); w.writerows(rows)
with open(os.path.join(D4, "calib_temperature.csv"), "w", newline="") as f: w = csv.writer(f); w.writerow(["seed", "T", "val0cm_nll_T1", "val0cm_nll_T", "floor_new_median_cm", "floor_grid_depth_used_cm"]); w.writerows(trows)
R = np.array(rows, float); S = np.arange(101, 125); fd = {int(t[0]): t[5] for t in trows}
def cell(seed, d): return R[(R[:, 0] == seed) & (np.isclose(R[:, 1], d))][0]
def boot(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; idx = rb.randint(0, len(v), (10000, len(v))); b = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), len(v)
names = {2: "accuracy", 3: "mean_max_softmax", 4: "ECE", 5: "binned_gap", 6: "mean_max_softmax_T", 7: "ECE_T", 8: "binned_gap_T", 9: "mean_logit_gap", 11: "AUROC_max_softmax", 12: "AUROC_logit_gap"}
out = [["where", "measure", "mean_over_seeds", "ci_lo", "ci_hi", "n_seeds_defined"]]
for d in (0.0, 4.0, 6.5, 8.0, 10.0, "floor"):
    for j, nm in names.items():
        v = [cell(s, d if d != "floor" else fd[s])[j] for s in S]; mu, lo, hi, n = boot(v); out.append([f"{d} cm" if d != "floor" else "each seed's floor depth", nm, round(mu, 4), round(lo, 4), round(hi, 4), n])
with open(os.path.join(D4, "calib_summary_by_depth.csv"), "w", newline="") as f: csv.writer(f).writerows(out)
# confidence at floor depth versus 0 cm
fo = [["measure", "mean_difference_floor_minus_0cm", "ci_lo", "ci_hi", "wilcoxon_p_two_sided", "n_nonzero_differences", "seeds_with_floor_depth_0 (difference is 0 by construction)"]]
for j, nm in ((3, "mean_max_softmax"), (9, "mean_logit_gap"), (2, "accuracy")):
    dv = np.array([cell(s, fd[s])[j] - cell(s, 0.0)[j] for s in S]); mu, lo, hi, n = boot(dv); nz = dv[dv != 0]
    p = wilcoxon(nz).pvalue if len(nz) > 5 else float("nan")
    fo.append([nm, round(mu, 4), round(lo, 4), round(hi, 4), round(float(p), 5), len(nz), " ".join(str(s) for s in S if fd[s] == 0.0)])
with open(os.path.join(D4, "calib_floor_vs_zero.csv"), "w", newline="") as f: csv.writer(f).writerows(fo)
T = np.array([t[1] for t in trows]); print("T: mean %.2f SD %.2f min %.2f max %.2f; T<1 in %d/24; val0cm NLL %.3f -> %.3f" % (T.mean(), T.std(ddof=1), T.min(), T.max(), (T < 1).sum(), np.mean([t[2] for t in trows]), np.mean([t[3] for t in trows])))
for r in out[1:]: print(*r, sep=" | ")
for r in fo: print(*r, sep=" | ")

"""Addendum section B. (ii) recompute attributions for an independent random init (manual_seed 1000+seed) with the code of p4v2_attr.py
(imported functions copied verbatim); (i),(iii) similarity statistics from saved profiles. Writes data4/attr_profiles_init2.csv, attr_randinit_summary.csv."""
import csv, os, sys, numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); P4 = os.path.abspath(os.path.join(HERE, "..")); D4 = os.path.join(P4, "data4")
sys.path.insert(0, os.path.join(P4, "..", "phase3", "code"))
from p3_common import *  # noqa
from p3_grid import GRID
from p3_run import eval_cells
from cnn_classifier import SimpleCNN
DEPTHS = [0.0, 4.0, 6.5, 8.0, 10.0]; IDX = {d: list(GRID).index(d) for d in DEPTHS}; seeds = list(range(101, 125))
def gradcam(m, x, tgt):
    x = x.clone().requires_grad_(True)
    a = m.relu(m.conv2(m.pool(m.relu(m.conv1(x))))); a.retain_grad()
    out = m.fc2(m.relu(m.fc1(m.pool(a).view(len(x), -1)))); out.gather(1, tgt[:, None]).sum().backward()
    cam = F.relu((a.grad.mean(2, keepdim=True) * a).sum(1, keepdim=True)); return F.interpolate(cam, size=100, mode="linear", align_corners=False)[:, 0].detach().numpy()
def ig(m, x, tgt, steps=32):
    tot = torch.zeros_like(x)
    for k in range(1, steps + 1):
        xi = (x * k / steps).clone().requires_grad_(True); m(xi).gather(1, tgt[:, None]).sum().backward(); tot += xi.grad
    return (x * tot / steps).abs().detach().numpy()[:, 0]
w = csv.writer(open(os.path.join(D4, "attr_profiles_init2.csv"), "w", newline="")); w.writerow(["net", "method", "depth_cm", "group", "seed"] + [f"t{i}" for i in range(100)])
def emit(net, meth, d, seed, A, y):
    A = A / (A.sum(1, keepdims=True) + 1e-12); w.writerow([net, meth, d, "all", seed, *np.round(A.mean(0), 6)])
    for k in range(3): w.writerow([net, meth, d, f"class{k}", seed, *np.round(A[y == k].mean(0), 6)])
for seed in seeds:
    torch.manual_seed(1000 + seed); m0 = SimpleCNN(n_classes=3); m0.eval(); cx, cy = eval_cells(seed)
    for d in DEPTHS:
        x = torch.tensor(cx[IDX[d]], dtype=torch.float32).unsqueeze(1); y = cy[IDX[d]]; tgt = m0(x).argmax(1)
        emit("init2", "gradcam", d, seed, gradcam(m0, x, tgt), y); emit("init2", "ig", d, seed, ig(m0, x, tgt), y)
w = None
P = {}
for fn in ("attr_profiles.csv", "attr_profiles_init2.csv"):
    for r in csv.reader(open(os.path.join(D4, fn))):
        if r[0] != "net": P[(r[0], r[1], float(r[2]), r[3], int(r[4]))] = np.array(r[5:], float)
rb = np.random.RandomState(20261008)
def rho(a, b): return np.nan if np.ptp(a) == 0 or np.ptp(b) == 0 else spearmanr(a, b)[0]
def ci(v):
    v = np.array([x for x in v if not np.isnan(x)]); b = [np.median(v[rb.randint(0, len(v), len(v))]) for _ in range(10000)]; return round(float(np.median(v)), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3), len(v)
def pairwise(net, meth, d, g):
    M = [P[(net, meth, d, g, s)] for s in seeds]; v = [rho(M[i], M[j]) for i in range(24) for j in range(i + 1, 24)]; return round(float(np.nanmedian(v)), 3)
agree = {(r["method"], float(r["depth_cm"]), r["group"]): r for r in csv.DictReader(open(os.path.join(D4, "attr_agreement.csv")))}
H = ["method", "depth_cm", "group", "trained_vs_init1_median", "lo", "hi", "n", "trained_vs_init2_median", "lo", "hi", "n", "init2_pairwise_median", "init1_pairwise_median", "trained_pairwise_median", "trained_pairwise_ci", "distinguishable_by_rule"]
out = [H]
for meth in ("gradcam", "ig"):
    for d in DEPTHS:
        for g in ("all", "class0", "class1", "class2"):
            a = ci([rho(P[("trained", meth, d, g, s)], P[("init", meth, d, g, s)]) for s in seeds]); b = ci([rho(P[("trained", meth, d, g, s)], P[("init2", meth, d, g, s)]) for s in seeds])
            p2 = pairwise("init2", meth, d, g); p1 = pairwise("init", meth, d, g); ag = agree[(meth, d, g)] if (meth, d, g) in agree else agree[(meth, d, g)]
            tm = float(ag["median_pairwise_rho"]); tlo, thi = float(ag["ci_lo"]), float(ag["ci_hi"])
            rule = tlo > max(a[2], b[2], p1, p2)
            out.append([meth, d, g, *a, *b, p2, p1, tm, f"{tlo} to {thi}", rule])
csv.writer(open(os.path.join(D4, "attr_randinit_summary.csv"), "w", newline="")).writerows(out)
for o in out: print(*o, sep=" | ")

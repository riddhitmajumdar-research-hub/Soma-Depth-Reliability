"""A0 ablation. See A0_PLAN.md. CNN only, seeds 101-108, accuracy at 0 and 8 cm. No permutation controls.
Datasets are built in memory (nothing is written into outputs/phase2)."""
import csv, os, sys, time, warnings
import numpy as np, torch
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *  # noqa
import degradation_aware_final_sweep as sweep
torch.set_num_threads(4)
rows = []
for cid in ("p3_a0_a_v1", "p3_a0_b_v1", "p3_a0_c_v1", "p3_a0_e_v1"):
    cfg, h, _ = load_cfg(cid)
    for seed in range(101, 109):
        t0 = time.time()
        with stage_ranges(cfg):
            ds = p2.build_dataset(cfg, seed)
        torch.manual_seed(seed); np.random.seed(seed)
        m = sweep.train_cnn(*ds["train"])
        def a(k):
            with torch.no_grad():
                return p2.acc(m(torch.tensor(ds[k][0], dtype=torch.float32).unsqueeze(1)).argmax(1).numpy(), ds[k][1])
        rows.append(dict(variant=cid, seed=seed, train_acc=a("train"), val_acc=a("val"), acc_clean=a("eval_clean"),
                         acc_d0=a("eval_d0"), acc_d8=a("eval_d8"), wall_s=time.time() - t0, config_hash=h, power=p2.power_state()))
        print(cid, seed, {k: round(v, 3) for k, v in rows[-1].items() if k.startswith("acc") or k.endswith("_acc")}, flush=True)
# variant (d) from the Phase 2 records
for r in csv.DictReader(open(os.path.join(WORKTREE, "outputs/phase2/records/seed_records_cnn_p2_primary_v1.csv"))):
    if int(r["seed"]) <= 108:
        rows.append(dict(variant="p2_primary_v1 (d)", seed=int(r["seed"]), train_acc=float(r["train_acc"]), val_acc=float(r["val_acc"]),
                         acc_clean=float(r["acc_eval_clean"]), acc_d0=float(r["acc_eval_d0"]), acc_d8=float(r["acc_eval_d8"]),
                         wall_s=float(r["total_wall_s"]), config_hash=r["config_hash"], power=r["power_label"]))
os.makedirs(os.path.join(P3, "data"), exist_ok=True)
with open(os.path.join(P3, "data", "a0_cnn_variants.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\nvariant                 n  train   val    clean  d0 (SD)        d8 (SD)")
for v in ["p3_a0_a_v1", "p3_a0_b_v1", "p3_a0_c_v1", "p2_primary_v1 (d)", "p3_a0_e_v1"]:
    r = [x for x in rows if x["variant"] == v]
    g = lambda k: np.array([x[k] for x in r])
    print(f"{v:22s} {len(r)} {g('train_acc').mean():.3f}  {g('val_acc').mean():.3f}  {g('acc_clean').mean():.3f}  "
          f"{g('acc_d0').mean():.3f} ({g('acc_d0').std(ddof=1):.3f})  {g('acc_d8').mean():.3f} ({g('acc_d8').std(ddof=1):.3f})")

# ceiling: random forest on noise-free, jitter-free, hardware-off windows at 0 cm, primary stiffness ranges
from sklearn.ensemble import RandomForestClassifier
from p1_common import gen_windows
from hardware_effects import HardwareConfig
cfg, _, _ = load_cfg("p2_primary_v1")
ceil = []
for seed in range(101, 109):
    def make(stream, n):
        rng = p2.stream_rng("p3_a0_ceiling", seed, stream)
        X, y = [], []
        for name in p2.CLASS_NAMES:
            w = gen_windows(cfg["physics"]["class_nominal_kpa"][name], 0.0, 100, n, rng, class_name=name, mu_mode="range",
                            eta_mode="stage", hw=HardwareConfig(), snr_db=float("inf"), jitter_std=0.0)
            X.extend(w); y.extend([p2.LABEL[name]] * len(w))
        return np.array(X), np.array(y)
    Xa, ya = make("train", 500); Xb, yb = make("test", 200)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=seed).fit(Xa, ya)
    ceil.append(p2.acc(rf.predict(Xb), yb))
print(f"\nCeiling RF on noise-free, jitter-free, hardware-off windows at 0 cm (continuous stiffness, 1500 train / 600 test windows), 8 seeds: mean {np.mean(ceil):.3f} (min {min(ceil):.3f}, max {max(ceil):.3f})")
# mass in more than one stage range
rng = np.random.RandomState(p2.stream_rng("p3_a0_mass", 0, "x").randint(0, 2**31 - 1))
R = cfg["physics"]["stage_mu_range_kpa"]
mu = np.concatenate([rng.uniform(lo, hi, 200000) for lo, hi in R.values()])
inr = sum(((mu >= lo) & (mu <= hi)).astype(int) for lo, hi in R.values())
print(f"Primary config: share of stiffness samples lying in more than one stage range (interior points): {np.mean(inr > 1):.4f} (ranges abut at 6.0 and 9.0, a set of measure zero)")
# repo-quoted Zayadeen distributions: mass outside own range, and mu-only Bayes error with equal priors, truncated to [3,15]
from math import erf, sqrt
def cdf(x, m, s): return 0.5 * (1 + erf((x - m) / (s * sqrt(2))))
Z = {"healthy": (5.39, 0.62), "significant_fibrosis": (7.32, 0.41), "cirrhosis": (11.42, 2.8)}
print("Zayadeen means/SDs as quoted in Source/physics_model.py (F0-F1, F2, F4), untruncated normal mass outside own assumed stage range:")
for k, (m, s) in Z.items():
    lo, hi = R[k]; print(f"  {k:22s} mean {m} SD {s}: mass outside [{lo},{hi}] = {1 - (cdf(hi, m, s) - cdf(lo, m, s)):.3f}")
grid = np.linspace(3, 15, 24001); pdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
dens = np.array([pdf(grid, *Z[k]) for k in Z]); dens = dens / (dens.sum(1, keepdims=True) * (grid[1] - grid[0]))
bayes = 1 - (dens.max(0).sum() * (grid[1] - grid[0])) / 3
print(f"  mu-only Bayes accuracy with equal priors on these (truncated to 3-15 kPa) distributions: {1 - bayes:.3f}")

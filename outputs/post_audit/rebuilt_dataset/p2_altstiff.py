"""Retest of the alternative stiffness set with eta_by_stage on (config p2_altstiff_v1). CNN only, seeds 101-108.
Three arms: default class kPa (4.5/7.5/12.0), alternative kPa (5.4/7.3/11.4) with stage eta, alternative kPa with the old
exact-match lookup. Class-exact stiffness, hardware off, certified noise. Streams follow the committed rule
crc32(config_id|arm|seed|stream), so arms use independent random draws.
No checkpoints are saved for this retest (not required for a retest; trained models are not reused)."""
import csv, os, sys, time, warnings
import numpy as np, torch
warnings.simplefilter("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_common import *  # noqa
from p2_common import _gen_set, _gen_eval_cell
import degradation_aware_final_sweep as sweep
from hardware_effects import HardwareConfig
cfg, cfg_hash, _ = load_config("p2_altstiff_v1")
base, _, _ = load_config("p2_primary_v1")
pseudo = {"physics": {**base["physics"], "mu_mode": "fixed"}}
HW = HardwareConfig()
torch.set_num_threads(4)
rows = []
for seed in cfg["seeds"]:
    for arm, a in cfg["arms"].items():
        t0 = time.time(); p0 = power_state()
        em = "stage" if a["eta_by_stage"] else "lookup"
        Xtr, ytr = _gen_set(pseudo, stream_rng("p2_altstiff_v1", seed, "train", arm), [0, 3, 6, 9, 12], 25, HW,
                            class_kpa=a["class_kpa"], mu_mode="fixed", eta_mode=em)
        torch.manual_seed(seed); np.random.seed(seed)
        t = time.time(); m = sweep.train_cnn(Xtr, ytr); tt = time.time() - t
        r = dict(seed=seed, arm=arm, train_acc=0.0, train_wall_s=tt)
        with torch.no_grad():
            r["train_acc"] = acc(m(torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1)).argmax(1).numpy(), ytr)
            for d in (0.0, 6.5, 8.0, 10.0):
                X, y = _gen_eval_cell(pseudo, stream_rng("p2_altstiff_v1", seed, f"eval_d{d}", arm), d, 100, HW,
                                      class_kpa=a["class_kpa"], mu_mode="fixed", eta_mode=em)
                r[f"acc_d{d}"] = acc(m(torch.tensor(X, dtype=torch.float32).unsqueeze(1)).argmax(1).numpy(), y)
        r.update(total_wall_s=time.time() - t0, power_label=power_label(p0, power_state()), config_hash=cfg_hash)
        rows.append(r)
        print(seed, arm, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k.startswith("acc") or k == "train_acc"}, flush=True)
with open(os.path.join(P2, "records", "altstiff_p2_altstiff_v1.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
arms = list(cfg["arms"])
def col(arm, k): return np.array([r[k] for r in rows if r["arm"] == arm])
print("\nMean over 8 seeds")
for arm in arms:
    print(f"  {arm:20s} " + " ".join(f"{k} {col(arm,k).mean():.3f}" for k in ("train_acc", "acc_d0.0", "acc_d6.5", "acc_d8.0", "acc_d10.0")))
for A, B in (("alt_set_stage_eta", "default_set"), ("alt_set_old_lookup", "default_set"), ("alt_set_stage_eta", "alt_set_old_lookup")):
    print(f"  paired {A} - {B}: " + " ".join(f"{k} {(col(A,k)-col(B,k)).mean():+.3f} (SD {(col(A,k)-col(B,k)).std(ddof=1):.3f})" for k in ("acc_d0.0", "acc_d6.5", "acc_d8.0", "acc_d10.0")))

"""Phase 1 certification runner (Steps 1.2 and 1.3).

Usage: p1_run.py --task {eta3,floor13} --seeds 101 102 ... --out DIR [--threads N]

Per seed: np.random.seed / torch.manual_seed(seed); build the baseline training set with
Source's own builder; train the real CNN and 200 shuffled-label controls with Source's
train_cnn (same call order as the cloud battery harness). Then each condition builds a
23-depth x 300-window test set with p1_common.gen_windows and runs the permutation test
and BH-FDR at 0.05. Controls are trained once per training distribution and reused across
conditions, as in the battery.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import (gen_windows, build_training_set, cond_rng, bh_adjusted, CLASS_NAMES,  # noqa: E402
                       stiffness_classes)
import degradation_aware_final_sweep as sweep  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

DEPTHS = [float(d) for d in sweep.EVAL_DEPTHS_CM]
N_SHUF = sweep.N_SHUFFLES
N_PER_CLASS = sweep.N_SAMPLES_PER_CELL // 3
FDR = 0.05

ETA3_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def conds_for(task):
    """Each condition: (id, spec). spec keys: mu_mode, mu_cv (per class dict), eta_mode,
    eta_override (per class dict), train ('base' or 'range') selects which model set."""
    C = []
    if task == "eta3":
        C.append(("eta3_1.77_ref", dict(train="base", eta_mode="lookup")))
        for e in ETA3_GRID:
            C.append((f"eta3_{e:.1f}", dict(train="base", eta_mode="lookup",
                                            eta_override={"cirrhosis": e})))
    elif task == "floor13":
        cv_pc = {"healthy": 0.115, "significant_fibrosis": 0.056, "cirrhosis": 0.245}
        cv_u = {k: 0.15 for k in cv_pc}
        for tr in ("base", "range"):
            if tr == "base":
                C.append(("base", dict(train="base")))
                C.append(("cv4a_lookup", dict(train="base", mu_mode="cv", mu_cv=cv_pc)))
                C.append(("cv4b_lookup", dict(train="base", mu_mode="cv", mu_cv=cv_u)))
                C.append(("cv4a_stage", dict(train="base", mu_mode="cv", mu_cv=cv_pc, eta_mode="stage")))
                C.append(("cv4b_stage", dict(train="base", mu_mode="cv", mu_cv=cv_u, eta_mode="stage")))
            C.append((f"range_lookup_tr{tr}", dict(train=tr, mu_mode="range")))
            C.append((f"range_stage_tr{tr}", dict(train=tr, mu_mode="range", eta_mode="stage")))
    return C


def build_cell(depth_cm, spec, encoder, rng):
    X, y = [], []
    for name in CLASS_NAMES:
        kpa = stiffness_classes[name]
        kw = dict(mu_mode=spec.get("mu_mode", "fixed"), eta_mode=spec.get("eta_mode", "lookup"))
        if kw["mu_mode"] == "cv":
            kw["mu_cv"] = spec["mu_cv"][name]
        eo = spec.get("eta_override", {})
        if name in eo:
            kw["eta_override"] = eo[name]
        w = gen_windows(kpa, depth_cm / 100, 100, N_PER_CLASS, rng, class_name=name, **kw)
        X.extend(w)
        y.extend([name] * len(w))
    yt = encoder.transform(y)
    return torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1), yt


def floor_from(sig):
    for d, s in zip(DEPTHS, sig):
        if not s:
            return d, False
    return DEPTHS[-1], True


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def run_seed(seed, task, out, ckpt_dir):
    """One training pass per training distribution; after each pass the conditions that use it
    are evaluated and written, so a stopped run keeps finished passes. Wall-clock seconds per
    model type are written to timing_seed<seed>.csv."""
    conds = conds_for(task)
    passes = sorted({c[1]["train"] for c in conds}, key=lambda t: (t != "base", t))
    encoder = LabelEncoder()
    encoder.fit(sorted(stiffness_classes.keys()))
    os.makedirs(out, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    for tr in passes:
        if os.path.exists(os.path.join(out, f"floors_seed{seed}_{tr}.csv")):
            print(f"seed={seed} pass={tr} exists, skipping", flush=True)
            continue
        t0 = time.time()
        torch.manual_seed(seed)
        np.random.seed(seed)
        t = time.time()
        if tr == "base":
            X, yn = sweep.build_degradation_aware_dataset()
        else:
            X, yn = build_training_set(np.random, mu_mode="range", eta_mode="stage")
        t_data = time.time() - t
        ye = LabelEncoder().fit(sorted(stiffness_classes.keys())).transform(yn)
        t = time.time()
        real = sweep.train_cnn(X, ye)
        t_real = time.time() - t
        t = time.time()
        ctr = [sweep.train_cnn(X, np.random.permutation(ye)) for _ in range(N_SHUF)]
        t_ctl = time.time() - t
        # DIAGNOSTIC checkpoints, not certified models
        torch.save(real.state_dict(), os.path.join(ckpt_dir, f"DIAGNOSTIC_{task}_{tr}_seed{seed}_real.pt"))
        torch.save([m.state_dict() for m in ctr], os.path.join(ckpt_dir, f"DIAGNOSTIC_{task}_{tr}_seed{seed}_controls200.pt"))
        print(f"seed={seed} pass={tr} data {t_data:.1f}s real {t_real:.1f}s controls(200) {t_ctl:.1f}s", flush=True)

        floors, detail, ctl_store = [], [], {}
        t = time.time()
        for cid, spec in [c for c in conds if c[1]["train"] == tr]:
            rng = cond_rng(seed, cid)
            P, ACC, CT, RECALL, PRED = [], [], [], [], []
            for d in DEPTHS:
                Xt, yt = build_cell(d, spec, encoder, rng)
                with torch.no_grad():
                    pr = real(Xt).argmax(1).numpy()
                    ca = np.array([float(np.mean(m(Xt).argmax(1).numpy() == yt)) for m in ctr])
                acc = float(np.mean(pr == yt))
                p = (np.sum(ca >= acc) + 1) / (N_SHUF + 1)
                P.append(p); ACC.append(acc); CT.append(ca)
                RECALL.append([float(np.mean(pr[yt == k] == k)) for k in range(3)])
                PRED.append([float(np.mean(pr == k)) for k in range(3)])
            sig = sweep.bh_fdr_correction(P, alpha=FDR)
            adj = bh_adjusted(P)
            fl, cens = floor_from(sig)
            lastsig = max([d for d, s in zip(DEPTHS, sig) if s], default=None)
            floors.append(dict(seed=seed, condition=cid, train=tr, floor_cm=fl, censored=cens,
                               last_sig_cm=lastsig, acc_depth0=ACC[0], acc_depth15=ACC[-1], n_sig=int(np.sum(sig))))
            for i, d in enumerate(DEPTHS):
                detail.append(dict(seed=seed, condition=cid, depth_cm=d, real_acc=ACC[i], raw_p=P[i],
                                   bh_adj_p=adj[i], sig=bool(sig[i]),
                                   rec_cirr=RECALL[i][CLASS_NAMES.index("cirrhosis")],
                                   rec_heal=RECALL[i][CLASS_NAMES.index("healthy")],
                                   rec_fibr=RECALL[i][CLASS_NAMES.index("significant_fibrosis")],
                                   pred_frac_0=PRED[i][0], pred_frac_1=PRED[i][1], pred_frac_2=PRED[i][2],
                                   ctl_min=CT[i].min(), ctl_med=float(np.median(CT[i])), ctl_max=CT[i].max()))
            ctl_store[cid] = np.array(CT)
            print(f"seed={seed} {cid}: floor={fl:.2f} censored={cens} acc0={ACC[0]:.3f} acc15={ACC[-1]:.3f}", flush=True)
        t_eval = time.time() - t
        write_csv(os.path.join(out, f"detail_seed{seed}_{tr}.csv"), detail)
        np.savez_compressed(os.path.join(out, f"controls_seed{seed}_{tr}.npz"), depths=np.array(DEPTHS), **ctl_store)
        write_csv(os.path.join(out, f"timing_seed{seed}_{tr}.csv"),
                  [dict(seed=seed, train=tr, data_build_s=t_data, real_model_train_s=t_real,
                        controls200_train_s=t_ctl, eval_s=t_eval, n_conditions=len(floors),
                        total_s=time.time() - t0, torch_threads=torch.get_num_threads())])
        write_csv(os.path.join(out, f"floors_seed{seed}_{tr}.csv"), floors)  # written last = pass complete
        print(f"seed={seed} pass={tr} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    ck = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")
    for s in a.seeds:
        run_seed(s, a.task, a.out, ck)

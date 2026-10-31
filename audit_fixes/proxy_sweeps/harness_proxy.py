# Proxy-sweep harness: eight networks (seeds 101 to 108). For every condition the
# real model and its 200 shuffled-label controls are retrained on that condition's
# training data (controls: shuffled labels of the same data, same number of epochs
# as the real model) and evaluated on that condition's evaluation windows: the same
# 23-depth grid, 300 windows per depth, BH correction per (seed, condition),
# floor = first depth that is not significant, 15.0 cm = fallback cap (censored),
# 27.05 dB sensor noise. Decisions are in PREREG.md.
#
# Reuses, without altering them: Source/degradation_aware_final_sweep.py (training
# depths, evaluation grid, BH rule), Source/cnn_classifier.py (clean dataset,
# network) and the evaluation procedure of
# audit_fixes/cloud/battery/seeds_101-104/harness.py (copied here in compact form
# because importing that module creates folders and a log file inside the battery
# folder). Training-set generation and the changed parameters are in
# wrappers_proxy.py.
#
# Usage (from the repository root):
#   python audit_fixes/proxy_sweeps/harness_proxy.py --list
#   python audit_fixes/proxy_sweeps/harness_proxy.py --seeds 101 --conditions 0_baseline
#   python audit_fixes/proxy_sweeps/harness_proxy.py            # all seeds, all default conditions
# Output (default folder audit_fixes/cloud/proxy_sweeps/results/), one set of files per seed so that
# parallel processes (one per seed, --threads 1) never write the same file:
#   proxy_floors_<seed>.csv, proxy_depth_detail_<seed>.csv, proxy_run_<seed>.log
# A restart skips the (seed, condition) rows that already exist.

import os
import sys
import csv
import time
import zlib
import argparse
import logging
import platform
import datetime
import numpy as np


def _early_threads(argv):
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return None


_THREADS = _early_threads(sys.argv)
if _THREADS is not None:
    os.environ["OMP_NUM_THREADS"] = str(_THREADS)   # set before torch is imported
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Source"))
sys.path.insert(0, HERE)

from degradation_aware_final_sweep import (  # noqa: E402
    TRAIN_DEPTHS_CM, N_SAMPLES_PER_DEPTH_PER_CLASS, bh_fdr_correction,
    EVAL_DEPTHS_CM, FREQ, N_SAMPLES_PER_CELL,
)
from cnn_classifier import build_raw_dataset  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from wrappers_proxy import (  # noqa: E402
    CONDITIONS, BY_ID, TRAIN_WRAPPERS, EVAL_WRAPPERS, classes_for, train_cnn_proxy,
)

SEEDS = [101, 102, 103, 104, 105, 106, 107, 108]
N_SHUFFLES = 200
FDR_ALPHA = 0.05
N_TORCH_THREADS = 4          # default; --threads overrides
DEPTHS = list(EVAL_DEPTHS_CM)
N_PER_CLASS = N_SAMPLES_PER_CELL // 3

FLOOR_FIELDS = ["seed", "condition", "condition_label", "epochs", "floor_cm", "last_significant_cm", "censored",
                "n_shuffles", "seconds_train_real", "seconds_controls", "seconds_eval"]
DETAIL_FIELDS = ["seed", "condition", "depth_cm", "real_acc", "raw_p_value", "bh_adjusted_p_value", "fdr_significant",
                 "control_acc_min", "control_acc_median", "control_acc_max"]


def bh_adjusted_pvalues(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    raw = p[order] * n / np.arange(1, n + 1)
    adj_sorted = np.clip(np.minimum.accumulate(raw[::-1])[::-1], 0, 1)
    out = np.empty(n)
    out[order] = adj_sorted
    return out


def build_training_set(cond, seed):
    """Mixed-depth data generated with the condition's shift (training depths and counts of
    build_degradation_aware_dataset), or the clean-only set of lane5_harness_clean.py."""
    if cond["kind"] == "clean":
        return build_raw_dataset(seed=seed)
    wrapper = TRAIN_WRAPPERS[cond["id"]]
    X, y = [], []
    for depth_cm in TRAIN_DEPTHS_CM:
        for name in classes_for(cond["train_shift"]):
            w = wrapper(name, depth_cm / 100, FREQ, N_SAMPLES_PER_DEPTH_PER_CLASS)
            X.extend(w)
            y.extend([name] * len(w))
    return np.array(X), np.array(y)


def build_eval_cell(cond, depth_cm, encoder, rng):
    wrapper = EVAL_WRAPPERS[cond["id"]]
    X, names = [], []
    for name in sorted(classes_for(cond["eval_shift"])):
        w = wrapper(name, depth_cm / 100, FREQ, N_PER_CLASS, rng=rng)
        X.extend(w)
        names.extend([name] * len(w))
    return torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1), encoder.transform(names)


def acc(model, X, y):
    with torch.no_grad():
        return float(np.mean(model(X).argmax(dim=1).numpy() == y))


def run_condition(seed, cond, n_shuffles, floors_w, floors_f, detail_dir, log):
    """Trains the real model and the controls for this condition, evaluates, checkpoints."""
    wsize = cond["shift"].get("window_size", 100)
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.time()
    X, names = build_training_set(cond, seed)
    encoder = LabelEncoder()
    y = encoder.fit_transform(names)
    real = train_cnn_proxy(X, y, n_epochs=cond["epochs"], window_size=wsize)
    t_real = time.time() - t0
    t1 = time.time()
    controls = [train_cnn_proxy(X, np.random.permutation(y), n_epochs=cond["epochs"], window_size=wsize)
                for _ in range(n_shuffles)]
    t_ctrl = time.time() - t1
    log.info(f"seed={seed} condition={cond['id']} epochs={cond['epochs']}: real model {t_real:.1f}s, {n_shuffles} controls "
             f"{t_ctrl:.1f}s ({t_ctrl / max(n_shuffles, 1):.2f}s per control)")
    t2 = time.time()
    rng = np.random.RandomState((seed * 100003 + zlib.crc32(cond["id"].encode())) % (2 ** 32 - 1))
    real_acc, pvals, cstats = [], [], []
    for depth_cm in DEPTHS:
        Xe, ye = build_eval_cell(cond, depth_cm, encoder, rng)
        a = acc(real, Xe, ye)
        ca = np.array([acc(m, Xe, ye) for m in controls])
        real_acc.append(a)
        pvals.append((np.sum(ca >= a) + 1) / (n_shuffles + 1))
        cstats.append((ca.min(), float(np.median(ca)), ca.max()))
    sig = bh_fdr_correction(pvals, alpha=FDR_ALPHA)
    adj = bh_adjusted_pvalues(pvals)
    floor = next((d for d, s in zip(DEPTHS, sig) if not s), None)
    censored = floor is None
    if censored:
        floor = DEPTHS[-1]
    last_sig = None
    for d, s in zip(DEPTHS, sig):
        if s:
            last_sig = d
    dpath = os.path.join(detail_dir, f"proxy_depth_detail_{seed}.csv")
    dnew = not os.path.exists(dpath)
    with open(dpath, "a", newline="") as df:
        dw = csv.DictWriter(df, fieldnames=DETAIL_FIELDS)
        if dnew:
            dw.writeheader()
        for i, d in enumerate(DEPTHS):
            dw.writerow({"seed": seed, "condition": cond["id"], "depth_cm": d, "real_acc": real_acc[i], "raw_p_value": pvals[i],
                         "bh_adjusted_p_value": adj[i], "fdr_significant": bool(sig[i]),
                         "control_acc_min": cstats[i][0], "control_acc_median": cstats[i][1], "control_acc_max": cstats[i][2]})
    t_eval = time.time() - t2
    floors_w.writerow({"seed": seed, "condition": cond["id"], "condition_label": cond["label"], "epochs": cond["epochs"],
                       "floor_cm": floor, "last_significant_cm": last_sig, "censored": censored, "n_shuffles": n_shuffles,
                       "seconds_train_real": round(t_real, 1), "seconds_controls": round(t_ctrl, 1), "seconds_eval": round(t_eval, 1)})
    floors_f.flush()
    log.info(f"seed={seed} condition={cond['id']} floor={floor:.2f}cm censored={censored} (eval {t_eval:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--conditions", nargs="*", default=None, help="condition ids; default: every condition marked default")
    ap.add_argument("--threads", type=int, default=None, help="torch threads and OMP_NUM_THREADS (default 4)")
    ap.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "audit_fixes", "cloud", "proxy_sweeps", "results"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for c in CONDITIONS:
            req = (f"retrain real model + {N_SHUFFLES} controls ({c['epochs']} epochs) on "
                   + ("clean-only data (lane5_harness_clean.py set), evaluate under the baseline degradation" if c["kind"] == "clean"
                      else "mixed-depth data generated with the changed parameter; evaluate with the changed parameter"))
            print(f"{c['id']} | {c['label']} | group: own training group | {req}"
                  + ("" if c["default_run"] else " | not run by default (same data as the baseline)"))
        return
    threads = a.threads if a.threads is not None else N_TORCH_THREADS
    torch.set_num_threads(threads)
    os.makedirs(a.out_dir, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s")
    log = logging.getLogger("proxy")
    log.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    wanted = [BY_ID[i] for i in a.conditions] if a.conditions else [c for c in CONDITIONS if c["default_run"]]
    for seed in a.seeds:
        fh = logging.FileHandler(os.path.join(a.out_dir, f"proxy_run_{seed}.log"))
        fh.setFormatter(fmt)
        log.addHandler(fh)
        log.info(f"START {datetime.datetime.now().astimezone().isoformat()} seed={seed} platform={platform.platform()} "
                 f"python={platform.python_version()} torch={torch.__version__} numpy={np.__version__} cpus={os.cpu_count()} "
                 f"threads={threads} n_shuffles={a.n_shuffles}")
        floors_path = os.path.join(a.out_dir, f"proxy_floors_{seed}.csv")
        done = set()
        if os.path.exists(floors_path):
            with open(floors_path) as f:
                done = {r["condition"] for r in csv.DictReader(f)}
        new = not os.path.exists(floors_path)
        seed_t0 = time.time()
        with open(floors_path, "a", newline="") as ff:
            floors_w = csv.DictWriter(ff, fieldnames=FLOOR_FIELDS)
            if new:
                floors_w.writeheader()
            for cond in wanted:
                if cond["id"] in done:
                    log.info(f"seed={seed} condition={cond['id']} already complete, skipping")
                    continue
                run_condition(seed, cond, a.n_shuffles, floors_w, ff, a.out_dir, log)
        log.info(f"seed={seed} TOTAL {time.time() - seed_t0:.0f}s")
        log.info(f"END {datetime.datetime.now().astimezone().isoformat()} seed={seed}")
        log.removeHandler(fh)
        fh.close()


if __name__ == "__main__":
    main()

"""Phase 2 shared code: config loading, deterministic dataset construction, hashing,
power labels, checkpoint manifests. Reuses the Phase 1 generator (outputs/phase1/code/p1_common.py)
and Source/ without modifying either. Every random stream is derived with zlib.crc32; no Python hash().
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P2 = os.path.join(WORKTREE, "outputs", "phase2")
sys.path.insert(0, os.path.join(WORKTREE, "outputs", "phase1", "code"))
sys.path.insert(0, os.path.join(WORKTREE, "Source"))

from p1_common import gen_windows  # noqa: E402
from hardware_effects import HardwareConfig  # noqa: E402

CLASS_ORDER_GEN = ["healthy", "significant_fibrosis", "cirrhosis"]  # stiffness_classes iteration order
CLASS_NAMES = sorted(CLASS_ORDER_GEN)                               # label encoding order (alphabetical)
LABEL = {n: i for i, n in enumerate(CLASS_NAMES)}


def load_config(config_id):
    path = os.path.join(P2, "configs", f"dataset_config_{config_id}.json")
    raw = open(path, "rb").read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest(), path


def stream_rng(config_id, seed, stream, arm=None):
    key = f"{config_id}|{arm}|{seed}|{stream}" if arm is not None else f"{config_id}|{seed}|{stream}"
    return np.random.RandomState(zlib.crc32(key.encode()))


def hw_from_cfg(cfg):
    h = cfg["hardware"]
    return HardwareConfig(timing_jitter_ms=float(h["timing_jitter_ms"]), noise_beta=h["noise_beta"],
                          gain_range=tuple(h["gain_range"]) if h["gain_range"] else None)


def _gen_set(cfg, rng, depths_cm, n_per, hw, snr_db=None, order="depth_outer", class_kpa=None,
             mu_mode=None, eta_mode="stage"):
    phys = cfg["physics"]
    mu_mode = mu_mode or phys["mu_mode"]
    kpa = class_kpa or phys["class_nominal_kpa"]
    X, y = [], []
    for d in depths_cm:
        for name in CLASS_ORDER_GEN:
            w = gen_windows(kpa[name], d / 100, phys["frequency_hz"], n_per, rng, class_name=name,
                            mu_mode=mu_mode, eta_mode=eta_mode, hw=hw, snr_db=snr_db,
                            jitter_std=phys["heart_rate_jitter_sd_hz"])
            X.extend(w)
            y.extend([LABEL[name]] * len(w))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


def _gen_eval_cell(cfg, rng, depth_cm, n_per_class, hw, snr_db=None, class_kpa=None, mu_mode=None, eta_mode="stage"):
    phys = cfg["physics"]
    mu_mode = mu_mode or phys["mu_mode"]
    kpa = class_kpa or phys["class_nominal_kpa"]
    X, y = [], []
    for name in CLASS_NAMES:  # sorted, as in the certified evaluation cells
        w = gen_windows(kpa[name], depth_cm / 100, phys["frequency_hz"], n_per_class, rng, class_name=name,
                        mu_mode=mu_mode, eta_mode=eta_mode, hw=hw, snr_db=snr_db,
                        jitter_std=phys["heart_rate_jitter_sd_hz"])
        X.extend(w)
        y.extend([LABEL[name]] * len(w))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


EVAL_NAMES = ["eval_clean", "eval_d0", "eval_d6.5", "eval_d8", "eval_d10"]


def build_dataset(cfg, seed):
    cid = cfg["config_id"]
    data = cfg["data"]
    hw = hw_from_cfg(cfg)
    ds = {}
    ds["train"] = _gen_set(cfg, stream_rng(cid, seed, "train"), data["train_depths_cm"],
                           data["train_windows_per_depth_per_class"], hw)
    ds["val"] = _gen_set(cfg, stream_rng(cid, seed, "val"), data["train_depths_cm"],
                         data["validation_windows_per_depth_per_class"], hw)
    n = data["eval_windows_per_class_per_cell"]
    ds["eval_clean"] = _gen_eval_cell(cfg, stream_rng(cid, seed, "eval_clean"), 0.0, n, HardwareConfig(),
                                      snr_db=float("inf"))
    for d, nm in zip(data["eval_depths_cm"], EVAL_NAMES[1:]):
        ds[nm] = _gen_eval_cell(cfg, stream_rng(cid, seed, nm), float(d), n, hw)
    return ds


def split_hash(ds):
    h = hashlib.sha256()
    for k in ["train", "val"] + EVAL_NAMES:
        X, y = ds[k]
        h.update(np.ascontiguousarray(X, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(y, dtype="<i8").tobytes())
    return h.hexdigest()


def get_dataset(config_id, seed):
    """Load the cached dataset or build and cache it (read-only file). Returns (ds, split_hash, built_now)."""
    cfg, _, _ = load_config(config_id)
    d = os.path.join(P2, "datasets", config_id)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"seed{seed}.npz")
    built = False
    if not os.path.exists(path):
        ds = build_dataset(cfg, seed)
        np.savez_compressed(path, **{f"{k}_X": v[0] for k, v in ds.items()}, **{f"{k}_y": v[1] for k, v in ds.items()})
        os.chmod(path, 0o444)
        built = True
    z = np.load(path)
    ds = {k: (z[f"{k}_X"], z[f"{k}_y"]) for k in ["train", "val"] + EVAL_NAMES}
    return ds, split_hash(ds), built


def git_commit():
    head = subprocess.run(["git", "-C", WORKTREE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", WORKTREE, "status", "--porcelain", "Source", "outputs/phase2/code", "outputs/phase2/configs"],
                           capture_output=True, text=True).stdout.strip()
    return head, bool(dirty)


def power_state():
    out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    src = "AC" if "AC Power" in out else "battery"
    lp = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    lpm = "?"
    for line in lp.splitlines():
        if "lowpowermode" in line:
            lpm = line.split()[-1]
    return f"{src}, LPM {'on' if lpm == '1' else 'off' if lpm == '0' else '?'}"


def power_label(start, end):
    return start if start == end else f"mixed ({start} -> {end})"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(ckpt_dir, pattern_suffixes):
    """Rewrite MANIFEST.sha256 from the files currently in ckpt_dir (sha256sum format)."""
    lines = []
    for fn in sorted(os.listdir(ckpt_dir)):
        if fn.endswith(pattern_suffixes):
            lines.append(f"{sha256_file(os.path.join(ckpt_dir, fn))}  {fn}")
    mp = os.path.join(ckpt_dir, "MANIFEST.sha256")
    if os.path.exists(mp):
        os.chmod(mp, 0o644)
    open(mp, "w").write("\n".join(lines) + "\n")
    os.chmod(mp, 0o444)


def acc(pred, y):
    return float(np.mean(pred == y))

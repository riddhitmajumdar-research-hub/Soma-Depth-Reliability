"""Shared simulator wrapper and certification harness pieces for Phase 1.

Reads Source/ from the worktree. Writes nothing under Source/. The generator below
draws random numbers in the same order as degradation_aware_final_sweep.generate_degraded_windows
when every option is at its default, which test_regression.py checks bit-for-bit.

Certification follows degradation_aware_final_sweep.run_one_seed and the cloud battery
harness (origin/cloud/battery-105-108): train one real model and 200 shuffled-label
controls on the baseline training set, then evaluate on a per-depth test cell, permutation
p-value against the controls, Benjamini-Hochberg at 0.05 over the 23-depth grid, floor =
first non-significant depth, or 15.0 (censored) when every depth is significant.
"""
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
SRC = os.path.join(WORKTREE, "Source")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from physics_model import stiffness_classes  # noqa: E402
from attenuation_model import (attenuation_coefficient, LIVER_VISCOSITY_BY_CLASS)  # noqa: E402
from cnn_classifier import extract_raw_windows  # noqa: E402
from constants import SENSOR_SNR_DB  # noqa: E402
from hardware_effects import (HardwareConfig, generate_signal_hw, add_sensor_noise,  # noqa: E402
                              apply_gain)

CLASS_NAMES = sorted(stiffness_classes.keys())

# ASSUMPTION: stage intervals for continuous intra-stage stiffness. Abutting intervals
# whose midpoints are the existing class values 4.5 / 7.5 / 12.0 kPa and whose union is
# the requested [3, 15] kPa. No source in the repo defines stage intervals.
STAGE_MU_RANGE_KPA = {
    "healthy": (3.0, 6.0),
    "significant_fibrosis": (6.0, 9.0),
    "cirrhosis": (9.0, 15.0),
}


def snr_to_std(snr_db):
    return 10 ** (-snr_db / 20)


def gen_windows(kpa_nominal, depth_m, freq_hz, n_needed, rng, class_name=None,
                jitter_std=0.03, mu_mode="fixed", mu_cv=None, eta_mode="lookup",
                eta_override=None, hw=None, snr_db=None, return_diag=False, eta_sampler=None):
    """mu_mode: 'fixed' (baseline), 'cv' (kpa*(1+cv*z), z~N(0,1) clipped at +/-2.5, as in the
    cloud battery condition 4), 'range' (uniform within STAGE_MU_RANGE_KPA[class_name]).
    eta_mode: 'lookup' uses attenuation_model's existing exact-match lookup, which returns 0.5
    for any kPa that is not exactly 4.5, 7.5 or 12.0; 'stage' uses the class eta from the class
    label. eta_override (Pa.s) replaces the eta in either mode."""
    hw = hw or HardwareConfig()
    std = snr_to_std(snr_db if snr_db is not None else SENSOR_SNR_DB)
    windows, diag = [], []
    while len(windows) < n_needed:
        hr = 1.15 + rng.normal(0, jitter_std)
        kpa = kpa_nominal
        if mu_mode == "cv":
            z = float(np.clip(rng.normal(0, 1), -2.5, 2.5))
            kpa = kpa_nominal * (1 + mu_cv * z)
        elif mu_mode == "range":
            lo, hi = STAGE_MU_RANGE_KPA[class_name]
            kpa = rng.uniform(lo, hi)

        time, clean, tof = generate_signal_hw(kpa, rng, hw, heart_rate_hz=hr)

        if eta_sampler is not None and class_name == "cirrhosis":
            eta = eta_sampler(rng)  # per-record draw; only used when a sampler is passed
        elif eta_override is not None:
            eta = eta_override
        else:
            eta = None
        # eta_mode 'stage' uses Source/attenuation_model's eta_by_stage flag (Phase 1 fix);
        # 'lookup' is the original exact-match lookup.
        alpha = attenuation_coefficient(kpa, freq_hz, eta=eta,
                                        eta_by_stage=(eta_mode == "stage"), stage=class_name)
        sig = clean * np.exp(-alpha * depth_m)
        sig = apply_gain(sig, rng, hw)
        noisy = add_sensor_noise(sig, std, rng, hw)
        rows = extract_raw_windows(noisy, time, hr)
        windows.extend(rows)
        if return_diag:
            diag.append({"kpa": kpa, "tof_ms": tof * 1e3, "alpha": alpha,
                         "peak_clean": float(np.abs(clean).max()),
                         "peak_atten": float(np.abs(clean * np.exp(-alpha * depth_m)).max())})
    out = windows[:n_needed]
    return (out, diag) if return_diag else out


def build_training_set(rng, mu_mode="fixed", eta_mode="lookup", train_depths_cm=(0, 3, 6, 9, 12),
                       n_per=25, freq_hz=100):
    X, y = [], []
    for d in train_depths_cm:
        for name, kpa in stiffness_classes.items():
            w = gen_windows(kpa, d / 100, freq_hz, n_per, rng, class_name=name,
                            mu_mode=mu_mode, eta_mode=eta_mode)
            X.extend(w)
            y.extend([name] * len(w))
    return np.array(X), np.array(y)


def cond_rng(seed, cond_id):
    """Deterministic per-(seed, condition) RandomState. The cloud battery used Python's
    str hash here, which changes between processes unless PYTHONHASHSEED is set."""
    return np.random.RandomState((seed * 100003 + zlib.crc32(cond_id.encode())) % (2 ** 32 - 1))


def bh_adjusted(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    raw = p[order] * n / np.arange(1, n + 1)
    adj_sorted = np.clip(np.minimum.accumulate(raw[::-1])[::-1], 0, 1)
    adj = np.empty(n)
    adj[order] = adj_sorted
    return adj

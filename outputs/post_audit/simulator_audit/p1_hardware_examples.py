"""Step 1.4: example signals, plots and summary statistics for each hardware setting,
plus an end-to-end shape and finiteness check. No model is trained and no certification
is run. Seed 20261002 for everything here.

Usage: p1_hardware_examples.py   (writes under outputs/phase1/figures and outputs/phase1/data/hardware)
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_common import (gen_windows, stiffness_classes, snr_to_std, WORKTREE, SENSOR_SNR_DB,  # noqa: E402
                       attenuation_coefficient, HardwareConfig, generate_signal_hw)
from hardware_effects import colored_noise  # noqa: E402

SEED = 20261002
FS = 500
OUT_F = os.path.join(WORKTREE, "outputs", "phase1", "figures")
OUT_D = os.path.join(WORKTREE, "outputs", "phase1", "data", "hardware")
os.makedirs(OUT_F, exist_ok=True)
os.makedirs(OUT_D, exist_ok=True)

SETTINGS = [
    ("baseline_off", HardwareConfig()),
    ("jitter_0.5ms", HardwareConfig(timing_jitter_ms=0.5)),
    ("jitter_1.0ms", HardwareConfig(timing_jitter_ms=1.0)),
    ("jitter_2.0ms", HardwareConfig(timing_jitter_ms=2.0)),
    ("colored_noise_beta1", HardwareConfig(noise_beta=1.0)),
    ("colored_noise_beta2", HardwareConfig(noise_beta=2.0)),
    ("gain_0.5_1.5", HardwareConfig(gain_range=(0.5, 1.5))),
    ("all_on_j1.0_b1_gain", HardwareConfig(timing_jitter_ms=1.0, noise_beta=1.0, gain_range=(0.5, 1.5))),
]
DEPTHS_CM = [0.0, 4.0]
NOISE_STD = snr_to_std(SENSOR_SNR_DB)
HR = 1.15


def one_record(kpa, depth_cm, hw, rng):
    """Same operations and order as gen_windows, but keeps the components apart."""
    shifts = []
    t, clean, tof = generate_signal_hw(kpa, rng, hw, heart_rate_hz=HR, shifts_out=shifts)
    alpha = attenuation_coefficient(kpa, 100)
    sig = clean * np.exp(-alpha * depth_cm / 100)
    g = 1.0
    if hw.gain_range is not None:
        g = rng.uniform(*hw.gain_range)
        sig = sig * g
    if hw.noise_beta is None:
        noise = rng.normal(0, NOISE_STD, size=sig.shape)
    else:
        noise = colored_noise(sig.size, NOISE_STD, hw.noise_beta, rng)
    return t, clean, sig, noise, np.array(shifts), g, tof


def measured_delay_ms(ref_sig, test_sig, beat_period=1 / HR):
    """Per-beat delay of test_sig against ref_sig (both noise-free) from cross-correlation with
    parabolic peak interpolation. Positive = test arrives later."""
    out = []
    for b in np.arange(0, 3.0, beat_period):
        i0 = int(b * FS); i1 = i0 + 100
        if i1 > len(ref_sig):
            continue
        a = ref_sig[i0:i1]; c = test_sig[i0:i1]
        if np.abs(a).max() == 0 or np.abs(c).max() == 0:
            continue
        xc = np.correlate(c, a, mode="full")
        k = int(np.argmax(xc))
        if 0 < k < len(xc) - 1:
            y0, y1, y2 = xc[k - 1], xc[k], xc[k + 1]
            den = (y0 - 2 * y1 + y2)
            frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        else:
            frac = 0.0
        lag = (k - (len(a) - 1)) + frac
        out.append(lag / FS * 1e3)
    return np.array(out)


def main():
    rng_master = np.random.RandomState(SEED)
    rows = []
    fig, axes = plt.subplots(len(SETTINGS), 1, figsize=(11, 2.0 * len(SETTINGS)), sharex=True)
    kpa = stiffness_classes["significant_fibrosis"]
    spec_fig, spec_ax = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, hw) in zip(axes, SETTINGS):
        # the same RandomState seed for every setting, so differences come from the setting
        rng = np.random.RandomState(SEED)
        t, clean, sig, noise, shifts, g, tof = one_record(kpa, 0.0, hw, rng)
        ax.plot(t, sig + noise, lw=0.6, color="#555", label="signal + noise")
        ax.plot(t, sig, lw=0.9, color="#0F6E56", label="signal")
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")
        ax.set_xlim(0, 1.6)
        if name == SETTINGS[0][0]:
            ax.legend(fontsize=7, loc="upper right")

        # statistics over 200 records per setting and depth
        for depth_cm in DEPTHS_CM:
            snr_peak, snr_rms, tim_applied, tim_meas, gains, noise_stds = [], [], [], [], [], []
            psd_n = []
            for r in range(200):
                rr = np.random.RandomState(rng_master.randint(0, 2 ** 31 - 1))
                t, clean, sig, noise, shifts, g, tof = one_record(kpa, depth_cm, hw, rr)
                pk = np.abs(sig).max()
                snr_peak.append(20 * np.log10(pk / noise.std()) if pk > 0 else np.nan)
                active = np.abs(sig) > 1e-12
                snr_rms.append(10 * np.log10(np.mean(sig[active] ** 2) / np.mean(noise ** 2)) if active.any() else np.nan)
                gains.append(g); noise_stds.append(noise.std())
                if hw.timing_jitter_ms > 0:
                    tim_applied.extend((shifts * 1e3).tolist())
                    # delay against the jitter-off signal of the same record geometry
                    _, ref, _ = generate_signal_hw(kpa, np.random.RandomState(0), HardwareConfig(), heart_rate_hz=HR)
                    tim_meas.extend(measured_delay_ms(ref, clean).tolist())
                f, p = welch(noise, fs=FS, nperseg=256)
                psd_n.append(p)
            psd_n = np.mean(psd_n, axis=0)
            if depth_cm == 0.0:
                spec_ax[0].semilogy(f[1:], psd_n[1:], label=name, lw=1.1)
            ta = np.array(tim_applied) if tim_applied else np.array([0.0])
            tm = np.array(tim_meas) if tim_meas else np.array([0.0])
            rows.append(dict(
                setting=name, depth_cm=depth_cm, n_records=200,
                snr_peak_db_mean=np.nanmean(snr_peak), snr_peak_db_sd=np.nanstd(snr_peak),
                snr_rms_db_mean=np.nanmean(snr_rms), snr_rms_db_sd=np.nanstd(snr_rms),
                noise_std_mean=np.mean(noise_stds), noise_std_target=NOISE_STD,
                gain_mean=np.mean(gains), gain_min=np.min(gains), gain_max=np.max(gains),
                timing_applied_ms_mean=ta.mean(), timing_applied_ms_sd=ta.std(), timing_applied_ms_maxabs=np.abs(ta).max(),
                timing_measured_ms_sd=tm.std(), timing_measured_ms_maxabs=np.abs(tm).max()))
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Significant-fibrosis record at 0 cm: one record per hardware setting, same seed", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_F, "p1_4_example_signals.png"), dpi=140)

    spec_ax[0].set_title("Noise PSD at 0 cm (mean of 200 records)")
    spec_ax[0].set_xlabel("frequency (Hz)"); spec_ax[0].set_ylabel("PSD (1/Hz)")
    spec_ax[0].legend(fontsize=6)
    # signal spectrum for baseline vs jitter settings (clean-with-jitter, no noise)
    for name, hw in SETTINGS[:4]:
        ps = []
        for r in range(100):
            rr = np.random.RandomState(1000 + r)
            t, clean, sig, noise, shifts, g, tof = one_record(kpa, 0.0, hw, rr)
            f, p = welch(sig, fs=FS, nperseg=256)
            ps.append(p)
        spec_ax[1].semilogy(f[1:], np.mean(ps, axis=0)[1:], label=name, lw=1.1)
    spec_ax[1].set_title("Signal PSD at 0 cm, jitter settings (no noise)")
    spec_ax[1].set_xlabel("frequency (Hz)"); spec_ax[1].legend(fontsize=7)
    spec_fig.tight_layout()
    spec_fig.savefig(os.path.join(OUT_F, "p1_4_spectra.png"), dpi=140)

    with open(os.path.join(OUT_D, "hardware_summary_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # end-to-end check: dataset build through window extraction, then a forward pass of an
    # untrained SimpleCNN (shape check only; nothing is trained or evaluated for accuracy)
    import torch
    from cnn_classifier import SimpleCNN
    torch.manual_seed(SEED)
    net = SimpleCNN(n_classes=3).eval()
    chk = []
    base_shape = None
    for name, hw in SETTINGS:
        rng = np.random.RandomState(SEED)
        X = []
        for depth_cm in (0.0, 4.0, 15.0):
            for cname, k in stiffness_classes.items():
                X.extend(gen_windows(k, depth_cm / 100, 100, 50, rng, class_name=cname, hw=hw))
        X = np.array(X)
        with torch.no_grad():
            out = net(torch.tensor(X, dtype=torch.float32).unsqueeze(1))
        if base_shape is None:
            base_shape = X.shape
        chk.append(dict(setting=name, X_shape=str(X.shape), finite=bool(np.isfinite(X).all()),
                        shape_matches_baseline=(X.shape == base_shape), logits_shape=str(tuple(out.shape)),
                        logits_finite=bool(torch.isfinite(out).all()), x_std=float(X.std()), x_absmax=float(np.abs(X).max())))
    with open(os.path.join(OUT_D, "hardware_pipeline_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chk[0].keys()))
        w.writeheader(); w.writerows(chk)
    for c in chk:
        print(c)
    print("stats rows:", len(rows))


if __name__ == "__main__":
    main()

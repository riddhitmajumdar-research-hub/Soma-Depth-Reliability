# SOMA -- optional hardware effects for the synthetic signal path (Phase 1, Step 1.4)
# File: Source/hardware_effects.py
#
# Every effect is OFF by default. With the default HardwareConfig(), the functions here
# draw the same random numbers in the same order as the existing path in
# physics_model.generate_signal and degradation_aware_final_sweep.generate_degraded_windows,
# so existing results are unchanged. That is checked bit-for-bit by
# outputs/phase1/code/test_regression.py.
#
# No existing file under Source/ is edited. This module is a parallel path that the
# Phase 1 harness uses; wiring it into the certified sweep is a Phase 2 decision.
#
# ASSUMPTIONS (none of these has a source in the repo):
#   - Timing jitter is uniform on [-J, +J] ms, drawn independently per heartbeat and added
#     to that beat's pulse arrival time. The requested settings are J = 0.5, 1.0, 2.0 ms.
#     At 500 Hz one sample is 2 ms, so 0.5 and 1.0 ms are sub-sample shifts. The pulse
#     shape is evaluated analytically at the shifted arrival time, so sub-sample shifts act
#     on the waveform and not only on the start index.
#   - Colored noise has PSD proportional to 1/f^beta. beta = 1 (pink) is a placeholder; the
#     repo has no measured MEMS noise spectrum. The colored noise is rescaled to the same
#     total standard deviation as the white-noise baseline, so SNR stays at SENSOR_SNR_DB
#     and only the spectral shape changes.
#   - Random gain is uniform on [lo, hi] per record, applied after attenuation and before
#     noise. (0.5, 1.5) matches the corrected amplitude draw in the cloud robustness battery.

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from physics_model import time_of_flight_s


@dataclass
class HardwareConfig:
    timing_jitter_ms: float = 0.0                       # 0 = off; J for U(-J, +J) ms per beat
    noise_beta: Optional[float] = None                  # None = white (baseline); else 1/f^beta
    gain_range: Optional[Tuple[float, float]] = None    # None = off; else U(lo, hi) per record


def generate_signal_hw(youngs_modulus_kpa, rng, hw, duration=3.0, heart_rate_hz=1.15,
                       sample_rate=500, shifts_out=None):
    """Copy of physics_model.generate_signal with optional per-beat arrival jitter.
    With hw.timing_jitter_ms == 0 no random number is drawn and the output is identical."""
    tof = time_of_flight_s(youngs_modulus_kpa)
    beat_times = np.arange(0, duration, 1.0 / heart_rate_hz)

    n_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)

    for beat_t in beat_times:
        arrival_t = beat_t + tof
        if hw.timing_jitter_ms > 0:
            shift_s = rng.uniform(-hw.timing_jitter_ms, hw.timing_jitter_ms) * 1e-3
            arrival_t += shift_s
            if shifts_out is not None:
                shifts_out.append(shift_s)
        pulse_window = 0.15

        mask = (time >= arrival_t) & (time <= arrival_t + pulse_window)
        local_t = time[mask] - arrival_t
        pulse_shape = np.exp(-15 * local_t) * np.sin(2 * np.pi * 100 * local_t)
        signal[mask] += pulse_shape

    return time, signal, tof


def colored_noise(n, std, beta, rng):
    """Zero-mean noise with PSD ~ 1/f^beta, rescaled to the requested std.
    The DC bin is zeroed. The rescale uses the sample std of this draw, so every call
    returns a vector with exactly the requested std."""
    white = rng.normal(0, 1, size=n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    shape = np.ones_like(f)
    shape[1:] = f[1:] ** (-beta / 2.0)
    shape[0] = 0.0
    x = np.fft.irfft(spec * shape, n=n)
    x = x - x.mean()
    s = x.std()
    return x * (std / s) if s > 0 else x


def add_sensor_noise(signal, noise_std, rng, hw):
    if hw.noise_beta is None:
        return signal + rng.normal(0, noise_std, size=signal.shape)
    return signal + colored_noise(signal.size, noise_std, hw.noise_beta, rng)


def apply_gain(signal, rng, hw):
    if hw.gain_range is None:
        return signal
    lo, hi = hw.gain_range
    return signal * rng.uniform(lo, hi)

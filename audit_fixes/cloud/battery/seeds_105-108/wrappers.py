# Battery-A wrappers.
# Each function below copies the relevant original Source/ function's body
# verbatim except the ONE stated change for that condition, and takes a
# `shift` dict of parameters. Nothing in Source/ is imported-and-monkeypatched
# or edited. See PREREG.md for the full protocol and disclosed judgment calls.

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(HERE, "..", "..", "..", "Source")
sys.path.insert(0, SOURCE_DIR)

from physics_model import generate_signal, shear_wave_velocity_m_s, stiffness_classes  # noqa: E402
from attenuation_model import attenuation_coefficient, LIVER_VISCOSITY_BY_CLASS, _eta_for_kpa  # noqa: E402
from cnn_classifier import extract_raw_windows  # noqa: E402
from constants import SENSOR_SNR_DB  # noqa: E402

RHO = 1060.0


def snr_db_to_std(snr_db):
    """Identical convention to degradation_aware_final_sweep.SENSOR_NOISE_STD."""
    return 10 ** (-snr_db / 20)


BASELINE_NOISE_STD = snr_db_to_std(SENSOR_SNR_DB)


def alpha_exact(E_kpa, eta, frequency_hz, rho=RHO):
    """Condition 6: exact complex-wavenumber attenuation,
    alpha_exact = |Im k|, k = w*sqrt(rho/(G + i*w*eta)), G = E/3 (kPa->Pa)."""
    G_pa = E_kpa * 1000 / 3.0
    w = 2 * np.pi * frequency_hz
    k = w * np.sqrt(rho / (G_pa + 1j * w * eta))
    return abs(k.imag)


def c_exact(E_kpa, eta, frequency_hz, rho=RHO):
    """Condition 6b: exact phase velocity c_exact = w / Re(k)."""
    G_pa = E_kpa * 1000 / 3.0
    w = 2 * np.pi * frequency_hz
    k = w * np.sqrt(rho / (G_pa + 1j * w * eta))
    return w / k.real


def kpa_eff_from_c(c, rho=RHO):
    """Inverse of shear_wave_velocity_m_s under G=E/3, rho fixed:
    c = sqrt(G/rho) = sqrt((E_pa/3)/rho)  =>  E_pa = 3*rho*c^2  =>  kPa."""
    return 3 * rho * c ** 2 / 1000.0


def generate_degraded_windows_shifted(kpa, depth_m, frequency_hz, n_needed,
                                       jitter_std=0.03, shift=None,
                                       class_name=None, return_beat_idx=False,
                                       rng=None):
    """Verbatim copy of degradation_aware_final_sweep.generate_degraded_windows,
    with each stated shift applied at the point the task specifies. shift is a
    dict; recognized keys (all optional, default = original behavior):

      amp_gain: bool -- condition 2, multiply by g=exp(U(ln .5, ln 2)) after
        attenuation, before noise.
      amp_gain_linear: bool -- condition 2b, multiply by g~Uniform(0.5,1.5)
        (linear space, mean exactly 1.0) after attenuation, before noise.
        Corrected replacement for amp_gain's log-uniform draw.
      spreading_p: float -- condition 3, multiply by min(1, (x0/x)^p) with
        x0=0.01m, applied alongside the existing exponential decay (which has
        no spreading term of its own -- confirmed by reading attenuate_signal).
      stiffness_cv: float -- condition 4, per-signal kpa_eff =
        kpa*(1 + cv*z), z~N(0,1) truncated at +-2.5, used for BOTH tof
        (generate_signal) and attenuation.
      snr_db: float -- condition 5, override SENSOR_SNR_DB for the noise std.
      exact_atten: bool -- condition 6/6a, use alpha_exact instead of the
        original weak-damping attenuation_coefficient.
      exact_tof: bool -- condition 6b, additionally use c_exact -> kpa_eff to
        also perturb generate_signal's TOF (implies exact_atten).
      eta_override: float -- condition 7, override this class's eta (Pa.s)
        used by both attenuation_coefficient/alpha_exact.
    """
    shift = shift or {}
    rng = rng if rng is not None else np.random
    windows = []
    beat_idxs = []
    while len(windows) < n_needed:
        jittered_hr = 1.15 + rng.normal(0, jitter_std)

        kpa_signal = kpa  # kPa passed to generate_signal (TOF)
        kpa_atten = kpa   # kPa used for attenuation coefficient lookup

        if "stiffness_cv" in shift:
            z = rng.normal(0, 1)
            z = float(np.clip(z, -2.5, 2.5))
            kpa_eff = kpa * (1 + shift["stiffness_cv"] * z)
            kpa_signal = kpa_eff
            kpa_atten = kpa_eff

        time, clean_signal, tof = generate_signal(kpa_signal, heart_rate_hz=jittered_hr)

        eta = shift.get("eta_override")
        if eta is None:
            eta = _eta_for_kpa(kpa_atten)

        if shift.get("exact_tof"):
            # condition 6b: also perturb TOF via c_exact -> kpa_eff, then
            # regenerate the signal with that effective kPa (kpa_atten
            # unaffected -- physical eta/E for attenuation stay as-is,
            # only the TOF-generating call uses c_exact's implied stiffness).
            c_ex = c_exact(kpa_atten, eta, frequency_hz)
            kpa_tof_eff = kpa_eff_from_c(c_ex)
            time, clean_signal, tof = generate_signal(kpa_tof_eff, heart_rate_hz=jittered_hr)

        if shift.get("exact_atten") or shift.get("exact_tof"):
            alpha = alpha_exact(kpa_atten, eta, frequency_hz)
        else:
            alpha = attenuation_coefficient(kpa_atten, frequency_hz, eta=eta)

        decay_factor = np.exp(-alpha * depth_m)
        attenuated = clean_signal * decay_factor

        if "spreading_p" in shift:
            x0 = 0.01  # 1 cm
            x = max(depth_m, 1e-6)
            spread = min(1.0, (x0 / x) ** shift["spreading_p"])
            attenuated = attenuated * spread

        if shift.get("amp_gain"):
            g = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
            attenuated = attenuated * g

        if shift.get("amp_gain_linear"):
            # condition 2b: corrected distribution -- g ~ Uniform(0.5, 1.5)
            # in LINEAR space (mean exactly 1.0), replacing condition 2's
            # log-uniform g ~ exp(U(ln .5, ln 2)) (E[g]~1.08, biased high).
            g = rng.uniform(0.5, 1.5)
            attenuated = attenuated * g

        noise_std = snr_db_to_std(shift.get("snr_db", SENSOR_SNR_DB))
        noisy = attenuated + rng.normal(0, noise_std, size=attenuated.shape)

        rows = extract_raw_windows(noisy, time, jittered_hr)
        if return_beat_idx:
            beat_idxs.extend(range(len(rows)))
        windows.extend(rows)

    if return_beat_idx:
        return windows[:n_needed], beat_idxs[:n_needed]
    return windows[:n_needed]


def build_pure_noise_windows(n_needed, snr_db, window_size=100, rng=None):
    """Condition 8: pure-noise probe -- no pulse at all."""
    rng = rng if rng is not None else np.random
    std = snr_db_to_std(snr_db)
    return rng.normal(0, std, size=(n_needed, window_size))

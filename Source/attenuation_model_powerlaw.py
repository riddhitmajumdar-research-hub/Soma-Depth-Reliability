# Stage 4+ -- Power-Law Attenuation Sensitivity Model
# File: Source/attenuation_model_powerlaw.py
#
# Tests how much the certified degradation-aware CNN's accuracy depends on
# the Kelvin-Voigt model's derived alpha ~ omega^2 frequency scaling, versus
# the power-law scaling actually measured in liver tissue.
#
# Sourced exponent: n = 0.51, from a 2025 shear-wave dispersion study
# (Multi-Frequency Reverberant OCE) measuring liver-specific shear-wave
# attenuation directly on ex vivo bovine liver -- chosen over the broader
# compressional-ultrasound power-law literature (typically n in [1,2])
# because SOMA's physics is shear-wave based, where the literature range
# is n in [0,1] and liver specifically measures close to 0.51.
#
# Anchored at 20Hz rather than SOMA's operating 100Hz, deliberately: at 20Hz
# the Kelvin-Voigt weak-damping assumption (omega*eta << G) holds most
# comfortably (ratio ~0.042 for healthy tissue), so anchoring there and
# letting the two models diverge toward 100Hz tests the assumption honestly,
# rather than forcing agreement at the exact frequency being tested.

import numpy as np
from attenuation_model import attenuation_coefficient

ANCHOR_FREQ_HZ = 20
POWER_LAW_EXPONENT = 0.51


def attenuation_coefficient_powerlaw(youngs_modulus_kpa, frequency_hz,
                                      eta=None,
                                      n=POWER_LAW_EXPONENT,
                                      anchor_freq=ANCHOR_FREQ_HZ):
    # FIX (post-audit): previously imported a LIVER_VISCOSITY_PA_S constant that
    # no longer exists after Fix #3 made viscosity class-dependent. eta=None now
    # lets attenuation_coefficient() resolve the correct per-class eta itself,
    # matching Kelvin-Voigt'"'"'s behavior instead of silently using one fixed value.
    """Power-law attenuation, anchored to match Kelvin-Voigt exactly at anchor_freq."""
    alpha_anchor = attenuation_coefficient(youngs_modulus_kpa, anchor_freq, eta=eta)
    alpha = alpha_anchor * (frequency_hz / anchor_freq) ** n
    return alpha


def attenuate_signal_powerlaw(signal, youngs_modulus_kpa, depth_m, frequency_hz):
    """Drop-in replacement for attenuate_signal(), using the power-law model."""
    alpha = attenuation_coefficient_powerlaw(youngs_modulus_kpa, frequency_hz)
    decay_factor = np.exp(-alpha * depth_m)
    return signal * decay_factor, alpha


if __name__ == "__main__":
    from physics_model import stiffness_classes
    from attenuation_model import attenuation_coefficient as alpha_kv

    print("=" * 78)
    print(f"KELVIN-VOIGT vs POWER-LAW ATTENUATION -- anchor={ANCHOR_FREQ_HZ}Hz, n={POWER_LAW_EXPONENT}")
    print("=" * 78)
    print(f"{'Class':<22s}{'Freq (Hz)':>10s}{'KV alpha (1/m)':>18s}{'PL alpha (1/m)':>18s}{'Ratio (KV/PL)':>16s}")
    for name, kpa in stiffness_classes.items():
        for freq in [20, 100, 500]:
            a_kv = alpha_kv(kpa, freq)
            a_pl = attenuation_coefficient_powerlaw(kpa, freq)
            ratio = a_kv / a_pl if a_pl > 0 else float("nan")
            print(f"{name:<22s}{freq:>10d}{a_kv:>18.4f}{a_pl:>18.4f}{ratio:>16.2f}")

    print("\nAt the anchor frequency (20Hz), KV and PL are identical by construction --")
    print("the ratio column shows how much they diverge as frequency moves away from it.")
    print(f"At SOMA's actual 100Hz operating frequency, this is the divergence that matters.")

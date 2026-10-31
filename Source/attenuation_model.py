# Stage 2 — Noise & Attenuation Model
# File: Source/attenuation_model.py

import numpy as np
from physics_model import shear_wave_velocity_m_s, stiffness_classes

# --- FIX #3: class-dependent viscosity -------------------------------------
# PREVIOUS: single eta=0.5 Pa.s for all three classes, justified by an
# uncited "rat-model SDUV/DMA comparisons" claim that couldn't be verified
# (the closest real match found -- a rat NAFLD SWE-vs-DMA study -- actually
# reports poor cross-method correlation, not stage-invariance).
#
# REPLACED with: Reiter et al. 2021, J Mech Behav Biomed Mater 121:104645
# (multifrequency MRE, 15 healthy + 37 biopsy-confirmed CLD patients,
# 35-60Hz). Loss angle phi: F0=0.63+/-0.05 rad, F1-F3=0.60+/-0.05 rad (NOT
# significantly different from F0, p=0.21), F4=0.81+/-0.16 rad
# (significantly elevated, p<0.001). Quote: "Cirrhosis is characterized by
# stiff yet viscous tissue properties (high loss angle)... Mild/severe
# fibrosis is characterized by solid tissue properties and low loss angle."
#
# So: viscosity is stage-invariant between healthy and significant_fibrosis,
# but rises distinctly at cirrhosis.
#
# HONEST LIMITATION: Reiter et al. report loss angle (phi), not eta (Pa.s)
# directly, and the storage-modulus values needed to convert phi->eta via
# tan(phi)=omega*eta/G are behind a paywall I couldn't access. Rather than
# fabricate that conversion, cirrhosis's eta below uses the ALREADY-SOURCED
# upper bound (1.77 Pa.s) from the same viscosity study used for the base
# value (2%-fat vs 22%-fat liver comparison) -- a fat-content axis, not a
# fibrosis-stage axis. Treat this as directionally-correct and
# magnitude-approximate, not a precise derivation. Recommend running this
# through the existing Stage 3b/3g sensitivity-sweep machinery to see how
# much the Stage 4 floor moves across a plausible range for this value.
LIVER_VISCOSITY_BY_CLASS = {
    "healthy": 0.5,
    "significant_fibrosis": 0.5,   # indistinguishable from healthy (p=0.21, Reiter et al.)
    "cirrhosis": 1.77,             # distinctly elevated (p<0.001, Reiter et al.) -- magnitude approximated
}
LIVER_VISCOSITY_PA_S_DEFAULT = 0.5

_KPA_TO_CLASS = {v: k for k, v in stiffness_classes.items()}


def _eta_for_kpa(youngs_modulus_kpa):
    name = _KPA_TO_CLASS.get(youngs_modulus_kpa)
    if name is None:
        return LIVER_VISCOSITY_PA_S_DEFAULT
    return LIVER_VISCOSITY_BY_CLASS[name]


# --- eta_by_stage flag ---
# _eta_for_kpa() matches the stiffness against the three class values by exact float
# equality. Any other kPa (continuous within-stage stiffness) falls through to
# LIVER_VISCOSITY_PA_S_DEFAULT = 0.5, so a cirrhosis record at 11.99 kPa gets eta 0.5
# instead of 1.77. With eta_by_stage on, eta comes from the stage label and stiffness
# can vary inside a stage without changing eta. eta is not interpolated between class
# values. Default is off, so existing results are unchanged.
ETA_BY_STAGE = False


def attenuation_coefficient(youngs_modulus_kpa, frequency_hz, eta=None,
                            eta_by_stage=None, stage=None):
    if eta is None:
        use_stage = ETA_BY_STAGE if eta_by_stage is None else eta_by_stage
        if use_stage:
            if stage is None:
                raise ValueError("eta_by_stage is on but no stage label was given")
            eta = LIVER_VISCOSITY_BY_CLASS[stage]
        else:
            eta = _eta_for_kpa(youngs_modulus_kpa)

    E_pa = youngs_modulus_kpa * 1000
    G_pa = E_pa / 3
    c = shear_wave_velocity_m_s(youngs_modulus_kpa)

    omega = 2 * np.pi * frequency_hz
    alpha = (omega ** 2 * eta) / (2 * G_pa * c)

    return alpha


def attenuate_signal(signal, youngs_modulus_kpa, depth_m, frequency_hz,
                     eta_by_stage=None, stage=None):
    alpha = attenuation_coefficient(youngs_modulus_kpa, frequency_hz,
                                    eta_by_stage=eta_by_stage, stage=stage)
    decay_factor = np.exp(-alpha * depth_m)

    return signal * decay_factor, alpha


if __name__ == "__main__":
    from physics_model import generate_signal

    print("Attenuation coefficient (alpha, 1/m) by stiffness class and frequency:")
    print("(now using class-dependent eta -- see FIX #3 note above)")
    for name, kpa in stiffness_classes.items():
        eta_used = _eta_for_kpa(kpa)
        for freq in [20, 100, 500]:
            alpha = attenuation_coefficient(kpa, freq)
            print(f"  {name:22s} f={freq:4d}Hz   eta={eta_used:.2f} Pa.s   alpha={alpha:.4f} /m")

    print("\nSignal amplitude retained at 8cm depth (healthy tissue, 100Hz):")
    time, clean_signal, tof = generate_signal(4.5)
    attenuated_signal, alpha = attenuate_signal(clean_signal, 4.5, depth_m=0.08, frequency_hz=100)
    print(f"  alpha = {alpha:.4f} /m")
    print(f"  peak amplitude before: {np.max(np.abs(clean_signal)):.4f}")
    print(f"  peak amplitude after:  {np.max(np.abs(attenuated_signal)):.4f}")

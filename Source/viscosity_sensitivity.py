# SOMA Signal Fidelity Investigation
# Stage 3b — Viscosity sensitivity sweep
# File: Source/viscosity_sensitivity.py

import numpy as np
import matplotlib.pyplot as plt
from physics_model import generate_signal, shear_wave_velocity_m_s
from attenuation_model import LIVER_VISCOSITY_BY_CLASS

# Endpoints anchored to the canonical healthy/cirrhosis values in
# attenuation_model.py; 0.9 and 1.3 are intermediate sweep points, not
# duplicated constants -- those stay hardcoded on purpose.
VISCOSITY_VALUES = [
    LIVER_VISCOSITY_BY_CLASS["healthy"],
    0.9,
    1.3,
    LIVER_VISCOSITY_BY_CLASS["cirrhosis"],
]

HEALTHY_KPA = 4.5  # tests the fastest-collapsing class -- clearest signal


def attenuation_coefficient(youngs_modulus_kpa, frequency_hz, eta):
    E_pa = youngs_modulus_kpa * 1000
    G_pa = E_pa / 3
    c = shear_wave_velocity_m_s(youngs_modulus_kpa)
    omega = 2 * np.pi * frequency_hz
    alpha = (omega ** 2 * eta) / (2 * G_pa * c)
    return alpha


def depth_at_threshold(youngs_modulus_kpa, frequency_hz, eta, threshold_pct=10):
    alpha = attenuation_coefficient(youngs_modulus_kpa, frequency_hz, eta)
    return -np.log(threshold_pct / 100) / alpha


if __name__ == "__main__":
    time, signal, tof = generate_signal(HEALTHY_KPA)
    peak_before = np.max(np.abs(signal))

    depths_cm = np.linspace(0, 15, 100)
    depths_m = depths_cm / 100

    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis
    colors = [cmap(i / (len(VISCOSITY_VALUES) - 1)) for i in range(len(VISCOSITY_VALUES))]

    print("Depth at which signal drops to 10% amplitude, by viscosity:")
    for eta, color in zip(VISCOSITY_VALUES, colors):
        retained = []
        for d in depths_m:
            alpha = attenuation_coefficient(HEALTHY_KPA, 100, eta)
            peak_after = peak_before * np.exp(-alpha * d)
            retained.append(100 * peak_after / peak_before)
        plt.plot(depths_cm, retained, label=f"eta = {eta} Pa\u00b7s", color=color)

        d10_m = depth_at_threshold(HEALTHY_KPA, 100, eta, threshold_pct=10)
        print(f"  eta = {eta:.2f} Pa\u00b7s   ->   10% depth = {d10_m*100:.2f} cm")

    plt.axhline(y=10, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Depth (cm)")
    plt.ylabel("Signal amplitude retained (%)")
    plt.title("Viscosity sensitivity: healthy tissue, 100Hz, across sourced literature range")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("stage3b_viscosity_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.show()

# SOMA Signal Fidelity Investigation
# Stage 3g -- Joint uncertainty propagation
# File: Source/joint_uncertainty.py

import numpy as np
import matplotlib.pyplot as plt

from constants import SEED, NOMINAL_FREQ_HZ

np.random.seed(SEED)

N_SAMPLES = 5000
HEALTHY_KPA = 4.5
FREQUENCY_HZ = NOMINAL_FREQ_HZ

from attenuation_model import LIVER_VISCOSITY_BY_CLASS
VISCOSITY_RANGE = (LIVER_VISCOSITY_BY_CLASS["healthy"], LIVER_VISCOSITY_BY_CLASS["cirrhosis"])
DENSITY_RANGE = (1060 * 0.92, 1060 * 1.08)


def shear_wave_velocity(E_kpa, rho):
    E_pa = E_kpa * 1000
    G_pa = E_pa / 3
    return (G_pa / rho) ** 0.5


def attenuation_coefficient(E_kpa, frequency_hz, eta, rho):
    E_pa = E_kpa * 1000
    G_pa = E_pa / 3
    c = shear_wave_velocity(E_kpa, rho)
    omega = 2 * np.pi * frequency_hz
    return (omega ** 2 * eta) / (2 * G_pa * c)


def depth_at_threshold(E_kpa, frequency_hz, eta, rho, threshold_pct=10):
    alpha = attenuation_coefficient(E_kpa, frequency_hz, eta, rho)
    return -np.log(threshold_pct / 100) / alpha


if __name__ == "__main__":
    viscosity_samples = np.random.uniform(*VISCOSITY_RANGE, N_SAMPLES)
    density_samples = np.random.uniform(*DENSITY_RANGE, N_SAMPLES)

    depth_results = np.array([
        depth_at_threshold(HEALTHY_KPA, FREQUENCY_HZ, eta, rho)
        for eta, rho in zip(viscosity_samples, density_samples)
    ]) * 100

    print(f"Joint Monte Carlo over viscosity and density ({N_SAMPLES} draws):")
    print(f"  Mean 10%-depth:   {depth_results.mean():.3f} cm")
    print(f"  Std deviation:    {depth_results.std():.3f} cm")
    print(f"  5th percentile:   {np.percentile(depth_results, 5):.3f} cm")
    print(f"  95th percentile:  {np.percentile(depth_results, 95):.3f} cm")
    print(f"  Min / Max:        {depth_results.min():.3f} / {depth_results.max():.3f} cm")

    print(f"\nFor comparison, Stage 3b (viscosity varied alone, density fixed):")
    print(f"  Range was 1.18 cm - 4.16 cm")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(depth_results, bins=50, color="#0F6E56", alpha=0.75, edgecolor="white")
    axes[0].axvline(np.percentile(depth_results, 5), color="#993C1D", linestyle="--",
                     linewidth=1, label="5th / 95th percentile")
    axes[0].axvline(np.percentile(depth_results, 95), color="#993C1D", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Depth at 10% amplitude (cm)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of critical depth\n(joint viscosity + density uncertainty)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    scatter = axes[1].scatter(viscosity_samples, density_samples, c=depth_results,
                                cmap="viridis", s=4, alpha=0.5)
    axes[1].set_xlabel("Viscosity (Pa\u00b7s)")
    axes[1].set_ylabel("Density (kg/m\u00b3)")
    axes[1].set_title("Parameter space, colored by resulting\ncritical depth (cm)")
    plt.colorbar(scatter, ax=axes[1], label="Depth at 10% (cm)")

    plt.suptitle("Stage 3g -- Joint uncertainty propagation")
    plt.tight_layout()
    plt.savefig("stage3g_joint_uncertainty.png", dpi=200, bbox_inches="tight")
    plt.show()

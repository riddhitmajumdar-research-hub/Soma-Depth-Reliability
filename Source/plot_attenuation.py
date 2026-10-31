# Stage 2 — Attenuation visualization
# File: Source/plot_attenuation.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from physics_model import generate_signal, stiffness_classes
from attenuation_model import attenuate_signal, attenuation_coefficient
from constants import NOMINAL_FREQ_HZ

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "figures_final")
os.makedirs(FIG_DIR, exist_ok=True)
SCRATCH_DIR = os.path.expanduser("~/soma_audit_scratch/figplan")  # Panel 1 (not a record figure) goes here, not into the repo

# --- Panel 1: before vs after, healthy tissue, one fixed depth ---
time, clean_signal, tof = generate_signal(4.5)
attenuated_signal, alpha = attenuate_signal(clean_signal, 4.5, depth_m=0.08, frequency_hz=100)

fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(time, clean_signal, color="#0F6E56")
axes[0].set_title("Healthy tissue, clean signal (no attenuation)")
axes[0].set_ylabel("Amplitude")
axes[0].set_xlim(0, 1.0)
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, attenuated_signal, color="#993C1D")
axes[1].set_title(f"Same signal after 8cm depth, 100Hz attenuation (alpha={alpha:.2f}/m)")
axes[1].set_ylabel("Amplitude")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_xlim(0, 1.0)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SCRATCH_DIR, "stage2_before_after.png"), dpi=200, bbox_inches="tight")
plt.close()

# --- Panel 2: amplitude retained vs depth, all three classes, 100Hz ---
depths_cm = np.linspace(0, 15, 100)
depths_m = depths_cm / 100

plt.figure(figsize=(10, 6))
colors = {"healthy": "#0F6E56", "significant_fibrosis": "#BA7517", "cirrhosis": "#993C1D"}
display_names = {"healthy": "Healthy", "significant_fibrosis": "Fibrosis", "cirrhosis": "Cirrhosis"}

for name, kpa in stiffness_classes.items():
    time, signal, tof = generate_signal(kpa)
    peak_before = np.max(np.abs(signal))
    retained = []
    for d in depths_m:
        att_signal, alpha = attenuate_signal(signal, kpa, depth_m=d, frequency_hz=NOMINAL_FREQ_HZ)
        peak_after = np.max(np.abs(att_signal))
        retained.append(100 * peak_after / peak_before)
    alpha_100 = attenuation_coefficient(kpa, NOMINAL_FREQ_HZ)
    plt.plot(depths_cm, retained, label=f"{display_names[name]} (alpha = {alpha_100:.1f} 1/m)", color=colors[name])

plt.axhline(y=10, color="gray", linestyle="--", linewidth=0.8, label="10% retained")
plt.xlabel("Depth (cm)")
plt.ylabel("Signal amplitude retained (%)")
plt.title(f"Amplitude retained vs depth at {NOMINAL_FREQ_HZ} Hz, by tissue class")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIG_DIR, "fig03_amplitude_vs_depth.png"), dpi=200, bbox_inches="tight")
plt.close()
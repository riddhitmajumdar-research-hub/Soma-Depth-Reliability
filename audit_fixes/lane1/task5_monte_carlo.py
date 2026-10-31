"""
Lane 1, Task 5 -- Monte Carlo (5,000 samples) over the depth at which the
100Hz EXACT-model amplitude equals the nominal (fixed-eta,fixed-rho) amplitude
at 8.48cm. HEURISTIC amplitude-equivalence, NOT a classifier prediction.

Sampling ranges reused EXACTLY from Source/joint_uncertainty.py (found by
grep, read directly, not re-derived):
  N_SAMPLES = 5000
  VISCOSITY_RANGE = (LIVER_VISCOSITY_BY_CLASS["healthy"], LIVER_VISCOSITY_BY_CLASS["cirrhosis"])
                   = (0.5, 1.77) Pa.s, uniform
  DENSITY_RANGE = (1060*0.92, 1060*1.08), uniform  [density +/-8%, matches protocol]
  seed: constants.SEED (42), matching joint_uncertainty.py's np.random.seed(SEED)

Class fixed at "healthy" (E=4.5kPa) to match 8.48cm as reported at nominal
(healthy, 100Hz) conditions in constants.py -- same class joint_uncertainty.py
itself uses (HEALTHY_KPA).

Nominal (non-MC) amplitude-equivalent point uses the class's nominal eta
(LIVER_VISCOSITY_BY_CLASS["healthy"] = 0.5 Pa.s) and nominal rho=1060, i.e.
the exact-model alpha at those nominal values -- this defines the "nominal
amplitude at 8.48cm" the MC samples are compared against.
"""
import sys
import os
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Source"))
from constants import SEED, NOMINAL_FREQ_HZ  # noqa: E402
from physics_model import stiffness_classes  # noqa: E402
from attenuation_model import LIVER_VISCOSITY_BY_CLASS  # noqa: E402
from joint_uncertainty import VISCOSITY_RANGE, DENSITY_RANGE, N_SAMPLES  # noqa: E402 -- reused, not redefined

CLASS = "healthy"
E_KPA = stiffness_classes[CLASS]
FREQ_HZ = NOMINAL_FREQ_HZ
X_REF_CM = 8.48


def alpha_exact(E_kpa, freq_hz, eta, rho):
    G_pa = (E_kpa * 1000) / 3
    omega = 2 * np.pi * freq_hz
    G_star = G_pa + 1j * omega * eta
    k = omega * np.sqrt(rho / G_star)
    return abs(np.imag(k))


# Nominal (non-MC) reference alpha, at the class's nominal eta/rho.
nominal_eta = LIVER_VISCOSITY_BY_CLASS[CLASS]
nominal_rho = 1060.0
alpha_nominal = alpha_exact(E_KPA, FREQ_HZ, nominal_eta, nominal_rho)
print(f"Nominal alpha_exact ({CLASS}, {FREQ_HZ}Hz, eta={nominal_eta}, rho={nominal_rho}): {alpha_nominal:.4f} /m")
print(f"Reference depth x_ref = {X_REF_CM} cm (24-seed mean certified floor)")
print(f"Sampling ranges reused from Source/joint_uncertainty.py: "
      f"VISCOSITY_RANGE={VISCOSITY_RANGE}, DENSITY_RANGE=({DENSITY_RANGE[0]:.2f}, {DENSITY_RANGE[1]:.2f}), "
      f"N_SAMPLES={N_SAMPLES}, seed={SEED}")

np.random.seed(SEED)
eta_samples = np.random.uniform(*VISCOSITY_RANGE, N_SAMPLES)
rho_samples = np.random.uniform(*DENSITY_RANGE, N_SAMPLES)

# x_eq (cm) such that exp(-alpha_sample * x_eq) = exp(-alpha_nominal * x_ref)
#  => x_eq = x_ref * alpha_nominal / alpha_sample
x_eq_samples_cm = np.array([
    X_REF_CM * alpha_nominal / alpha_exact(E_KPA, FREQ_HZ, eta, rho)
    for eta, rho in zip(eta_samples, rho_samples)
])

mean_ = float(np.mean(x_eq_samples_cm))
sd_ = float(np.std(x_eq_samples_cm, ddof=1))
p5 = float(np.percentile(x_eq_samples_cm, 5))
p95 = float(np.percentile(x_eq_samples_cm, 95))

print(f"\nMC distribution of x_eq (cm), N={N_SAMPLES}:")
print(f"  mean = {mean_:.4f} cm")
print(f"  SD (ddof=1) = {sd_:.4f} cm")
print(f"  5th percentile = {p5:.4f} cm")
print(f"  95th percentile = {p95:.4f} cm")

out_csv = os.path.join(os.path.dirname(__file__), "task5_monte_carlo_samples.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sample_idx", "eta_Pa_s", "rho_kg_m3", "x_eq_cm"])
    for i, (eta, rho, x) in enumerate(zip(eta_samples, rho_samples, x_eq_samples_cm)):
        w.writerow([i, eta, rho, x])

summary_csv = os.path.join(os.path.dirname(__file__), "task5_monte_carlo_summary.csv")
with open(summary_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["class", "freq_hz", "x_ref_cm", "n_samples", "mean_cm", "sd_cm", "p5_cm", "p95_cm"])
    w.writerow([CLASS, FREQ_HZ, X_REF_CM, N_SAMPLES, mean_, sd_, p5, p95])

print(f"\nSaved: {out_csv}")
print(f"Saved: {summary_csv}")

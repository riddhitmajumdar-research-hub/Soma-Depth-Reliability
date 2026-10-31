# SOMA Signal Fidelity Investigation
# Part F4 -- leading-order error bound on the weak-damping approximation
# used everywhere in this codebase's Kelvin-Voigt attenuation model.
#
# DERIVATION (analytical, not a new simulation):
# Kelvin-Voigt complex shear modulus: G* = G(1 + i*x), where x = omega*eta/G
# is the "weak-damping" small parameter (this codebase's own comment calls
# this ratio out directly: attenuation_model_powerlaw.py, "omega*eta << G").
#
# Complex wavenumber: k = omega*sqrt(rho/G*) = (omega/c) * (1+ix)^(-1/2),
# where c = sqrt(G/rho) is the (undamped) shear wave velocity already used
# in physics_model.shear_wave_velocity_m_s().
#
# Binomial-expanding (1+ix)^(-1/2) in x:
#   (1+ix)^(-1/2) = 1 - (i x)/2 - (3/8)x^2 + (5/16) i x^3 + O(x^4)
# The imaginary part (which gives the attenuation coefficient alpha = Im(k)):
#   Im[(1+ix)^(-1/2)] = -x/2 + (5/16) x^3 + O(x^5)
#
# The codebase's weak-damping approximation keeps ONLY the leading (-x/2)
# term:
#   alpha_approx = (omega/c) * (x/2) = omega^2 * eta / (2*G*c)
#     <- this is EXACTLY attenuation_model.attenuation_coefficient()'s formula.
#
# The next-order term the approximation drops is (5/16)x^3, so:
#   alpha_exact  ~= (omega/c) * (x/2 - (5/16)x^3)
#   relative error = (alpha_approx - alpha_exact) / alpha_exact
#                  ~= (5/8) * x^2      [leading order in x]
#
# This is a leading-order BOUND (the neglected O(x^5) term is smaller still
# for x<<1), not the full complex-wavenumber solution -- that stays deferred
# per the protocol.

import numpy as np
import csv

from physics_model import stiffness_classes, shear_wave_velocity_m_s
from attenuation_model import LIVER_VISCOSITY_BY_CLASS

FREQS_HZ = [250, 500]


def weak_damping_param_x(youngs_modulus_kpa, frequency_hz, eta):
    E_pa = youngs_modulus_kpa * 1000
    G_pa = E_pa / 3
    omega = 2 * np.pi * frequency_hz
    return (omega * eta) / G_pa


rows = []
print("Leading-order weak-damping relative error bound: (5/8) * x^2, x = omega*eta/G")
print(f"{'class':<22}{'freq(Hz)':>9}{'x':>12}{'rel_error(%)':>15}")
for name, kpa in stiffness_classes.items():
    eta = LIVER_VISCOSITY_BY_CLASS[name]
    for freq in FREQS_HZ:
        x = weak_damping_param_x(kpa, freq, eta)
        rel_error_pct = (5 / 8) * x ** 2 * 100
        print(f"{name:<22}{freq:>9}{x:>12.5f}{rel_error_pct:>15.6f}")
        rows.append({
            "class": name, "frequency_hz": freq, "eta_pa_s": eta,
            "weak_damping_x": x, "leading_order_relative_error_pct": rel_error_pct,
        })

with open("logs/weak_damping_error_bound.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("\nSaved to logs/weak_damping_error_bound.csv")

max_err = max(r["leading_order_relative_error_pct"] for r in rows)
max_x = max(r["weak_damping_x"] for r in rows)
print(f"\nWorst-case leading-order relative error across all class/frequency combinations "
      f"tested (250Hz/500Hz, all 3 tissue classes): {max_err:.6f}%")
print(f"Worst-case x (omega*eta/G): {max_x:.5f}")
print("\nCOMPLICATES the weak-damping assumption at these frequencies, not confirms it: x is "
      "NOT << 1 here (cirrhosis at 500Hz reaches x=1.39, healthy reaches x=1.05) -- the small-"
      "parameter expansion this bound itself relies on is only valid while x<<1, so a >100% "
      "'leading-order error' at 500Hz doesn't mean a small, quantifiable correction -- it means "
      "the perturbative approximation has broken down entirely at that frequency for those "
      "classes, and the leading-order bound can no longer be trusted as an error estimate there. "
      "At 250Hz the assumption is on firmer ground (x=0.31-0.70, errors 6-30%) but still not "
      "uniformly small across classes. This directly contradicts attenuation_model_powerlaw.py's "
      "framing that anchoring at 20Hz (x~0.042) and letting the models diverge toward 100Hz "
      "'tests the assumption honestly' -- 250Hz/500Hz sit well outside the regime where weak-"
      "damping is even self-consistently defined, let alone small-error. This needs disclosure "
      "wherever 250Hz/500Hz attenuation coefficients are reported alongside the Kelvin-Voigt model.")

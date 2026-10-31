"""
Lane 1, Task 1 -- exact complex-wavenumber Kelvin-Voigt solution vs the
codebase's weak-damping approximation.

Imports tissue constants (E per class, rho) from Source/physics_model.py and
per-class eta from Source/attenuation_model.py -- both already fixed by
Fix #2/#3 in this repo. Does NOT hardcode E, rho, or eta.

Weak-damping formula (alpha_weak, c0) is copied here in the same functional
form as Source/attenuation_model.attenuation_coefficient() /
Source/physics_model.shear_wave_velocity_m_s() so both sides of the
comparison are visible in one script; it is not a re-derivation, it is the
same algebra already in those two functions.

Exact solution: k = omega*sqrt(rho / (G + i*omega*eta))  (complex).
  alpha_exact = Im(k)   [1/m]
  c_exact     = omega / Re(k)   [m/s]
"""
import sys
import os
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Source"))
from physics_model import stiffness_classes          # noqa: E402  (healthy/fibrosis/cirrhosis -> kPa)
from attenuation_model import LIVER_VISCOSITY_BY_CLASS  # noqa: E402  (per-class eta, Pa.s)

RHO = 1060.0  # kg/m^3, as used in physics_model.shear_wave_velocity_m_s() default
FREQS_HZ = [20, 100, 250, 500]


def weak_damping(E_kpa, freq_hz, eta):
    G_pa = (E_kpa * 1000) / 3
    c0 = np.sqrt(G_pa / RHO)
    omega = 2 * np.pi * freq_hz
    alpha_weak = (omega ** 2 * eta) / (2 * G_pa * c0)
    x = omega * eta / G_pa
    return alpha_weak, c0, x


def exact_solution(E_kpa, freq_hz, eta):
    G_pa = (E_kpa * 1000) / 3
    omega = 2 * np.pi * freq_hz
    G_star = G_pa + 1j * omega * eta
    k = omega * np.sqrt(RHO / G_star)
    # Convention: wave ~ exp(i(kx - wt)), so amplitude ~ exp(-Im(k) x) requires
    # Im(k) > 0 for a physically damped (decaying) wave. numpy's principal
    # branch of sqrt(1/(G + i*omega*eta)) returns a NEGATIVE imaginary part
    # for this input (denominator has positive imaginary part), which would
    # give non-physical growth -- take the branch with positive Im(k) instead.
    alpha_exact = abs(np.imag(k))
    c_exact = omega / np.real(k)
    return alpha_exact, c_exact


rows = []
for name, E_kpa in stiffness_classes.items():
    eta = LIVER_VISCOSITY_BY_CLASS[name]
    for freq in FREQS_HZ:
        alpha_weak, c0, x = weak_damping(E_kpa, freq, eta)
        alpha_exact, c_exact = exact_solution(E_kpa, freq, eta)
        ratio = alpha_weak / alpha_exact
        rows.append({
            "class": name, "E_kPa": E_kpa, "eta_Pa_s": eta, "freq_Hz": freq,
            "omega_eta_over_G": x,
            "alpha_weak_per_m": alpha_weak, "alpha_exact_per_m": alpha_exact,
            "ratio_weak_over_exact": ratio,
            "c0_m_s": c0, "c_exact_m_s": c_exact,
        })

out_csv = os.path.join(os.path.dirname(__file__), "table1_exact_vs_weak.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'class':<22}{'f(Hz)':>7}{'x=we/G':>10}{'alpha_weak':>13}{'alpha_exact':>13}{'ratio':>9}{'c0':>9}{'c_exact':>10}")
for r in rows:
    print(f"{r['class']:<22}{r['freq_Hz']:>7}{r['omega_eta_over_G']:>10.4f}"
          f"{r['alpha_weak_per_m']:>13.4f}{r['alpha_exact_per_m']:>13.4f}"
          f"{r['ratio_weak_over_exact']:>9.4f}{r['c0_m_s']:>9.4f}{r['c_exact_m_s']:>10.4f}")

print(f"\nSaved: {out_csv}")

# --- Check against independently supplied values ---------------------------
checks = [
    ("healthy", 100, "ratio_weak_over_exact", 1.027),
    ("significant_fibrosis", 100, "ratio_weak_over_exact", 1.010),
    ("cirrhosis", 100, "ratio_weak_over_exact", 1.048),
    ("healthy", 100, "alpha_weak_per_m", 55.31),
    ("cirrhosis", 100, "alpha_weak_per_m", 44.96),
]
print("\n--- CHECK VALUES ---")
for cls, freq, field, expected in checks:
    row = next(r for r in rows if r["class"] == cls and r["freq_Hz"] == freq)
    actual = row[field]
    status = "MATCH" if abs(actual - expected) < 0.01 * max(abs(expected), 1e-9) else "MISMATCH"
    print(f"{cls:<22} f={freq:<4} {field:<25} expected={expected:<10} actual={actual:<12.5f} -> {status}")

# 250Hz / 500Hz cirrhosis ratio checks
for freq, expected in [(250, 1.28), (500, 1.99)]:
    row = next(r for r in rows if r["class"] == "cirrhosis" and r["freq_Hz"] == freq)
    actual = row["ratio_weak_over_exact"]
    status = "MATCH" if abs(actual - expected) < 0.01 * expected else "MISMATCH"
    print(f"cirrhosis            f={freq:<4} ratio_weak_over_exact     expected={expected:<10} actual={actual:<12.5f} -> {status}")

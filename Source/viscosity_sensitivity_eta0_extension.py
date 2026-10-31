# SOMA Signal Fidelity Investigation
# Part F3 -- extend the existing viscosity sensitivity sweep (Source/
# viscosity_sensitivity.py, currently eta=0.5-1.77 Pa*s) down to eta=0
# (purely elastic anchor point), reusing that script's own
# attenuation_coefficient() / depth_at_threshold() functions unchanged.

import numpy as np
import csv
from viscosity_sensitivity import (
    attenuation_coefficient, depth_at_threshold, HEALTHY_KPA, VISCOSITY_VALUES
)

EXTENDED_VISCOSITY_VALUES = [0.0] + VISCOSITY_VALUES  # anchor purely-elastic point at the bottom

depths_cm = np.linspace(0, 15, 100)
depths_m = depths_cm / 100

rows = []
print("Depth at which signal drops to 10% amplitude, by viscosity (eta=0 anchor added):")
for eta in EXTENDED_VISCOSITY_VALUES:
    alpha = attenuation_coefficient(HEALTHY_KPA, 100, eta)
    if alpha == 0:
        d10_cm = np.inf
        print(f"  eta = {eta:.2f} Pa·s   ->   alpha=0 (purely elastic, no attenuation in this "
              f"model) -> 10% depth = infinite (signal never decays)")
    else:
        d10_m = depth_at_threshold(HEALTHY_KPA, 100, eta, threshold_pct=10)
        d10_cm = d10_m * 100
        print(f"  eta = {eta:.2f} Pa·s   ->   10% depth = {d10_cm:.2f} cm")
    rows.append({"eta_pa_s": eta, "alpha_per_m": alpha, "depth_at_10pct_amplitude_cm": d10_cm})

with open("logs/viscosity_sensitivity_eta0_extension.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["eta_pa_s", "alpha_per_m", "depth_at_10pct_amplitude_cm"])
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved to logs/viscosity_sensitivity_eta0_extension.csv")
print("\nNote: in this model, attenuation coefficient alpha is proportional to eta directly "
      "(alpha = omega^2 * eta / (2*G*c)) -- so eta=0 is a genuine limit case (zero attenuation), "
      "not an extrapolation artifact. The curve from eta=0 up through the sourced literature "
      "range (0.5-1.77 Pa*s) is continuous and monotonic by construction: alpha scales linearly "
      "with eta, so 10%-depth scales as 1/eta, producing a smooth elastic-to-viscoelastic "
      "transition rather than a jump.")

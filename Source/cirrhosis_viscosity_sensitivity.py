# SOMA -- Cirrhosis viscosity sensitivity sweep
# Follow-up to Fix #3 (README Log 18): cirrhosis eta=1.77 Pa.s is direction-
# confirmed by Reiter et al. (p<0.001 elevated vs healthy/fibrosis) but the
# MAGNITUDE is borrowed from a different record's fat-content axis -- flagged
# as an approximation, not a derivation. This sweep checks whether the two
# findings that depend on it (healthy attenuates fastest; the 4.4x->1.2x
# margin) hold across a plausible range, or only at that one borrowed point.
#
# Healthy/fibrosis eta stay FIXED at 0.5 Pa.s -- that value is the real,
# citation-matched Reiter et al. result (p=0.21, indistinguishable), not in
# question here. Only cirrhosis eta is swept.

import numpy as np
import matplotlib.pyplot as plt
from attenuation_model import attenuation_coefficient, LIVER_VISCOSITY_BY_CLASS
from constants import NOMINAL_FREQ_HZ

HEALTHY_KPA = 4.5
FIBROSIS_KPA = 7.5
CIRRHOSIS_KPA = 12.0
FREQ_HZ = NOMINAL_FREQ_HZ  # SOMA's stated operating frequency

CIRRHOSIS_ETA_RANGE = np.linspace(1.2, 2.5, 14)  # Pa.s, plausible bounded range
CIRRHOSIS_ETA_ORIGINAL = LIVER_VISCOSITY_BY_CLASS["cirrhosis"]  # 1.77, the borrowed point

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=NOMINAL_FREQ_HZ,
                    help="carrier frequency in Hz (default: NOMINAL_FREQ_HZ = 100). Absolute alphas scale with f^2; the margin does not.")
    ap.add_argument("--out", default="../outputs/figures_final/fig04_viscosity_margin.png")
    args = ap.parse_args()
    FREQ_HZ = args.freq
    eta_healthy = LIVER_VISCOSITY_BY_CLASS["healthy"]
    eta_fibrosis = LIVER_VISCOSITY_BY_CLASS["significant_fibrosis"]

    alpha_healthy = attenuation_coefficient(HEALTHY_KPA, FREQ_HZ, eta=eta_healthy)
    alpha_fibrosis = attenuation_coefficient(FIBROSIS_KPA, FREQ_HZ, eta=eta_fibrosis)

    alphas_cirrhosis = [attenuation_coefficient(CIRRHOSIS_KPA, FREQ_HZ, eta=e)
                         for e in CIRRHOSIS_ETA_RANGE]
    margins = [alpha_healthy / a for a in alphas_cirrhosis]

    print(f"Fixed: healthy alpha = {alpha_healthy:.4f} /m (eta={eta_healthy})")
    print(f"Fixed: fibrosis alpha = {alpha_fibrosis:.4f} /m (eta={eta_fibrosis})")
    print(f"\nCirrhosis eta swept {CIRRHOSIS_ETA_RANGE[0]:.2f}-{CIRRHOSIS_ETA_RANGE[-1]:.2f} Pa.s:")
    print(f"{'eta (Pa.s)':>12} {'alpha (/m)':>12} {'healthy/cirrhosis margin':>26} {'healthy still fastest?':>24}")
    for e, a, m in zip(CIRRHOSIS_ETA_RANGE, alphas_cirrhosis, margins):
        still_fastest = "YES" if alpha_healthy > a else "NO"
        marker = "  <- original Fix #3 value" if abs(e - CIRRHOSIS_ETA_ORIGINAL) < 0.06 else ""
        print(f"{e:12.2f} {a:12.4f} {m:26.3f} {still_fastest:>24}{marker}")

    min_margin, max_margin = min(margins), max(margins)
    all_healthy_fastest = all(alpha_healthy > a for a in alphas_cirrhosis)

    print(f"\nMargin range across full swept range: {min_margin:.3f}x - {max_margin:.3f}x")
    print(f"'Healthy attenuates fastest' holds across ENTIRE range: {all_healthy_fastest}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    ax.plot(CIRRHOSIS_ETA_RANGE, alphas_cirrhosis, color="#0F6E56", linewidth=2,
             label="Cirrhosis alpha (swept)")
    ax.axhline(alpha_healthy, color="#C0392B", linestyle="--", linewidth=1.3,
                label=f"Healthy alpha = {alpha_healthy:.2f}/m (fixed)")
    ax.axhline(alpha_fibrosis, color="#BA7517", linestyle="--", linewidth=1.3,
                label=f"Fibrosis alpha = {alpha_fibrosis:.2f}/m (fixed)")
    ax.axvline(CIRRHOSIS_ETA_ORIGINAL, color="gray", linestyle=":", linewidth=1.3,
                label=f"Value used in the model ({CIRRHOSIS_ETA_ORIGINAL} Pa\u00b7s)")
    ax.set_xlabel("Cirrhosis viscosity, eta (Pa\u00b7s)")
    ax.set_ylabel(f"Attenuation coefficient, alpha (1/m) at {FREQ_HZ:g} Hz")
    ax.set_title(f"Cirrhosis attenuation vs. viscosity ({FREQ_HZ:g} Hz)")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(CIRRHOSIS_ETA_RANGE, margins, color="#34495E", linewidth=2)
    ax2.axvline(CIRRHOSIS_ETA_ORIGINAL, color="gray", linestyle=":", linewidth=1.3)
    ax2.axhline(1.0, color="#C0392B", linestyle="--", linewidth=1,
                 label="Margin = 1 (healthy/cirrhosis attenuate equally)")
    # annotate the two numbers the record quotes, computed from the model (alpha is linear in eta)
    alpha_c_orig = attenuation_coefficient(CIRRHOSIS_KPA, FREQ_HZ, eta=CIRRHOSIS_ETA_ORIGINAL)
    margin_orig = alpha_healthy / alpha_c_orig
    eta_flip = CIRRHOSIS_ETA_ORIGINAL * alpha_healthy / alpha_c_orig
    ax2.plot([CIRRHOSIS_ETA_ORIGINAL], [margin_orig], "o", color="#34495E")
    ax2.annotate(f"{margin_orig:.2f}x at {CIRRHOSIS_ETA_ORIGINAL} Pa\u00b7s", (CIRRHOSIS_ETA_ORIGINAL, margin_orig),
                 xytext=(CIRRHOSIS_ETA_ORIGINAL + 0.08, margin_orig + 0.12), fontsize=9)
    ax2.plot([eta_flip], [1.0], "o", color="#C0392B")
    ax2.annotate(f"margin = 1.0 at {eta_flip:.2f} Pa\u00b7s", (eta_flip, 1.0),
                 xytext=(eta_flip - 0.45, 1.08), fontsize=9, color="#C0392B")
    print(f"margin at {CIRRHOSIS_ETA_ORIGINAL} Pa.s = {margin_orig:.4f}; margin = 1 at eta = {eta_flip:.4f} Pa.s")
    fig.suptitle(f"Carrier frequency {FREQ_HZ:g} Hz; healthy and fibrosis viscosity 0.5 Pa\u00b7s fixed", fontsize=10, y=1.0)
    ax2.set_xlabel("Cirrhosis viscosity, eta (Pa\u00b7s)")
    ax2.set_ylabel("Margin (healthy alpha / cirrhosis alpha)")
    ax2.set_title("Healthy-over-cirrhosis attenuation margin (frequency-independent)")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to {out_path}")

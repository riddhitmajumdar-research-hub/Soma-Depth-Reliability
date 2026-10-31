# Power-Law Attenuation Sensitivity -- three-panel publication figure
# File: power_law_figure.py

import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.insert(0, "Source")

from physics_model import stiffness_classes
from attenuation_model import attenuation_coefficient as alpha_kv
from attenuation_model_powerlaw import attenuation_coefficient_powerlaw as alpha_pl, ANCHOR_FREQ_HZ, POWER_LAW_EXPONENT
from constants import SEED121_CERTIFIED_FLOOR_CM

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

COLOR_KV = "#2E5EAA"
COLOR_PL = "#D95F02"
COLOR_CHANCE = "#999999"

# --- Load certified KV data (Log 15/13 original certification) ---
with open("outputs/stage4/data/stage4_degradation_aware_results.pkl", "rb") as f:
    kv_results = pickle.load(f)

# --- Load certified power-law data (just generated) ---
pl_results = []
with open("outputs/stage4/data/power_law_sensitivity_certified.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        pl_results.append({
            "depth_cm": float(row["depth_cm"]),
            "acc_real": float(row["acc_real"]),
            "control_mean": float(row["control_mean"]),
            "fdr_significant": row["fdr_significant"] == "True",
        })

fig = plt.figure(figsize=(15, 5.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.3, 1], wspace=0.32)

# ============================================================
# Panel A: log-log frequency-scaling divergence
# ============================================================
ax_a = fig.add_subplot(gs[0])
freqs = np.logspace(np.log10(10), np.log10(500), 100)
kpa_healthy = stiffness_classes["healthy"]

a_kv_curve = [alpha_kv(kpa_healthy, f) for f in freqs]
a_pl_curve = [alpha_pl(kpa_healthy, f) for f in freqs]

ax_a.loglog(freqs, a_kv_curve, color=COLOR_KV, linewidth=2.2, label=r"Kelvin-Voigt ($\alpha \propto \omega^2$)")
ax_a.loglog(freqs, a_pl_curve, color=COLOR_PL, linewidth=2.2, label=f"Power-law (n={POWER_LAW_EXPONENT})")
ax_a.axvline(ANCHOR_FREQ_HZ, color="#666666", linestyle=":", linewidth=1.2)
ax_a.axvline(100, color="#111111", linestyle="--", linewidth=1.2)
ax_a.scatter([ANCHOR_FREQ_HZ], [alpha_kv(kpa_healthy, ANCHOR_FREQ_HZ)], color="black", zorder=5, s=40, marker="o")
ax_a.text(ANCHOR_FREQ_HZ * 1.15, alpha_kv(kpa_healthy, ANCHOR_FREQ_HZ) * 0.5, "20Hz anchor\n(models agree)",
          fontsize=8, color="#444444")
ax_a.text(100 * 1.1, max(a_kv_curve) * 0.15, "100Hz\n(SOMA operating\nfrequency)", fontsize=8, color="#111111")

ax_a.set_xlabel("Frequency (Hz)", fontsize=10.5, fontweight="medium")
ax_a.set_ylabel(r"Attenuation coefficient $\alpha$ (1/m)", fontsize=10.5, fontweight="medium")
ax_a.set_title("A.  Frequency-Scaling Divergence\n(healthy tissue)", fontsize=11.5, fontweight="bold", loc="left")
ax_a.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.9)

# ============================================================
# Panel B: certified accuracy vs depth, both models, significance markers
# ============================================================
ax_b = fig.add_subplot(gs[1])

kv_depths = [r["depth_cm"] for r in kv_results]
kv_acc = [r["acc_real"] for r in kv_results]
kv_sig = [bool(r["fdr_significant"]) for r in kv_results]

pl_depths = [r["depth_cm"] for r in pl_results]
pl_acc = [r["acc_real"] for r in pl_results]
pl_sig = [r["fdr_significant"] for r in pl_results]

ax_b.axhspan(0.30, 0.37, color=COLOR_CHANCE, alpha=0.18, label="Chance level (~1/3)")

ax_b.plot(kv_depths, kv_acc, color=COLOR_KV, linewidth=1.8, zorder=2, label="Kelvin-Voigt (certified)")
ax_b.plot(pl_depths, pl_acc, color=COLOR_PL, linewidth=1.8, zorder=2, label=f"Power-law n={POWER_LAW_EXPONENT} (certified)")

for d, a, sig in zip(kv_depths, kv_acc, kv_sig):
    ax_b.scatter([d], [a], s=70, facecolor=COLOR_KV if sig else "white",
                 edgecolor=COLOR_KV, linewidth=1.6, zorder=3)
for d, a, sig in zip(pl_depths, pl_acc, pl_sig):
    ax_b.scatter([d], [a], s=70, facecolor=COLOR_PL if sig else "white",
                 edgecolor=COLOR_PL, linewidth=1.6, zorder=3, marker="D")

ax_b.axvline(SEED121_CERTIFIED_FLOOR_CM, color=COLOR_KV, linestyle=":", linewidth=1.3, alpha=0.7)
ax_b.text(SEED121_CERTIFIED_FLOOR_CM, 1.02, f"{SEED121_CERTIFIED_FLOOR_CM:.2f}cm\nKV floor", fontsize=8, ha="center", va="bottom", color=COLOR_KV, transform=ax_b.get_xaxis_transform())

ax_b.set_xlabel("Depth (cm)", fontsize=10.5, fontweight="medium")
ax_b.set_ylabel("Certified accuracy", fontsize=10.5, fontweight="medium")
ax_b.set_title("B.  Certified Accuracy Across Depth\n(filled = FDR-significant, hollow = not)", fontsize=11.5, fontweight="bold", loc="left")
ax_b.set_ylim(0.2, 1.05)
ax_b.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9)

# ============================================================
# Panel C: signal amplitude survival at 100Hz, both models
# ============================================================
ax_c = fig.add_subplot(gs[2])
depths_m = np.linspace(0, 0.15, 200)
amp_kv = np.exp(-alpha_kv(kpa_healthy, 100) * depths_m)
amp_pl = np.exp(-alpha_pl(kpa_healthy, 100) * depths_m)

ax_c.plot(depths_m * 100, amp_kv, color=COLOR_KV, linewidth=2.2, label="Kelvin-Voigt")
ax_c.plot(depths_m * 100, amp_pl, color=COLOR_PL, linewidth=2.2, label=f"Power-law n={POWER_LAW_EXPONENT}")
ax_c.axhline(0.1, color="#666666", linestyle=":", linewidth=1.1)
ax_c.text(0.3, 0.115, "10% amplitude", fontsize=8, color="#555555")
ax_c.set_xlabel("Depth (cm)", fontsize=10.5, fontweight="medium")
ax_c.set_ylabel("Fraction of amplitude surviving", fontsize=10.5, fontweight="medium")
ax_c.set_title("C.  Signal Survival at 100Hz\n(healthy tissue)", fontsize=11.5, fontweight="bold", loc="left")
ax_c.legend(loc="upper right", fontsize=8.5, frameon=True, framealpha=0.9)
ax_c.set_ylim(-0.02, 1.05)

fig.suptitle("SOMA — Power-Law Attenuation Sensitivity Analysis (Kelvin-Voigt vs. Liver Shear-Wave Literature, n=0.51)",
             fontsize=13.5, fontweight="bold", y=1.06)
fig.text(0.5, -0.06,
         "Both models are pinned to agree exactly at the 20Hz anchor, where the Kelvin-Voigt weak-damping assumption holds most comfortably;\n"
         "Panels A and C are analytically derived, Panel B uses independently certified 200-shuffle BH-FDR results for each model.",
         ha="center", fontsize=8.3, color="#555555")

plt.savefig("outputs/stage4/figures/power_law_sensitivity.png", dpi=200, bbox_inches="tight", pad_inches=0.35)
print("Saved outputs/stage4/figures/power_law_sensitivity.png")

# Stage 4 continuation -- Three-way floor comparison
# File: Source/plot_degradation_aware.py

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("results/stage4_results.pkl", "rb") as f:
    baseline_results = pickle.load(f)
with open("results/stage4_cnn_results.pkl", "rb") as f:
    cnn_clean_results = pickle.load(f)
with open("results/stage4_degradation_aware_results.pkl", "rb") as f:
    cnn_aware_results = pickle.load(f)

baseline_100 = sorted([r for r in baseline_results if r["freq"] == 100], key=lambda r: r["depth_cm"])
cnn_clean_100 = sorted([r for r in cnn_clean_results if r["freq"] == 100], key=lambda r: r["depth_cm"])
cnn_aware_100 = sorted(cnn_aware_results, key=lambda r: r["depth_cm"])

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot([r["depth_cm"] for r in baseline_100], [r["acc_real"] for r in baseline_100],
        marker="o", color="#993C1D", label="Baseline (hand-picked features)", linewidth=2)
ax.plot([r["depth_cm"] for r in cnn_clean_100], [r["acc_real"] for r in cnn_clean_100],
        marker="o", color="#BA7517", label="CNN (clean-trained only)", linewidth=2)
ax.plot([r["depth_cm"] for r in cnn_aware_100], [r["acc_real"] for r in cnn_aware_100],
        marker="o", color="#0F6E56", label="CNN (degradation-aware training)", linewidth=2)

ax.axvline(0.52, color="#993C1D", linestyle=":", alpha=0.5)
ax.axvline(5.45, color="#BA7517", linestyle=":", alpha=0.5)
ax.axvline(8.18, color="#0F6E56", linestyle=":", alpha=0.5)
ax.axhline(y=1/3, color="gray", linestyle="--", linewidth=1, label="chance (33.3%)")

ax.axvspan(8, 10, color="gray", alpha=0.1, label="typical organ depth range")

ax.set_xlabel("Depth (cm)")
ax.set_ylabel("Classification accuracy")
ax.set_title("Stage 4 -- Full progression at 100Hz: 0.52cm to 5.45cm to 8.18cm")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("stage4_full_progression.png", dpi=200, bbox_inches="tight")
plt.show()

print("Floor progression at 100Hz:")
print(f"  Hand-picked baseline (clean-trained): 0.52cm")
print(f"  CNN (clean-trained):                  5.45cm")
print(f"  CNN (degradation-aware training):     8.18cm")
print(f"  Total improvement factor:              {8.18/0.52:.1f}x")

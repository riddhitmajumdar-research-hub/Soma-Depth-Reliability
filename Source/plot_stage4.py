# Stage 4 -- Final result visualization
# File: Source/plot_stage4.py

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("results/stage4_results.pkl", "rb") as f:
    baseline_results = pickle.load(f)
with open("results/stage4_cnn_results.pkl", "rb") as f:
    cnn_results = pickle.load(f)

colors = {20: "#0F6E56", 100: "#BA7517", 250: "#993C1D", 500: "#5B2C6F"}

fig, ax = plt.subplots(figsize=(11, 7))
for freq in [20, 100, 250, 500]:
    freq_data = sorted([r for r in cnn_results if r["freq"] == freq], key=lambda r: r["depth_cm"])
    depths = [r["depth_cm"] for r in freq_data]
    accs = [r["acc_real"] for r in freq_data]
    ax.plot(depths, accs, marker="o", color=colors[freq], label=f"{freq}Hz", linewidth=2)

    floor_depth = None
    for r in freq_data:
        if not r["fdr_significant"]:
            floor_depth = r["depth_cm"]
            break
    if floor_depth is not None:
        ax.axvline(floor_depth, color=colors[freq], linestyle=":", alpha=0.5, linewidth=1)

ax.axhline(y=1/3, color="gray", linestyle="--", linewidth=1, label="chance (33.3%)")
ax.set_xlabel("Depth (cm)")
ax.set_ylabel("Classification accuracy")
ax.set_title("Stage 4 -- CNN classification accuracy vs. depth, by frequency\n"
              "(dotted lines = certified floor, FDR-corrected)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/stage4_cnn_accuracy_vs_depth.png", dpi=200, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(11, 7))

baseline_100 = sorted([r for r in baseline_results if r["freq"] == 100], key=lambda r: r["depth_cm"])
cnn_100 = sorted([r for r in cnn_results if r["freq"] == 100], key=lambda r: r["depth_cm"])

ax.plot([r["depth_cm"] for r in baseline_100], [r["acc_real"] for r in baseline_100],
        marker="o", color="#993C1D", label="Baseline (hand-picked features)", linewidth=2)
ax.plot([r["depth_cm"] for r in cnn_100], [r["acc_real"] for r in cnn_100],
        marker="o", color="#0F6E56", label="CNN (raw signal)", linewidth=2)

ax.axvline(0.52, color="#993C1D", linestyle=":", alpha=0.6, label="Baseline floor: 0.52cm")
ax.axvline(5.45, color="#0F6E56", linestyle=":", alpha=0.6, label="CNN floor: 5.45cm")
ax.axhline(y=1/3, color="gray", linestyle="--", linewidth=1, label="chance (33.3%)")

ax.set_xlabel("Depth (cm)")
ax.set_ylabel("Classification accuracy")
ax.set_title("Stage 4 -- Baseline vs. CNN at 100Hz: the raw-signal robustness advantage")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/stage4_baseline_vs_cnn.png", dpi=200, bbox_inches="tight")
plt.show()

print("Both plots saved.")
print(f"\nBaseline floor at 100Hz: 0.52cm")
print(f"CNN floor at 100Hz:      5.45cm")
print(f"Improvement factor:      {5.45/0.52:.1f}x")

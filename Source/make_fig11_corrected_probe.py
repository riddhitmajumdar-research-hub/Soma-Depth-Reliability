# Fig. 11 redrawn with the corrected 27.05dB frozen-probe results.
# Style follows section 3.6 of Source/make_figures.py (6x5 in, 175 dpi, same colors).
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path as _P; ROOT = str(_P(__file__).resolve().parents[1])
OUT = f"{ROOT}/outputs/figures_earlier"
DPI = 175


def row(depth):
    with open(f"{ROOT}/Source/logs/frozen_embedding_linear_probe_27p05dB_{depth}cm.csv") as f:
        return next(csv.DictReader(f))


a, b = row("8p18"), row("8p0")
assert float(a["deployed_model_accuracy"]) == float(b["deployed_model_accuracy"])
labels = ["Deployed model\n(end-to-end)", "Frozen-embedding\nprobe, 8.18cm", "Frozen-embedding\nprobe, 8.00cm", "Chance level"]
vals = [float(a["deployed_model_accuracy"]), float(a["frozen_probe_accuracy"]),
        float(b["frozen_probe_accuracy"]), float(a["chance_level"])]
colors = ["#C0392B", "#27AE60", "#27AE60", "#95A5A6"]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, vals, color=colors)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.1)
ax.set_title("Frozen-probe vs. deployed accuracy (SNR 27.05dB)")
ax.tick_params(axis="x", labelsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_06_frozen_probe_vs_deployed_27p05dB.png", dpi=DPI)

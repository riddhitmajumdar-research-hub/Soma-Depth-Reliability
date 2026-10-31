import csv
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path as _P; ROOT = str(_P(__file__).resolve().parents[1])
OUT = f"{ROOT}/outputs/figures_earlier"
DPI = 175


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------
# 3.8 per-class recall vs depth, 6.50cm gate marked
# ---------------------------------------------------------------
rows = read_csv(f"{ROOT}/Source/logs/degradation_aware_per_class_recall_by_depth.csv")
depth = np.array([float(r["depth_cm"]) for r in rows])
order = np.argsort(depth)
depth = depth[order]

def col(name):
    return np.array([float(r[name]) for r in rows])[order]

fig, ax = plt.subplots(figsize=(7.5, 5))
classes = [
    ("cirrhosis_recall", "Cirrhosis", "#C0392B"),
    ("significant_fibrosis_recall", "Significant fibrosis", "#E67E22"),
    ("healthy_recall", "Healthy", "#2471A3"),
]
for prefix, label, color in classes:
    mean = col(f"{prefix}_mean")
    lo = col(f"{prefix}_ci_lo")
    hi = col(f"{prefix}_ci_hi")
    ax.plot(depth, mean, marker="o", color=color, label=label)
    ax.fill_between(depth, lo, hi, color=color, alpha=0.15)

ax.axvline(6.50, color="black", linestyle="--", linewidth=1.5)
ax.text(6.50, ax.get_ylim()[1] if False else 1.02, "6.50cm gate", rotation=90,
        va="bottom", ha="right", fontsize=9, transform=ax.get_xaxis_transform())
ax.set_xlabel("Depth (cm)")
ax.set_ylabel("Recall")
ax.set_title("Per-class recall vs. depth, with the 6.50cm certified-safety gate")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_08_per_class_recall_vs_depth.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------
# 3.9 24-seed floor distribution histogram
# ---------------------------------------------------------------
rows = read_csv(f"{ROOT}/Source/logs/degradation_aware_final_results.csv")
floors = np.array([float(r["naive_floor_cm"]) for r in rows])
mean, sd = floors.mean(), floors.std(ddof=1)

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(floors, bins=8, color="#5B8DB8", edgecolor="white")
ax.axvline(mean, color="black", linewidth=2, label=f"mean = {mean:.2f}cm")
ax.axvline(mean - sd, color="black", linestyle=":", linewidth=1.3)
ax.axvline(mean + sd, color="black", linestyle=":", linewidth=1.3,
           label=f"±1 SD = {sd:.2f}cm")
ax.set_xlabel("Naive floor depth (cm)")
ax.set_ylabel("Count of seeds (n=24)")
ax.set_title("Distribution of naive floor depth across 24 seeds")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_09_24seed_floor_histogram.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------
# 3.6 frozen-probe vs deployed-model accuracy at 8.18cm
# ---------------------------------------------------------------
rows = read_csv(f"{ROOT}/Source/logs/frozen_embedding_linear_probe_8p18cm.csv")
r = rows[0]
labels = ["Deployed model\n(end-to-end)", "Frozen-embedding\nlinear probe", "Chance level"]
vals = [float(r["deployed_model_accuracy"]), float(r["frozen_probe_accuracy"]), float(r["chance_level"])]
colors = ["#C0392B", "#27AE60", "#95A5A6"]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v*100:.1f}%", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.1)
ax.set_title(f"Frozen-probe vs. deployed accuracy at {r['depth_cm']}cm (SNR {r['snr_db']}dB)")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_06_frozen_probe_vs_deployed.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------
# 3.5 Grad-CAM occlusion causal test: probability drop, 3 conditions
# ---------------------------------------------------------------
rows = read_csv(f"{ROOT}/Source/logs/gradcam_counterfactual_occlusion.csv")
zero_drop = np.array([float(r["prob_drop_zero_first20"]) for r in rows])
rand_drop = np.array([float(r["prob_drop_randomize_first20"]) for r in rows])
ctrl_drop = np.array([float(r["prob_drop_negative_control"]) for r in rows])

fig, ax = plt.subplots(figsize=(6.5, 5))
data = [zero_drop, rand_drop, ctrl_drop]
labels = ["Zero out\nGrad-CAM region", "Randomize\nGrad-CAM region", "Negative control\n(random region)"]
bp = ax.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
for patch, color in zip(bp["boxes"], ["#C0392B", "#E67E22", "#95A5A6"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Drop in true-class probability")
ax.set_title("Grad-CAM occlusion causal test: probability drop by condition")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_05_gradcam_occlusion_causal_test.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------
# 3.4 paired-trajectory, corrected/grouped (baseline -> CNN -> ResNet1D per fold)
# ---------------------------------------------------------------
rows = read_csv(f"{ROOT}/Source/logs/log05_log11_grouped_cv_verification.csv")
baseline_rows = [r for r in rows if r["item"] == "log05_baseline_grouped_cv" and r["fold"].isdigit()]
paired_rows = [r for r in rows if r["item"] == "log11_paired_grouped_cv"]

fig, ax = plt.subplots(figsize=(7, 5))
methods = ["Baseline", "CNN", "ResNet1D"]
x = np.arange(3)

for r in paired_rows:
    vals = [float(r["baseline_acc"]), float(r["cnn_acc"]), float(r["resnet1d_acc"])]
    ax.plot(x, vals, marker="o", alpha=0.6, linewidth=1.3, color="#5B8DB8")

if baseline_rows:
    bl_mean = np.mean([float(r["baseline_acc"]) for r in baseline_rows])
    ax.scatter([0], [bl_mean], color="black", zorder=5, marker="D",
               label=f"log05 baseline-only grouped-CV mean ({len(baseline_rows)} folds)")

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Accuracy")
ax.set_title("Paired trajectory across folds (grouped/leakage-corrected split)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_04_paired_trajectory_corrected.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------
# Discussion/Trap1: original vs leakage-corrected vs repeated-CV (CNN, ResNet1D)
# baseline omitted: no matching "original single-split accuracy" field found in the data
# ---------------------------------------------------------------
leak = {r["model"]: r for r in read_csv(f"{ROOT}/Source/logs/grouped_split_leakage_fix_results.csv")}
rep_raw = read_csv(f"{ROOT}/Source/logs/log11_repeated_grouped_cv_raw.csv")
cnn_rep = np.array([float(r["cnn_acc"]) for r in rep_raw])
resnet_rep = np.array([float(r["resnet1d_acc"]) for r in rep_raw])

models = ["CNN", "ResNet1D"]
original = [float(leak["cnn_stage3c"]["original_accuracy"]), float(leak["resnet1d_stage3e"]["original_accuracy"])]
corrected = [float(leak["cnn_stage3c"]["corrected_accuracy"]), float(leak["resnet1d_stage3e"]["corrected_accuracy"])]
repeated_mean = [cnn_rep.mean(), resnet_rep.mean()]
repeated_sd = [cnn_rep.std(ddof=1), resnet_rep.std(ddof=1)]

x = np.arange(len(models))
w = 0.25
fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.bar(x - w, original, width=w, label="Original (single split)", color="#C0392B")
ax.bar(x, corrected, width=w, label="Leakage-corrected (single grouped split)", color="#E67E22")
ax.bar(x + w, repeated_mean, width=w, yerr=repeated_sd, capsize=4,
       label="Repeated grouped CV (15 reps x 5 folds, mean ± SD)", color="#2471A3")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Original vs. leakage-corrected vs. repeated-CV accuracy")
ax.legend(fontsize=8, loc="lower right")
ax.text(0.5, -0.18,
        "Note: repeated-CV reps reuse the same 90 realizations under different fold splits\n"
        "and are not independent draws — SD understates true between-sample uncertainty.",
        transform=ax.transAxes, ha="center", fontsize=8, style="italic")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_trap1_original_vs_corrected_vs_repeatedcv.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------
# 3.6 accuracy-vs-depth-by-frequency (4 lines: 20/100/250/500Hz)
# certified-floor sweep data, corrected 27.05dB noise model
# ---------------------------------------------------------------
with open(f"{ROOT}/Source/results/stage4_cnn_results.pkl", "rb") as f:
    results = pickle.load(f)

freqs = sorted(set(r["freq"] for r in results))
colors = {20: "#8E44AD", 100: "#2471A3", 250: "#1E8449", 500: "#C0392B"}

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for freq in freqs:
    rows = sorted([r for r in results if r["freq"] == freq], key=lambda r: r["depth_cm"])
    depths = [r["depth_cm"] for r in rows]
    accs = [r["acc_real"] for r in rows]
    ax.plot(depths, accs, marker="o", markersize=3.5, label=f"{freq}Hz", color=colors.get(freq, "black"))

ax.axhline(1 / 3, color="gray", linestyle="--", linewidth=1, label="chance (33.3%)")
ax.set_xlabel("Depth (cm)")
ax.set_ylabel("Real-model accuracy")
ax.set_title("Certified-floor sweep: accuracy vs. depth across frequency (corrected 27.05dB noise)")
ax.set_ylim(0, 1.02)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_06_accuracy_vs_depth_by_frequency.png", dpi=DPI)
plt.close(fig)

# ---------------------------------------------------------------
# 3.2 viscosity-sensitivity sweep, extended to eta=0 anchor
# ---------------------------------------------------------------
rows = read_csv(f"{ROOT}/Source/logs/viscosity_sensitivity_eta0_extension.csv")
eta = np.array([float(r["eta_pa_s"]) for r in rows])
depth = [r["depth_at_10pct_amplitude_cm"] for r in rows]

finite_eta, finite_depth = [], []
inf_eta = None
for e, d in zip(eta, depth):
    if d == "inf":
        inf_eta = e
    else:
        finite_eta.append(e)
        finite_depth.append(float(d))

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(finite_eta, finite_depth, marker="o", color="#2471A3", linewidth=2,
        label="10%-amplitude depth")
if inf_eta is not None:
    y_top = max(finite_depth) * 1.35
    ax.annotate("eta = 0: no attenuation\n(never reaches 10%, depth -> infinity)",
                xy=(inf_eta, y_top * 0.92), xytext=(inf_eta + 0.15, y_top),
                fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="->", color="black", lw=1))
    ax.scatter([inf_eta], [y_top * 0.92], marker="^", s=70, color="#C0392B", zorder=5)
    ax.set_ylim(0, y_top * 1.1)

ax.set_xlabel("Viscosity, eta (Pa·s)")
ax.set_ylabel("Depth at 10% amplitude survival (cm)")
ax.set_title("Viscosity-sensitivity sweep, extended to the eta=0 anchor")
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_02_viscosity_sensitivity_sweep.png", dpi=DPI)
plt.close(fig)

print("done")

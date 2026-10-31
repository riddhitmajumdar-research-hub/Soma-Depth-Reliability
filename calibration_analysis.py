# SOMA Calibration Analysis -- publication-quality reliability diagram + depth-structured gap plot
# File: calibration_analysis.py
# Reads stage4_calibration_data.csv (produced by Source/degradation_aware_cnn.py)

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
import os
import glob

project_files = glob.glob("**/constants.py", recursive=True) + glob.glob("../**/constants.py", recursive=True)
if project_files:
    sys.path.append(os.path.dirname(os.path.abspath(project_files[0])))
from constants import MAX_VALID_DEPTH_CM

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/stage4/data/stage4_calibration_data.csv"
PLOT_PATH = sys.argv[2] if len(sys.argv) > 2 else "outputs/figures_final/fig10_reliability_diagram.png"
N_BINS = 10

# Panel B's reference-floor annotation depends on which calibration CSV was
# loaded -- each one reflects a different training run under different noise
# physics, so the "certified floor" line to draw is not a fixed number.
# Pass it explicitly as a 3rd CLI argument to override; otherwise inferred
# from the CSV filename.
if len(sys.argv) > 3:
    REFERENCE_FLOOR_CM = float(sys.argv[3])
elif "corrected" in os.path.basename(CSV_PATH):
    # Log 20: seed-42, corrected-noise, first-failure depth (degradation_aware_calibration_corrected.py's data)
    REFERENCE_FLOOR_CM = 9.55
else:
    # Log 15: seed-42, original pre-correction certified floor (degradation_aware_cnn_calibration_retrain.py's data)
    REFERENCE_FLOOR_CM = 8.18

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

COLOR_ACC = "#2E5EAA"
COLOR_CONF = "#D95F02"
COLOR_GAP_UNDER = "#4C72B0"
COLOR_GAP_OVER = "#C44E52"
COLOR_FLOOR = "#666666"

rows = list(csv.DictReader(open(CSV_PATH)))
prob_cols = [k for k in rows[0].keys() if k.startswith("prob_")]

depths, confidences, corrects = [], [], []
for r in rows:
    probs = {c: float(r[c]) for c in prob_cols}
    conf = max(probs.values())
    correct = 1 if r["pred_label"] == r["true_label"] else 0
    depths.append(float(r["depth_cm"]))
    confidences.append(conf)
    corrects.append(correct)

depths = np.array(depths)
confidences = np.array(confidences)
corrects = np.array(corrects)
n = len(confidences)


def fixed_width_bins(confidences, corrects, n_bins):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bin_edges) - 1, 0, n_bins - 1)
    stats = []
    for b in range(n_bins):
        mask = bin_idx == b
        stats.append({
            "lo": bin_edges[b], "hi": bin_edges[b + 1],
            "n": int(mask.sum()),
            "avg_conf": confidences[mask].mean() if mask.sum() else np.nan,
            "accuracy": corrects[mask].mean() if mask.sum() else np.nan,
        })
    return stats


def equal_mass_bins(confidences, corrects, n_bins):
    order = np.argsort(confidences)
    conf_sorted, correct_sorted = confidences[order], corrects[order]
    splits = np.array_split(np.arange(len(confidences)), n_bins)
    stats = []
    for idx in splits:
        if len(idx) == 0:
            continue
        c, y = conf_sorted[idx], correct_sorted[idx]
        stats.append({"n": len(idx), "avg_conf": c.mean(), "accuracy": y.mean()})
    return stats


def compute_uce(stats):
    gaps = [abs(s["avg_conf"] - s["accuracy"]) for s in stats if s["n"] > 0]
    return np.mean(gaps) if gaps else np.nan


def compute_ece(stats, n_total):
    terms = [(s["n"] / n_total) * abs(s["avg_conf"] - s["accuracy"]) for s in stats if s["n"] > 0]
    return sum(terms)


fixed_stats = fixed_width_bins(confidences, corrects, N_BINS)
equal_stats = equal_mass_bins(confidences, corrects, N_BINS)
uce_fixed = compute_uce(fixed_stats)
uce_equal = compute_uce(equal_stats)
ece_fixed = compute_ece(fixed_stats, n)

print("=" * 70)
print(f"CALIBRATION ANALYSIS -- {n} samples from {CSV_PATH}")
print("=" * 70)
print(f"\nOverall accuracy: {corrects.mean():.4f}")
print(f"Binned calibration gap (fixed-width, {N_BINS} bins): {uce_fixed:.4f}")
print(f"Binned calibration gap (equal-mass,  {N_BINS} bins): {uce_equal:.4f}")
print(f"Size-weighted binned gap (fixed-width; not used in the record): {ece_fixed:.4f}\n")
for s in fixed_stats:
    if s["n"] == 0:
        continue
    gap = s["avg_conf"] - s["accuracy"]
    print(f"  [{s['lo']:.2f},{s['hi']:.2f})  n={s['n']:4d}  conf={s['avg_conf']:.4f}  acc={s['accuracy']:.4f}  gap={gap:+.4f}")

# Per-depth stats for panel 2
unique_depths = sorted(set(depths))
depth_stats = []
for d in unique_depths:
    mask = depths == d
    depth_stats.append({
        "depth": d,
        "avg_conf": confidences[mask].mean(),
        "accuracy": corrects[mask].mean(),
    })

# ============================================================
# FIGURE: 2 columns x 2 rows
#   Top-left:    reliability diagram (bars)
#   Bottom-left: confidence histogram (shared x-axis)
#   Right (full height): depth-structured confidence vs accuracy
# ============================================================
fig = plt.figure(figsize=(13, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1.3], height_ratios=[3, 1], hspace=0.08, wspace=0.28)

# --- Panel A: reliability diagram ---
ax_rel = fig.add_subplot(gs[0, 0])
bin_width = 1.0 / N_BINS
bin_centers = np.array([(s["lo"] + s["hi"]) / 2 for s in fixed_stats])
accs = np.array([s["accuracy"] if s["n"] > 0 else 0 for s in fixed_stats])
confs = np.array([s["avg_conf"] if s["n"] > 0 else np.nan for s in fixed_stats])
ns = np.array([s["n"] for s in fixed_stats])

ax_rel.plot([0, 1], [0, 1], linestyle="--", color=COLOR_FLOOR, linewidth=1.2, label="Perfect calibration", zorder=1)
bars = ax_rel.bar(bin_centers, accs, width=bin_width * 0.92, color=COLOR_ACC, alpha=0.85,
                   edgecolor="white", linewidth=0.8, label="Observed accuracy", zorder=2)
# Overlay the confidence-gap as a hatched outline on top of each bar, standard convention
gap_top = np.where(confs > accs, confs, accs)
gap_bottom = np.where(confs > accs, accs, confs)
for i, (c, a, w) in enumerate(zip(confs, accs, ns)):
    if w == 0 or np.isnan(c):
        continue
    ax_rel.bar(bin_centers[i], gap_top[i] - gap_bottom[i], bottom=gap_bottom[i],
               width=bin_width * 0.92, color=COLOR_GAP_OVER if c < a else COLOR_GAP_UNDER,
               alpha=0.35, hatch="//", edgecolor="none", zorder=3)

ax_rel.scatter(confs[ns > 0], confs[ns > 0], marker="D", s=28, color="black", zorder=4, label="Mean confidence")
ax_rel.set_xlim(0.25, 1.0)
ax_rel.set_ylim(0, 1.02)
ax_rel.set_ylabel("Accuracy", fontsize=11, fontweight="medium")
ax_rel.set_title(f"A.  Reliability Diagram (n={n})", fontsize=12, fontweight="bold", loc="left")
ax_rel.legend(loc="upper left", fontsize=8.5, frameon=True, framealpha=0.9)
ax_rel.text(0.97, 0.05,
            f"Binned calibration gap\nfixed-width bins = {uce_fixed:.3f}\nequal-mass bins = {uce_equal:.3f}",
            transform=ax_rel.transAxes, fontsize=8.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"))
ax_rel.tick_params(labelbottom=False)

# --- Panel A2: confidence histogram, shares x-axis with panel A ---
ax_hist = fig.add_subplot(gs[1, 0], sharex=ax_rel)
ax_hist.bar(bin_centers, ns, width=bin_width * 0.92, color="#999999", edgecolor="white", linewidth=0.8)
ax_hist.set_xlabel("Confidence", fontsize=11, fontweight="medium")
ax_hist.set_ylabel("Count", fontsize=9.5)
ax_hist.set_xlim(0.25, 1.0)

# --- Panel B: depth-structured confidence vs accuracy ---
ax_depth = fig.add_subplot(gs[:, 1])
d_vals = [s["depth"] for s in depth_stats]
d_conf = [s["avg_conf"] for s in depth_stats]
d_acc = [s["accuracy"] for s in depth_stats]

ax_depth.plot(d_vals, d_acc, marker="o", color=COLOR_ACC, linewidth=2, markersize=6, label="Observed accuracy", zorder=3)
ax_depth.plot(d_vals, d_conf, marker="D", color=COLOR_CONF, linewidth=2, markersize=6, label="Mean confidence", zorder=3)
ax_depth.fill_between(d_vals, d_acc, d_conf,
                       where=[a >= c for a, c in zip(d_acc, d_conf)],
                       color=COLOR_GAP_UNDER, alpha=0.18, interpolate=True, label="Underconfident region")
ax_depth.fill_between(d_vals, d_acc, d_conf,
                       where=[a < c for a, c in zip(d_acc, d_conf)],
                       color=COLOR_GAP_OVER, alpha=0.25, interpolate=True, label="Overconfident region")

ax_depth.axvline(REFERENCE_FLOOR_CM, color=COLOR_FLOOR, linestyle=":", linewidth=1.4, ymax=0.93)
ax_depth.axvline(MAX_VALID_DEPTH_CM, color="#111111", linestyle="--", linewidth=1.2, ymax=0.93)

# Single combined threshold key, placed inside the plot in open space -- avoids
# any collision with the title above or the axis labels below.
ax_depth.text(0.985, 0.985,
              f"· · ·  {REFERENCE_FLOOR_CM:.2f} cm: floor of this single run (seed 42)\n- - -  {MAX_VALID_DEPTH_CM:.2f} cm safety gate",
              transform=ax_depth.transAxes, fontsize=8, ha="right", va="top",
              bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#999999", alpha=0.95))

ax_depth.set_xlabel("Depth (cm)", fontsize=11, fontweight="medium")
ax_depth.set_ylabel("Value", fontsize=11, fontweight="medium")
ax_depth.set_title("B.  Confidence vs. accuracy across depth", fontsize=12, fontweight="bold", loc="left", pad=12)
ax_depth.set_ylim(-0.03, 1.08)
ax_depth.set_xlim(-0.5, 15.5)
ax_depth.legend(loc="lower left", fontsize=8.5, frameon=True, framealpha=0.9)
ax_depth.annotate("Underconfident\n(right but unsure)", xy=(4.0, 0.58), fontsize=8.5, color=COLOR_GAP_UNDER,
                   ha="center", style="italic")

fig.suptitle("Calibration of one saved mixed-depth CNN run (seed 42), 27.05 dB noise", fontsize=14, fontweight="bold", y=1.04)
fig.text(0.5, -0.05,
         "Panel A follows the reliability-diagram convention of Guo et al. (2017); hatched overlay marks the confidence-accuracy gap per bin.\n"
         "Panel B: confidence stays far below accuracy where the signal is strong; both settle near chance (about 0.33) at depth.",
         ha="center", fontsize=8, color="#555555")

plt.savefig(PLOT_PATH, dpi=200, bbox_inches="tight", pad_inches=0.35)
print(f"\nSaved publication-style figure to {PLOT_PATH}")

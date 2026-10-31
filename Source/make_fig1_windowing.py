"""Fig. 1: beat-anchored, non-overlapping window segmentation.

Draws the actual signal from physics_model.generate_signal (healthy class,
nominal 1.15 Hz heart rate) and marks the windows exactly as
cnn_classifier.extract_raw_windows cuts them: one 100-sample (0.2 s) window per
heartbeat trigger, starting at the first sample at or after the trigger.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from physics_model import generate_signal, stiffness_classes

FS, WIN, DUR, HR = 500, 100, 3.0, 1.15
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures_earlier",
                   "fig3_01_beat_anchored_windows.png")

time, sig, tof = generate_signal(stiffness_classes["healthy"], duration=DUR, heart_rate_hz=HR, sample_rate=FS)
beats = np.arange(0, DUR, 1.0 / HR)
starts = [int(np.searchsorted(time, b)) for b in beats]
assert len(starts) == 4 and all(s + WIN <= len(time) for s in starts)
assert all(starts[i] + WIN <= starts[i + 1] for i in range(3))   # non-overlapping

colors = ["#8e7cc3", "#6fa8dc", "#76c7b7", "#b6d77a"]
fig, ax = plt.subplots(figsize=(13, 5.6), dpi=150)
ax.plot(time, sig, color="#2c3e50", lw=0.9, zorder=3)
for i, (b, s) in enumerate(zip(beats, starts)):
    t0, t1 = time[s], time[s] + WIN / FS
    ax.axvspan(t0, t1, color=colors[i], alpha=0.45, zorder=1)
    ax.axvline(b, color="#c0392b", ls="--", lw=1.1, zorder=2)
    ax.text((t0 + t1) / 2, 1.22, f"W{i+1}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(b + 0.02, -1.25, f"trigger {i+1}: {b:.2f} s", ha="left", va="center", fontsize=9, color="#c0392b")
ax.annotate("", xy=(time[starts[0]] + WIN / FS, 1.02), xytext=(time[starts[0]], 1.02),
            arrowprops=dict(arrowstyle="<->", color="#333"))
ax.text(time[starts[0]] + WIN / FS / 2, 0.93, "100 samples\n(0.2 s)", ha="center", va="top", fontsize=9)
ax.text(1.405, 1.22, "no window between beats", ha="center", va="center", fontsize=10, color="#666", style="italic")
ax.set_xlim(-0.05, DUR + 0.05); ax.set_ylim(-1.55, 1.4)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Signal amplitude (a.u.)")
ax.set_title("Beat-anchored, non-overlapping window segmentation", fontsize=15, fontweight="bold")
fig.text(0.5, 0.005, f"3-second recording ({int(DUR*FS)} samples)   |   fs = {FS} Hz   |   window = {WIN} samples (0.2 s)   |   "
         "one window per heartbeat trigger   |   4 windows per recording   |   HR drawn per realization from 1.15 +/- 0.03 Hz",
         ha="center", fontsize=8.5, style="italic", color="#555")
ax.grid(alpha=0.25)
fig.tight_layout(rect=(0, 0.03, 1, 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT)
print("saved", os.path.abspath(OUT), "starts", starts)

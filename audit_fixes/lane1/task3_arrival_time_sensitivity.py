"""
Lane 1, Task 3 -- arrival-time (time-of-flight) sensitivity to c0 vs c_exact
at 100Hz, and whether class separability survives sampling quantisation.

TOF = distance / velocity. Uses c0 and c_exact from task1's CSV (100Hz row
per class) -- not recomputed independently, so numbers are guaranteed
consistent with Task 1.

Sample period: 1000/fs ms with fs=500Hz (Source/physics_model.generate_signal
default sample_rate=500) => 2 ms/sample, matching the protocol's stated value;
verified against the code default rather than assumed.
"""
import sys
import os
import csv
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Source"))
from physics_model import generate_signal  # noqa: E402 -- confirms default sample_rate

# Verify fs=500Hz / 2ms-per-sample against the actual function default.
import inspect
sig = inspect.signature(generate_signal)
DEFAULT_FS = sig.parameters["sample_rate"].default
assert DEFAULT_FS == 500, f"physics_model.generate_signal default sample_rate changed to {DEFAULT_FS}, expected 500"
MS_PER_SAMPLE = 1000.0 / DEFAULT_FS
print(f"Verified from Source/physics_model.generate_signal signature: fs={DEFAULT_FS}Hz -> {MS_PER_SAMPLE:.2f} ms/sample")

DISTANCES_MM = [5, 7.5, 10, 15, 20]

table1_csv = os.path.join(os.path.dirname(__file__), "table1_exact_vs_weak.csv")
with open(table1_csv, newline="") as f:
    table1 = list(csv.DictReader(f))
velocities_100hz = {r["class"]: (float(r["c0_m_s"]), float(r["c_exact_m_s"]))
                    for r in table1 if int(r["freq_Hz"]) == 100}

classes = list(velocities_100hz.keys())

rows = []
for d_mm in DISTANCES_MM:
    d_m = d_mm / 1000.0
    tof = {}
    for cls in classes:
        c0, c_exact = velocities_100hz[cls]
        tof[cls] = {
            "tof_c0_ms": d_m / c0 * 1000,
            "tof_cexact_ms": d_m / c_exact * 1000,
        }
    for cls_a, cls_b in itertools.combinations(classes, 2):
        for model in ("c0", "cexact"):
            key = f"tof_{model}_ms"
            gap_ms = abs(tof[cls_a][key] - tof[cls_b][key])
            gap_samples = gap_ms / MS_PER_SAMPLE
            separable = gap_samples >= 1.0
            rows.append({
                "distance_mm": d_mm, "class_pair": f"{cls_a} vs {cls_b}",
                "velocity_model": model, "gap_ms": gap_ms,
                "gap_samples": gap_samples,
                "separable_after_2ms_quantisation": separable,
            })

out_csv = os.path.join(os.path.dirname(__file__), "table3_arrival_time_sensitivity.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"{'d(mm)':>7}{'pair':<40}{'model':>8}{'gap(ms)':>10}{'gap(samples)':>14}{'separable>=1samp':>18}")
for r in rows:
    print(f"{r['distance_mm']:>7}{r['class_pair']:<40}{r['velocity_model']:>8}"
          f"{r['gap_ms']:>10.4f}{r['gap_samples']:>14.4f}{str(r['separable_after_2ms_quantisation']):>18}")

print(f"\nSaved: {out_csv}")

worst = min(rows, key=lambda r: r["gap_samples"])
print(f"\nWorst-case (smallest) gap across all distances/pairs/models: "
      f"{worst['class_pair']} at d={worst['distance_mm']}mm, {worst['velocity_model']} model, "
      f"{worst['gap_samples']:.4f} samples ({worst['gap_ms']:.4f} ms) -- "
      f"{'separable' if worst['separable_after_2ms_quantisation'] else 'NOT separable'} at fs=500Hz.")
n_not_separable = sum(1 for r in rows if not r["separable_after_2ms_quantisation"])
print(f"{n_not_separable}/{len(rows)} (distance, class-pair, model) combinations fall below 1 sample "
      f"({MS_PER_SAMPLE:.0f}ms) gap and are NOT reliably separable after quantisation to {MS_PER_SAMPLE:.0f}ms samples.")

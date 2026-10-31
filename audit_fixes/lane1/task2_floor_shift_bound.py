"""
Lane 1, Task 2 -- first-order floor-shift bound at 100Hz.

For each class, find the depth x_eq at which the EXACT-model amplitude
equals the WEAK-model amplitude at a reference depth x_ref:
    exp(-alpha_exact * x_eq) = exp(-alpha_weak * x_ref)
    => x_eq = x_ref * alpha_weak / alpha_exact

Uses alpha_weak / alpha_exact ratios from task1_exact_vs_weak.py's own CSV
output (table1_exact_vs_weak.csv), not recomputed independently, so this
step is guaranteed self-consistent with Task 1's numbers.

Reference depths (both from Source/constants.py, verified against the file
directly -- NOT hardcoded here from the protocol):
  8.48 cm -- the 24-seed mean certified floor (comment on
             SEED121_CERTIFIED_FLOOR_CM: "NOT the 24-seed mean (8.48cm +/-
             1.81cm)")
  6.50 cm -- constants.MAX_VALID_DEPTH_CM (the safety-gate depth)

IMPORTANT LABEL: this is a bound on PHYSICS-MODEL error (how far the exact
attenuation law would shift an amplitude-matched depth vs the weak-damping
law), not a re-certified accuracy floor. Re-certifying the floor requires
re-running eval on the saved seed121 weights, which is out of scope here.
"""
import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Source"))
from constants import MAX_VALID_DEPTH_CM  # noqa: E402

FLOOR_24SEED_MEAN_CM = 8.48  # from constants.py comment on SEED121_CERTIFIED_FLOOR_CM
REF_DEPTHS_CM = {
    "24seed_mean_floor_8.48cm": FLOOR_24SEED_MEAN_CM,
    "safety_gate_MAX_VALID_DEPTH_CM": MAX_VALID_DEPTH_CM,
}

table1_csv = os.path.join(os.path.dirname(__file__), "table1_exact_vs_weak.csv")
with open(table1_csv, newline="") as f:
    table1 = list(csv.DictReader(f))

rows = []
for r in table1:
    if int(r["freq_Hz"]) != 100:
        continue
    ratio = float(r["ratio_weak_over_exact"])
    for ref_name, x_ref in REF_DEPTHS_CM.items():
        x_eq = x_ref * ratio
        shift_cm = x_eq - x_ref
        rows.append({
            "class": r["class"], "reference_depth_label": ref_name,
            "reference_depth_cm": x_ref, "ratio_weak_over_exact_100Hz": ratio,
            "x_eq_cm": x_eq, "shift_cm": shift_cm,
        })

out_csv = os.path.join(os.path.dirname(__file__), "task2_floor_shift_bound.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("PHYSICS-ERROR BOUND, not a re-certified floor (needs eval on saved weights).")
print(f"{'class':<22}{'reference':<32}{'x_ref(cm)':>10}{'ratio':>8}{'x_eq(cm)':>10}{'shift(cm)':>11}")
for r in rows:
    print(f"{r['class']:<22}{r['reference_depth_label']:<32}{r['reference_depth_cm']:>10.2f}"
          f"{r['ratio_weak_over_exact_100Hz']:>8.4f}{r['x_eq_cm']:>10.4f}{r['shift_cm']:>11.4f}")

print(f"\nSaved: {out_csv}")

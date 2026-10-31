"""
Lane 1, Task 4 -- 20Hz sanity check (HEURISTIC amplitude-equivalence, NOT a
classifier prediction).

exp(-alpha20 * x20) = exp(-alpha100 * x100_ref)
=> x20 = x100_ref * (alpha_exact_100Hz / alpha_exact_20Hz)

Uses alpha_exact (not alpha_weak) at 20Hz and 100Hz, as specified in the protocol
"using alpha_exact". Both pulled from task1's CSV, not recomputed.

Reference depths: 8.48cm (24-seed mean floor) and 6.50cm
(constants.MAX_VALID_DEPTH_CM) -- same two references as Task 2, verified
there against constants.py / the 24-seed CSV directly.
"""
import sys
import os
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Source"))
from constants import MAX_VALID_DEPTH_CM  # noqa: E402

FLOOR_24SEED_MEAN_CM = 8.48
REF_DEPTHS_CM = {
    "24seed_mean_floor_8.48cm": FLOOR_24SEED_MEAN_CM,
    "safety_gate_MAX_VALID_DEPTH_CM": MAX_VALID_DEPTH_CM,
}

table1_csv = os.path.join(os.path.dirname(__file__), "table1_exact_vs_weak.csv")
with open(table1_csv, newline="") as f:
    table1 = list(csv.DictReader(f))

alpha_exact = {}
for r in table1:
    alpha_exact.setdefault(r["class"], {})[int(r["freq_Hz"])] = float(r["alpha_exact_per_m"])

rows = []
for cls, by_freq in alpha_exact.items():
    a20 = by_freq[20]
    a100 = by_freq[100]
    for ref_name, x100 in REF_DEPTHS_CM.items():
        x20_cm = x100 * (a100 / a20)
        rows.append({
            "class": cls, "reference_depth_label": ref_name,
            "x100_ref_cm": x100, "alpha_exact_20Hz": a20, "alpha_exact_100Hz": a100,
            "x20_heuristic_cm": x20_cm,
        })

out_csv = os.path.join(os.path.dirname(__file__), "table4_20hz_sanity_check.csv")
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print("HEURISTIC amplitude-equivalence depth at 20Hz, NOT a classifier prediction.")
print(f"{'class':<22}{'reference':<32}{'x100(cm)':>10}{'a20':>10}{'a100':>10}{'x20_heur(cm)':>14}")
for r in rows:
    print(f"{r['class']:<22}{r['reference_depth_label']:<32}{r['x100_ref_cm']:>10.2f}"
          f"{r['alpha_exact_20Hz']:>10.4f}{r['alpha_exact_100Hz']:>10.4f}{r['x20_heuristic_cm']:>14.3f}")

print(f"\nSaved: {out_csv}")

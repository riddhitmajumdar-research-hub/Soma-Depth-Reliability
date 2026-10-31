"""
Lane 3 audit — verification script (read-only).
Re-derives every numeric claim in FACTS.md directly from checked-in
Source/logs/*.csv and Source/constants.py values, independent of the
earlier research pass. Run from repo root:
    python3 audit_fixes/lane3/verify_numbers.py
"""
import csv
import numpy as np

print("=== Q1(c): depth grid size ===")
BASE_EVAL_DEPTHS_CM = np.linspace(0, 15, 12)
DENSIFIED_DEPTHS_CM = [3.5, 4.0, 4.3, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
merged = np.sort(np.unique(np.concatenate([BASE_EVAL_DEPTHS_CM, DENSIFIED_DEPTHS_CM])))
print("BASE points:", len(BASE_EVAL_DEPTHS_CM), "DENSIFIED points:", len(DENSIFIED_DEPTHS_CM), "merged unique:", len(merged))

print("\n=== Q3: ETS Montreal shuffled-label null ===")
with open("Source/logs/ets_path2_shuffle_results.csv") as f:
    r = list(csv.DictReader(f))
vals = [float(row["control_acc"]) for row in r]
print("n_shuffles:", len(vals), "mean:", np.mean(vals), "sd(ddof=1):", np.std(vals, ddof=1))

print("\n=== Q2/Q7: CNN + ResNet1D sweep progress ===")
with open("Source/logs/degradation_aware_final_results.csv") as f:
    rows = list(csv.DictReader(f))
print("CNN sweep rows:", len(rows))
anomalous_cnn = [row["seed"] for row in rows if row["anomalous_recovery"].strip().lower() == "true"]
print("CNN anomalous_recovery seeds:", anomalous_cnn, "count:", len(anomalous_cnn))

with open("Source/logs/degradation_aware_resnet_final_results.csv") as f:
    rrows = list(csv.DictReader(f))
print("ResNet sweep rows so far:", len(rrows), "seeds:", [row["seed"] for row in rrows])
anomalous_resnet = [row["seed"] for row in rrows if row["anomalous_recovery"].strip().lower() == "true"]
print("ResNet anomalous_recovery seeds so far:", anomalous_resnet)

print("\n=== Q4: 'three seeds' crossover-ambiguity check ===")
for row in rows:
    gap = float(row["naive_floor_cm"]) - float(row["last_significant_depth_cm"])
    if gap > 1.0:
        print(f"  seed {row['seed']}: naive_floor={row['naive_floor_cm']} last_sig={row['last_significant_depth_cm']} gap={round(gap, 3)}")
count = sum(1 for row in rows if float(row["naive_floor_cm"]) - float(row["last_significant_depth_cm"]) > 1.0)
print("count of seeds with gap > 1.0cm:", count)

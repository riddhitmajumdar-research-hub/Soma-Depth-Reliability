# Regenerates logs/gate_statistic_verification.csv from the intact
# source-of-truth 24-seed floor data (logs/degradation_aware_final_results.csv),
# reproducing the gate-candidate percentile analysis after the file was
# accidentally overwritten by a later, differently-scoped script.

import numpy as np
import csv

SRC_CSV = "logs/degradation_aware_final_results.csv"
OUT_CSV = "logs/gate_statistic_verification.csv"

floors = []
with open(SRC_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        floors.append(float(row["naive_floor_cm"]))
floors = np.array(floors)
n = len(floors)

rng = np.random.default_rng(42)
N_BOOTSTRAP = 10000

# Bootstrap of resample MEANS, 5th percentile (wrong statistic for a safety
# gate -- bounds the population mean, not individual-unit risk)
boot_means = np.array([rng.choice(floors, size=n, replace=True).mean() for _ in range(N_BOOTSTRAP)])
boot_of_means_5th = np.percentile(boot_means, 5)

# Bootstrap of INDIVIDUAL resampled values, percentile
boot_individuals = np.array([rng.choice(floors, size=1, replace=True)[0] for _ in range(N_BOOTSTRAP)])
boot_of_individuals_5th = np.percentile(boot_individuals, 5)

raw_5th = np.percentile(floors, 5)
raw_10th = np.percentile(floors, 10)
worst_observed = floors.min()

print(f"n={n} seeds")
print(f"Bootstrap of resample MEANS, 5th percentile: {boot_of_means_5th:.4f}cm "
      f"(wrong statistic for a safety gate -- bounds the population mean, not individual-unit risk -- do not use)")
print(f"Bootstrap of INDIVIDUAL resampled values, percentile: {boot_of_individuals_5th:.4f}cm")
print(f"Raw empirical 5th percentile of the 24 observed floors: {raw_5th:.4f}cm")
print(f"Raw empirical 10th percentile of the 24 observed floors: {raw_10th:.4f}cm (adopted for MAX_VALID_DEPTH_CM)")
print(f"Worst observed seed (minimum): {worst_observed:.4f}cm")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["statistic", "value_cm", "note"])
    writer.writerow(["bootstrap_of_means_5th_percentile", f"{boot_of_means_5th:.4f}",
                      "wrong statistic for a safety gate -- bounds population mean, not individual-unit risk -- do not use"])
    writer.writerow(["bootstrap_of_individuals_percentile", f"{boot_of_individuals_5th:.4f}", ""])
    writer.writerow(["raw_empirical_5th_percentile", f"{raw_5th:.4f}", ""])
    writer.writerow(["raw_empirical_10th_percentile", f"{raw_10th:.4f}", "adopted for MAX_VALID_DEPTH_CM"])
    writer.writerow(["worst_observed_seed_minimum", f"{worst_observed:.4f}", ""])

print(f"\nSaved to {OUT_CSV}")

# post_audit

Data, scripts and saved results behind Logs 39 to 43 of the top-level README. Each log lists the files it uses in its Evidence section.

| Folder | Log | Configuration | Contents |
|---|---|---|---|
| `simulator_audit/` | 39 | A | Floors with the viscosity lookup and with the stage label, the cirrhosis viscosity sweep, the timing and noise generator check, and the harness and aggregation scripts. `hardware_effects.py` is the hardware module the harness imports. The attenuation module that carries the `eta_by_stage` flag is not included; the harness expects it on the import path. |
| `rebuilt_dataset/` | 40 | B (accuracies only) | Dataset configurations, per-seed records of the four model families, the five-variant ablation, the timing-error and alternative stiffness runs, the scaler check, and the training and data code. |
| `family_floors/` | 41 | B | Per-seed and per-evaluation floors, variance components, classwise floors, archive checks, and the permutation and analysis code. |
| `calibration_attribution/` | 42 | B | Calibration, logit-gap, attribution and initialisation-check tables and the analysis code. |
| `training_budget/` | 43 | B | A bundle with the saved permutation counts of the 200-epoch runs and the timing-error runs (`verify.py` recomputes the headline numbers from them and compares with `expected_results.txt`), and `derived/` with per-seed tables for 60, 200 and 600 epochs, the classwise floors, the accuracy against the shuffled-label null and the timing-error pairs. |

Changes made when the files were copied into this repository:

- The eight JSON files in `rebuilt_dataset/configs/` and `family_floors/protocol_p3_v1.json` had a one-line creation-date field removed; every other byte is unchanged.
- In `training_budget/`, absolute checkpoint paths inside the `.npz` files were shortened to relative paths before they were added (`SCRUB_MAP.tsv` lists the original and the new hash of each file); numeric arrays are unchanged.
- Files in `training_budget/derived/` were renamed from the names of their original figure-data files: `floor_by_budget_8_seeds.csv`, `floor_200_three_draws_vs_60.csv`, `classwise_vs_overall_floor_200.csv`, `accuracy_null_band.csv`, `jitter_pairs_200_epochs.csv`, `k3_200_epoch_results.md` (200-epoch summary of 24 networks and three draws), `jitter_2ms_results.md`, `budget_pilot_summary.txt`.
- Running `verify.py` writes `outputs/stage2c/` and `outputs/stage2d/` inside `training_budget/`; these generated folders are not tracked.

Not included: the raw permutation counts of the 60-epoch and 600-epoch runs, the trained networks, and the original working tree around these scripts.

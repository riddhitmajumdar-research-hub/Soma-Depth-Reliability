# rf cloud run report -- seeds 113-124

Protocol: unchanged from `audit_fixes/lane8/PREREG.md` and
`lane8_harness.py` (see that file for the full pre-registered method:
Random Forest, 300 trees, trained on raw 100-sample windows, 200
shuffled-label controls per seed, 23-depth eval grid, BH-FDR alpha=0.05,
first-failed-depth floor convention). This wrapper
(`audit_fixes/cloud/rf/run_rf_seeds.py`) only restricts the seed range to
113-124 and adds BH-adjusted p-values as a post-hoc derived column; no
protocol parameter was changed. Full data: `results/rf_floors_113-124.csv`.

## Per-seed floor summary

| seed | floor_cm | last_significant_cm | anomalous_recovery | total_seconds |
|---|---|---|---|---|
| 113 | 13.6364 | 12.2727 | False | 671.9 |
| 114 | 13.6364 | 12.2727 | False | 665.7 |
| 115 | 12.2727 | 13.6364 | **True** | 640.1 |
| 116 | 13.6364 | 12.2727 | False | 657.3 |
| 117 | 13.6364 | 12.2727 | False | 644.8 |
| 118 | 12.2727 | 10.9091 | False | 634.6 |
| 119 | 13.6364 | 12.2727 | False | 614.6 |
| 120 | 12.2727 | 10.9091 | False | 629.1 |
| 121 | 10.9091 | 9.5455 | False | 642.3 |
| 122 | 13.6364 | 12.2727 | False | 630.4 |
| 123 | 13.6364 | 12.2727 | False | 629.2 |
| 124 | 12.2727 | 10.9091 | False | 610.9 |

## Aggregate (12 seeds, 113-124 only -- not the full 24-seed set)

- Mean floor: **12.95 cm**
- SD (ddof=1): **0.92 cm**
- Median: **13.64 cm**
- Range: **10.91 - 13.64 cm**
- Seeds with floor < 6.50 cm gate: **0 / 12**

These numbers describe seeds 113-124 only. They are **not** a
certification result on their own -- PREREG.md section 7's full analysis
plan (overlap with CNN floors, Mann-Whitney/Wilcoxon tests, Clopper-Pearson
CI on the gate) requires all 24 seeds (101-124) and is explicitly deferred
until `run rf-101-112`'s seeds (currently incomplete -- see STATUS.md)
are also done. No comparison against CNN or ResNet
results was performed in this run.

## Flagged observations (not resolved here, per PREREG.md section 8 convention)

- One seed (115) shows `anomalous_recovery=True`: a *higher* depth
  (13.6364 cm) was BH-significant while a shallower depth (12.2727 cm,
  the reported floor) was not. This is a real feature of that seed's
  per-depth significance pattern, not a computation error -- the
  first-failed-depth convention (PREREG.md section 4) reports the floor
  as the first non-significant depth regardless of what happens at deeper
  depths, so a later "recovery" is surfaced via the
  `anomalous_recovery_rf` flag rather than silently overriding the floor.
  Left as-is per the pre-registered convention; flagged here for whoever
  does the full 24-seed analysis, not adjudicated.
- All 12 floors in this seed range fall on only 3 of the 23 grid depths
  (10.9091, 12.2727, 13.6364 cm) -- consistent with a genuinely tight
  floor distribution for this seed range, not inspected further (no
  n=12 statistical test was run, per the deferred full-24-seed analysis
  plan above).
- Per-seed runtime (611-672s) ran ~1.5-1.6x the single-seed PREREG.md
  benchmark (410s for seed 101) -- within the 3x stop threshold, not
  investigated further; see STATUS.md caveat 1.

## Reproducibility

- Data-generation seeding: `np.random.seed(seed)` at the top of
  `run_one_seed`, unchanged from `lane8_harness.py`.
- Real model: `RandomForestClassifier(random_state=seed)`.
- Control model `i`: `RandomForestClassifier(random_state=seed*10_000+i)`.
- All 12 seeds' full per-depth detail (real + 200 control accuracies,
  raw and BH-adjusted p-values) is in `results/rf_floors_113-124.csv`,
  sufficient to recompute every number in this report offline without
  retraining.

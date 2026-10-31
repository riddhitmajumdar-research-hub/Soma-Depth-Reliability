# 11. Lane 8 and the cloud runs: random forest floor

Run base `run base`, the repo snapshot.

## Question

Is the depth floor a property of the physics and noise, or of the CNN? Certify the same floor with a random forest on the identical pipeline.

## Method

`audit_fixes/lane8/PREREG.md` (written before any run) reuses the dataset builder, degraded-window generator, depth grid, 200 shuffled controls and BH function unchanged; only `train_rf` (300 trees, raw 100-sample windows) is new. Local benchmark, seed 101: 410 s, floor 13.64 cm. The 24-seed run was done on a cloud container in two runs through a wrapper (`audit_fixes/cloud/rf/run_rf_seeds.py`) because the harness hardcodes seeds 101 to 124. Each seed was committed and pushed as it finished.

## Numbers

- `audit_fixes/cloud/rf/results/rf_floors_101-112.csv`: mean 12.84 cm, SD 0.91, range 12.27 to 15.00. Seed 106 anomalous recovery; seed 110 pinned at the 15 cm grid maximum.
- `audit_fixes/cloud/rf/results/rf_floors_113-124.csv`: mean 12.95, SD 0.92, range 10.91 to 13.64. Seed 115 anomalous recovery.
- Across both, 0 of 24 seeds are below 6.50 cm and the median is 12.27 cm (matches the write-up ledger; recounted from the two per-seed tables here).
- Timing: 468 to 513 s per seed on 101 to 112, 611 to 672 s on 113 to 124.

## Deviations and caveats

- The RF run wrote BH-adjusted p-values itself, since the imported function returns only a flag.
- The harness writes per-seed CSVs into `audit_fixes/lane8/logs/`, which the cloud protocol excluded; those side-effect files were deleted before the results were saved.
- A cloud-side `matplotlib` import in `physics_model.py` is an undeclared dependency.
- Cloud seed-101 floor 9.55 matched the Mac value in the battery (12).

## Limitation

No 24-seed RF report or combined CNN/ResNet1D/RF comparison script is committed. `lane8/` has no REPORT.md. The RF median and the 0/24 count exist in the two run reports and the write-up. The RF results CSVs (`results/rf_floors_*.csv`) are also committed locally at `audit_fixes/cloud/rf/results/` (277 rows each, pulled from both runs and checked against them).

## Where the parts are

The [topic table in the index](README.md#where-to-look) lists every place that holds part of the random forest result. This note is the one that gives the numbers and their caveats; `audit_fixes/lane8/README.md` and the run reports point back here.

## Reproducibility

The run reports are in `audit_fixes/cloud/rf/`: `REPORT.md` (seeds 113 to 124) and `REPORT_101-112.md`. Re-run with `python audit_fixes/cloud/rf/run_rf_seeds.py --start 101 --end 112`.

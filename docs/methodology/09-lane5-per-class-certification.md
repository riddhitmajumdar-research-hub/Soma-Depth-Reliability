# 09. Lane 5: per-class floors across 24 seeds

Sequence: build and benchmark; main run start, interrupted (RESUME_COMMAND.txt), resumed, finished; clean-trained sweep; balanced-accuracy eval; diagnostics; clean per-class recomputation and bug fix (a code comment notes the fix).

Follows from the seed-121 per-class result in the external review script batch (02). The write-up, Section F and the Appendix B per-class row summarize the outcome.

## Question

Can the aggregate protocol (200 shuffled-label controls, BH-FDR, first non-significant depth) certify per-class floors over all 24 seeds?

## Method

`recall_floor_main_run/lane5_harness.py` imports training, data generation and BH from the certified sweep unchanged and adds per-control per-class recall, per-seed aggregate depth detail (fixing the gap Lane 2 found), and saved weights for the real model and all 200 controls per seed (about 16,000 files under `audit_fixes/lane5/`). PREREG.md was written before the run. Validation on seed 121 reproduced the depth-8.48 recall values (0.1475, 0.1575, 0.9688). Benchmark seed 101 took 815 s and reproduced the certified floor 9.545 cm.

## What happened

1. **Recall null is degenerate.** After 2 seeds every per-class floor was 0.0 cm. At depth 0, seed 101, 37 to 46% of the 200 shuffled controls have recall 0 for a given class and 12 to 14% have recall 1 (cirrhosis: 91 and 24 of 200), i.e. they collapse to one class, and a constant predictor scores recall 1.0 for its class, so the permutation p stays large even on easy depths. The result stands as a negative finding. Main run: all 24 seeds x 3 classes give 0.0.
2. **Aggregate floor did not reproduce with fresh eval noise.** The rerun floor matched the certified one in 11 of 24 seeds (seed 102: 6.50 vs 5.50; seed 114: 2.73 vs 7.50). Rerun mean 8.21 cm, SD 1.94; certified 8.48, 1.81. Two seeds fall below 6.50 in each set, and they are not the same two (rerun: 111, 114; certified: 102, 111). Pre-registered as a possibility and not resolved.
3. **Multiplicity check** (`multiplicity_robustness.py` at the lane 5 root): pooled BH across seeds and depths changed 3 seeds' floors; Westfall-Young max-T changed all 24 (mean 2.76 cm, median 3.75), a much stricter criterion.
4. **Balanced accuracy statistic** (`balanced_accuracy_retry/perclass_balacc_eval.py`, PREREG_perclass.md): caps a constant predictor at 0.5. Eval-only on the saved weights with a fixed regenerated test set per (seed, depth). Under a last-significant-depth reading of the saved detail CSVs: cirrhosis mean 6.88 cm (median 7.0, one seed never significant), healthy 13.47 (median 13.64), fibrosis 0.23 (23 of 24 seeds never significant). Healthy is high because a collapsed model predicting healthy gets partial credit; this is why the write-up calls the per-class floor unattainable under this protocol.
5. **Descriptive recall curves** (`balanced_accuracy_retry/analysis/soma_perclass_summary.txt`): recall below 0.45 at a median 6.82 cm for fibrosis (all 24 seeds; 4 before 6.50 cm) and 7.50 cm for cirrhosis (20 of 24; 3 before 6.50), healthy in 1 seed (104, 10.91 cm). Share of windows predicted healthy: median 0.52 at 6.50 cm to 0.90 at 15 cm.
6. **Clean-trained comparison sweep** (`clean_trained_sweep/lane5_harness_clean.py`, PREREG_clean.md, written before the run): same evaluation, training on clean noiseless windows. Aggregate floor mean 8.85 cm, SD 4.94, median 9.55, 6 seeds at the 15 cm grid maximum, 11 below 6.50 cm (degradation-aware: 2). One variable changed, training data.
7. **Bug in the clean harness.** The per-class floor loop broke at the first non-significant depth and returned 0.0 when depth 0 itself failed BH. Fixed to take the last significant depth (`clean_trained_sweep/lane5_harness_clean.py.bak_bugA` holds the old loop). Recomputing the clean per-class floors with the fixed rule (`diagnostics/recompute_clean_perclass.py`) still gives 0.0 for every seed and class.
8. Seed 103 partial artifacts from the interrupted run are in `quarantine_seed103_partial/` .

## Limitations

`lane5/REPORT.md` was last edited during the run and still describes it as in progress. STEP (sliding-window stride) was dropped because it needed new windowing code. The 0.45 threshold follows the body text of the spec; the header said 40%. No committed report covers items 3 to 7; this file is the only write-up besides the write-up.

## Reproducibility

`python audit_fixes/lane5/recall_floor_main_run/lane5_harness.py`, then `run_chain.py`, `balanced_accuracy_retry/perclass_balacc_eval.py`, `multiplicity_robustness.py --run main|clean`. About 5.4 h for the main run, 2.4 h for the clean sweep. Files saved from outside the repo are described below.

### Files saved from outside the repo

They were first saved together in `audit_fixes/lane5/diagnostics/` and moved beside their experiments in the reorganization of the same day. Contents are unchanged apart from path strings inside the scripts. Where they are now:

- `run_monitoring/`: `soma_status.sh`, `soma_status_log.txt`, `soma_diag.py`, `soma_diag_output.txt`
- `balanced_accuracy_retry/analysis/`: `aggregate_balacc.py`, `balacc_floors_by_class.csv`, `soma_perclass_summary.py`, `soma_perclass_summary.txt`
- `clean_trained_sweep/analysis/`: `recompute_clean_perclass.py`, `clean_per_class_floors_FIXED.csv`, `check_controls.py`, `soma_null_check.py`, `soma_nullcheck_output.txt`
- `audit_fixes/lane7/lane7_check.py`: compares the Lane 5 rerun floors and the ResNet1D floors, so it sits with Lane 7

`soma_status_log.txt`: 15-minute heartbeat log showing the harness stopping after seed 124 while the chain runner stayed up. `soma_diag_output.txt`: snapshot (8 clean seeds, 10 balanced-accuracy seeds, seed 106 confusion matrices at five depths). `soma_nullcheck_output.txt`: three-null floor comparison; it ends in a KeyError (`acc_real`) on the replicate section and was not rerun. `soma_perclass_summary.txt`: the recall summary in item 5. Generating scripts sit beside them, except that `aggregate_balacc.py` and `recompute_clean_perclass.py` write to `/tmp/`, and the CSVs here are copies of that output.

## Leads to

The write-up, Section F; the noise probe in 12.

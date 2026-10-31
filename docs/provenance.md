# Provenance record

How the files in this repository relate to the working files they came from, organised by kind of change. No result values were changed.

## Paths made relative

Machine-specific absolute paths were replaced in 35 files. In the 15 Python scripts below, the hard-coded repository root was replaced by a root derived from the script's own location (`Path(__file__).resolve().parents[N]`). In the 20 other files (logs, text, shell and Markdown), the literal absolute prefix was replaced by `<REPO_ROOT>` (19 files) or `<HOME>` (1 file, `audit_fixes/lane5/run_monitoring/soma_status_log.txt`).

The file `audit_fixes/lane5/run_monitoring/soma_status_log.txt` also had the home directory replaced by `<HOME>`. No result values were modified. In every non-script file, restoring the original prefix reproduces the original file byte for byte.

## Python scripts (relative root)
- Source/ets_controls_common.py
- Source/ets_controls_run_variants.py
- Source/ets_montreal_path2_retrain.py
- Source/log21_figures.py
- Source/make_fig11_corrected_probe.py
- Source/make_fig12_corrected_progression.py
- Source/make_figures.py
- audit_fixes/lane5/balanced_accuracy_retry/analysis/soma_perclass_summary.py
- audit_fixes/lane5/clean_trained_sweep/analysis/soma_null_check.py
- audit_fixes/lane5/run_monitoring/soma_diag.py
- audit_fixes/lane7/eval_seed110.py
- audit_fixes/lane7/run_lane7_analysis.py
- audit_fixes/lane7/run_lane7_v2_analysis.py
- audit_fixes/lane7/verify_seed_pairing.py
- outputs/external_validation_controls/run_variants_scratch_copy.py

## Other files (<REPO_ROOT> / <HOME>)
- audit_fixes/lane1/task1_stdout.log
- audit_fixes/lane1/task2_stdout.log
- audit_fixes/lane1/task3_stdout.log
- audit_fixes/lane1/task4_stdout.log
- audit_fixes/lane1/task5_stdout.log
- audit_fixes/lane3/FACTS.md
- audit_fixes/lane5/clean_trained_sweep/clean_sweep/logs/lane5_clean_run.log
- audit_fixes/lane5/recall_floor_main_run/RESUME_COMMAND.txt
- audit_fixes/lane5/recall_floor_main_run/logs/lane5_benchmark_raw_stdout.txt
- audit_fixes/lane5/recall_floor_main_run/logs/lane5_chain_run.log
- audit_fixes/lane5/recall_floor_main_run/logs/lane5_chain_run_nohup.log
- audit_fixes/lane5/recall_floor_main_run/logs/lane5_run.log
- audit_fixes/lane5/recall_floor_main_run/logs/lane5_run_resume.log
- audit_fixes/lane5/run_monitoring/soma_status.sh
- audit_fixes/lane5/run_monitoring/soma_status_log.txt
- audit_fixes/lane7/final_v1_output.txt
- audit_fixes/lane7/lane7_output_raw.txt
- audit_fixes/lane7/seed110_eval_raw.txt
- audit_fixes/lane8/logs/lane8_run.log
- outputs/figures_earlier/manifest.md

## Timestamps removed from saved run logs

Machine timestamps were removed from the listed log files that contained them, and from other saved run logs not listed here, which therefore no longer match their originals byte for byte.

## Folders regrouped and renamed

The original working files were reorganised for readability: folders were regrouped and renamed, log titles were tidied, and maps, tables and indexes were added to help navigation.

## Logs written up from the saved files

Logs 26 to 43 were written up afterwards from the saved files, and each says so. README Logs 26 to 37 were rewritten with their reasoning, methods and corrections, and one-line pointers were added to Logs 14, 15 and 19.

## Wording edited for consistency

The evidence files under `audit_fixes/cloud/` (reports, run logs, protocol and run files for the random forest and robustness runs, and three check reports), the README logs 26 to 37, the pointers added to Logs 14, 15 and 21, and the corrected provenance note in `outputs/external_validation_controls/` are part of the evidence files. In them, absolute paths became `<REPO_ROOT>`, and branch names, revision identifiers and process wording were replaced or removed; no result values were modified. Branch names, a cloned-folder name and a revision identifier in `docs/methodology/11-lane8-and-cloud-random-forest.md`, `docs/methodology/12-cloud-robustness-battery.md` and `audit_fixes/cloud/vischeck/FINDINGS.md` were replaced by "run <name>"; no numbers changed. The README index entries for Logs 27 to 37, and one sentence each in Log 21 and the introduction, were edited for consistency.

Process wording in the following files was edited for consistency; no result values were changed (the numeric tokens of each file are identical before and after): `Source/ets_montreal_path2_retrain.py`, `Source/weak_damping_error_bound.py`, `audit_fixes/cloud/smoke/RUNLOG.md`, `audit_fixes/lane1/REPORT.md`, `audit_fixes/lane1/task2_floor_shift_bound.py`, `audit_fixes/lane1/task3_arrival_time_sensitivity.py`, `audit_fixes/lane1/task4_20hz_sanity_check.py`, `audit_fixes/lane1/task5_monte_carlo.py`, `audit_fixes/lane3/FACTS.md`, `audit_fixes/lane4/PREREG.md`, `audit_fixes/lane4/PREREG_v2.md`, `audit_fixes/lane4/REPORT_v2.md`, `audit_fixes/lane5/REPORT.md`, `audit_fixes/lane5/balanced_accuracy_retry/PREREG_perclass.md`, `audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc_eval.py`, `audit_fixes/lane5/clean_trained_sweep/PREREG_clean.md`, `audit_fixes/lane5/lane5_config.py`, `audit_fixes/lane5/recall_floor_main_run/PREREG.md`, `audit_fixes/lane5/run_chain.py`, `audit_fixes/lane6/FACTS_6c.md`, `audit_fixes/lane6/q9_seed121_beat_index_sweep_6c.py`, `audit_fixes/lane7/PREREG.md`, `audit_fixes/lane7/REPORT.md`, `audit_fixes/lane8/PREREG.md`, `audit_fixes/lane8/lane8_config.py`, `audit_fixes/lane8/lane8_harness.py`, `docs/methodology/04-lane1-exact-attenuation.md`, `docs/methodology/11-lane8-and-cloud-random-forest.md`, `docs/methodology/12-cloud-robustness-battery.md`, `README.md`, `SOMA_Consolidated_Methodology.py`, `audit_fixes/lane2/REPORT.md`, `audit_fixes/lane4/REPORT.md`, `audit_fixes/lane5/multiplicity_robustness.py`, `audit_fixes/lane5/recall_floor_main_run/RESUME_COMMAND.txt`, `audit_fixes/lane5/recall_floor_main_run/lane5_harness.py`, `audit_fixes/lane5/recall_floor_main_run/logs/lane5_chain_run_nohup.log`, `audit_fixes/lane7/PREREG_v2.md`, `audit_fixes/lane7/REPORT_v2.md`, `audit_fixes/lane7/lane7_output_raw.txt`, `audit_fixes/lane7/run_lane7_analysis.py`, `docs/methodology/02-external-review-fixes.md`, `docs/methodology/06-lane3-fact-checks.md`, `docs/methodology/08-lane6-beat-offset-mechanism.md`, `audit_fixes/cloud/rf/REPORT_101-112.md`. The logs of a first phantom control run were added in `outputs/external_validation_controls/first_run/`.

## Files added during the reorganisation

Added during the reorganisation: `Source/pool_clean_confusion.py` and `audit_fixes/lane5/clean_trained_sweep/clean_sweep/pooled_confusion_summary.csv` (pooled clean-trained confusion matrices, Log 32).

Also added during the reorganisation: `outputs/stage4/data/stage4_progression_baseline_corrected.csv`, `stage4_progression_clean_cnn.csv`, `stage4_progression_degaware_seed42.csv` and `stage4_progression_plotted_values.csv` (exports and plotted points of the progression figure, Log 35), `outputs/stage4/figures/stage4_full_progression_corrected_first_version.png` (the first redraw, kept as the record), and a changed `Source/make_fig12_corrected_progression.py`, which now reads only those CSV files. The figure `stage4_full_progression_corrected.png` was redrawn.

Also added during the reorganisation: `Source/make_fig02_measurement_chain.py`, `Source/make_fig08a_pooled_confusion_2x2.py`, `Source/make_fig11_logit_gap_vs_depth.py`, `outputs/figures_final/fig02_measurement_chain.png`, `fig08a_pooled_confusion_2x2.png`, `fig11_logit_gap_vs_depth.png` with their three CSV files, and `audit_fixes/proxy_sweeps/` (README and `pure_noise_probe/`: the probe script with its absolute paths replaced by paths relative to the folder, the count files, the summary and the logs; no result values changed).

Also added during the reorganisation (Log 38): `audit_fixes/proxy_sweeps/PREREG.md`, `summary_table.py`, `diagnose_zero_floors.py` and, in `audit_fixes/cloud/proxy_sweeps/results/`, `proxy_floors_<seed>.csv`, `proxy_depth_detail_<seed>.csv`, `proxy_run_<seed>.log` (seeds 101 to 108), `DEVIATIONS.md`, `summary_output.txt`, `summary_by_condition.csv` and `diagnostic_zero_floors.csv`. Their content is unchanged from the run; no absolute path occurred in them, so no replacement was made.

## Approximate dates from the author's notebook

Each of the 43 README logs opens with an approximate date of execution, taken from the author's notebook and not from file timestamps; the Phases table has an "Approximate dates" column, and the Conventions block has one sentence explaining the dates. No result values were changed.

## Pins and environment note

`requirements.txt` pins exact versions of the local environment the scripts ran in, and `docs/environment.md` lists the other environments used for some runs and figures. The README has a short reading path and a sentence on how to read the three headline floors. No result values were changed.

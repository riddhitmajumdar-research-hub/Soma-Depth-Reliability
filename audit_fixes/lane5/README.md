# Lane 5: per-class floors across 24 seeds

**Question.** Can the aggregate protocol (200 shuffled-label controls, BH-FDR, first non-significant depth) certify per-class floors over all 24 seeds?

**Methodology note.** [09](../../docs/methodology/09-lane5-per-class-certification.md). It is the only write-up of the balanced-accuracy, multiplicity and clean-sweep results.

**Current source of truth.** `REPORT.md` here was last edited during the main run, and is superseded by note 09. It is kept as the record of that moment.

## Layout

| Path | What it is | Pre-registration |
|---|---|---|
| [recall_floor_main_run/](recall_floor_main_run/) | The 24-seed main run with a per-class recall statistic: harness, one-seed benchmark, seed-121 validation, resume and alive scripts, logs, shuffled controls. Weights and `quarantine_seed103_partial/` are local and ignored by git. | `PREREG.md` |
| [balanced_accuracy_retry/](balanced_accuracy_retry/) | Second statistic, evaluated on the saved main-run weights. `perclass_balacc/` is its output. `analysis/` holds the aggregation and the recall summary. | `PREREG_perclass.md` |
| [clean_trained_sweep/](clean_trained_sweep/) | Comparison sweep trained on clean windows. `clean_sweep/` is its output. `lane5_harness_clean.py.bak_bugA` is the harness before the loop fix. `analysis/` holds the recomputation and null checks. | `PREREG_clean.md` |
| [run_monitoring/](run_monitoring/) | Status script and log from the main run, and the diagnostic snapshot. | none |
| `lane5_config.py` | Shared settings, imported by the harnesses. | |
| `run_chain.py` | Waits for 24 rows in the main results CSV, then runs the clean sweep and the multiplicity script. | |
| `multiplicity_robustness.py` | Pooled BH and Westfall-Young over the main run and the clean sweep. | |

The scripts under `analysis/` and `run_monitoring/`, and `../lane7/lane7_check.py`, were written outside the repo and copied in. Their original write times are not recorded. `aggregate_balacc.py` and `recompute_clean_perclass.py` write to `/tmp/`; the CSVs beside them are copies of that output.

## Old names

The PREREG files and `REPORT.md` describe the run as it was before the folders were split. Their paths map as follows.

| In the text | Now |
|---|---|
| `lane5/logs/`, `controls/`, `seed_weights/`, `lane5_harness.py`, `lane5_benchmark.py`, `check_alive.sh`, `RESUME_COMMAND.txt` | `recall_floor_main_run/…` |
| `lane5/perclass_balacc/`, `perclass_balacc_eval.py` | `balanced_accuracy_retry/…` |
| `lane5/clean_sweep/`, `lane5_harness_clean.py` | `clean_trained_sweep/…` |
| `lane5/diagnostics/…` | `run_monitoring/`, `balanced_accuracy_retry/analysis/`, `clean_trained_sweep/analysis/`, or `../lane7/lane7_check.py` |

No script was run again as part of the move. Each moved script was checked to import; a full run needs the weights, which are local.

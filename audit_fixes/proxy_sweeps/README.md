# Proxy sweeps

These files back values in the record that come from retrained networks, not from the certified 24-seed weights. Each run uses 8 networks (seeds 101 to 108), the real model only and no shuffled controls, so none of it is a certified floor.

## `pure_noise_probe/`

What the retrained random forest and ResNet1D predict on 300 pulse-free noise windows per seed, and on 300 windows per true class at 6.50 cm, at the classifier's own median certified floor (12.27 cm for the random forest, 10.91 cm for the ResNet1D) and at 15.00 cm. The training code is the repository's own (`Source/degradation_aware_final_sweep.py`, `Source/degradation_aware_resnet_final_sweep.py`, the random forest of `audit_fixes/lane8/`); the noise windows are built as in the robustness battery's condition 8.

| File | What it is |
|---|---|
| `probe.py` | the script (run once for `rf`, once for `resnet`; training happens when it is run, so do not run it casually: about 8 s for the random forest and about 107 s for the ResNet1D, from the logs) |
| `raw_counts_rf.csv`, `raw_counts_resnet.csv`, `raw_counts_all.csv` | predicted-class counts per seed, condition and true class |
| `summarize.py`, `summary.txt` | the pooled shares and per-seed recall (run from this folder) |
| `rf.log`, `resnet.log` | run logs with the elapsed time |

Record values this backs (computed from `raw_counts_*.csv`): random forest on pure noise, pooled over 8 seeds: 66.5% healthy (62.7 to 71.3% per seed), 33.4% cirrhosis, 0.1% fibrosis; ResNet1D on pure noise: 51.6% healthy, 35.6% cirrhosis, 12.8% fibrosis (healthy 8.7 to 94.0% per seed, so no consistent default). They are indicative, not certified.

## Harness for the Table 1 conditions (run; results in `audit_fixes/cloud/proxy_sweeps/results/`)

The protocol, decided before any result, is in [`PREREG.md`](PREREG.md). The harness has been run for all 13 conditions and seeds 101 to 108; the results, logs, summary and zero-floor diagnostic are in `audit_fixes/cloud/proxy_sweeps/results/` and are written up in README Log 38.

| File | What it is |
|---|---|
| `PREREG.md` | the question, the conditions with the record's wording and the definition used for each, the pipeline, the control policy, what will be reported and the tests |
| `wrappers_proxy.py` | one training wrapper and one evaluation wrapper per condition: baseline, 600 epochs, beat-time jitter (SD 0.5 and 1.0 sample), pulse carrier (50 and 150 Hz), tracking distance (5, 10, 15, 20 mm), window length (50 and 150 samples, with the first dense layer resized), stiffness set 5.4 / 7.3 / 11.4 kPa, clean-only training. The window generator is a copy of the battery's `generate_degraded_windows_shifted` with the changed parameters added; with no shift it gives the same draws as the original. |
| `harness_proxy.py` | eight seeds (101 to 108); for every condition the real model and its 200 shuffled-label controls are retrained (controls: same data with shuffled labels, same number of epochs), 23-depth grid, 300 windows per depth, BH correction, floor = first depth that is not significant, 15.0 cm = fallback cap (censored). `--seeds` takes a list and `--threads` sets the thread count; each seed writes its own `proxy_floors_<seed>.csv`, `proxy_depth_detail_<seed>.csv` and `proxy_run_<seed>.log` (start and end times, platform), so parallel processes never share a file, and a restart skips finished (seed, condition) rows. `--list` prints every condition and what it retrains. Default output folder: `audit_fixes/cloud/proxy_sweeps/results/`. |
| `summarize_proxy.py` | merges the per-seed files; per condition the eight floors, mean, SD, median, censored count and seeds at 0.00 cm; the Wilcoxon rank-sum (primary) and paired signed-rank (secondary) tests of clean-only against mixed-depth; and the consistency checks of `PREREG.md` against the robustness battery, the certified sweep and the clean-trained sweep (reported, never used to adjust anything). |
| `calibration_24_models.py`, `calibration_24_models.csv` | evaluation-only calibration gap of the 24 degradation-aware networks (60 windows per depth, 27.05 dB, fixed evaluation draw) before and after a scoped temperature scaling re-implemented from the description in Log 23. The weights are not in the release; the script needs `--weights-dir`. |
| `summary_table.py` | writes `summary_by_condition.csv` (mean, SD, median, min, max, censored count and seeds at 0.00 per condition; a mean with a censored seed is marked as a lower bound). |
| `diagnose_zero_floors.py` | writes `diagnostic_zero_floors.csv`: for each floor of 0.00 cm, accuracy and p-values at the first four depths and a cause fixed by a rule declared in the script. |

The floors are proxy floors, not certified floors.

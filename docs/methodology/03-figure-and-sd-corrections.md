# 03. Figure export and the SD correction

Three changes: the figure regeneration, the reliability diagram and the floor SD.

## What was done

`Source/make_figures.py` builds 8 figures from saved CSVs and pickles; the rest are copies of Stage 1 to 4 outputs. `outputs/figures_earlier/manifest.md` lists the source data per figure. Two Stage 4 figures had to be regenerated, without retraining, after the noise correction was traced to the provenance re-certification. The clean CNN checkpoint was confirmed bit-identical to a fresh retrain under current code.

Bugs found in figure review:

1. **Reliability diagram.** The UCE values (0.356 fixed-width, 0.326 equal-mass) were current. The domain-of-validity line on the figure still said 9.0 cm from before the gate fix. The old UCE numbers in the record text (0.2266/0.2838) reproduced from the pre-Fix-1 calibration CSV, which confirmed those were stale Log 16 values.
2. **Paired-trajectory figure.** The baseline-fold filter matched any truthy `baseline_acc`, which pulled the CSV's summary "mean" and "sd" rows in as folds 6 and 7. The SD (0.0208) was plotted as an accuracy point at about 0.625. Filter changed to `fold.isdigit()`; corrected baseline mean is 0.7270.
3. **Floor SD.** The 8.48 ± 1.77 cm figure came from a hand-coded population variance (divide by n). Recomputed from `Source/logs/degradation_aware_final_results.csv` the sample SD is 1.8122 cm (ddof=1) against 1.7740 cm (ddof=0). The write-up text had been fixed earlier the same day and the repo had not. 16 stale occurrences (6 README, 10 across 8 scripts) were corrected. The 1.77 Pa·s cirrhosis viscosity is a different quantity and was left alone.

Lane 1 and Lane 7 (04, 10) independently re-derived 8.4754 and 1.8122 from the CSV the next day.

## Fig. 1 script

`Source/make_fig1_windowing.py` draws the beat-anchored windows for Fig. 1: one 100-sample (0.2 s) window per heartbeat trigger on a healthy-class signal from `physics_model.generate_signal`, cut the way `cnn_classifier.extract_raw_windows` cuts them. It writes `outputs/figures_earlier/fig3_01_beat_anchored_windows.png`. Both files were made after the work above and are not part of it. They were untracked until the reorganization. Neither is listed in the original manifest; a row was added there.

## Reproducibility

`python Source/make_figures.py`; `python calibration_analysis.py` for the reliability diagram.

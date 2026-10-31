# 05. Lane 2 and Q6: statistical rechecks

Follows from the grouped-CV fix (02). Results appear in write-up Appendix B (Nadeau-Bengio row, Cohen's d row).

## Lane 2

Question: do the reported p-values, effect sizes and gate statistics reproduce from the saved per-fold data, and does the repeated-CV significance survive variance correction?

Method: `audit_fixes/lane2/analysis.py` reads the grouped-CV CSVs in `Source/logs/` and the 24 certified floors. No retraining.

- Single-split paired t-tests (n=5 folds): all three p-values reproduce (0.0175, 0.0078, 0.2418 vs record 0.242). Cohen's d does not for two of three: computed 1.747 and 2.213 (d_z with paired-difference SD, ddof=1) against the stated 1.54 and 2.76. Two other d conventions did not produce those either. The source of the old numbers was never identified. The write-up ledger replaced them with d_z = t/sqrt(5) (1.75/2.21/0.61).
- Nadeau-Bengio corrected resampled t (75 differences, 15x5 CV): CNN vs baseline p=0.0036, ResNet1D vs baseline p=0.000001, ResNet1D vs CNN p=0.252. The earlier p=0.0002 was an uncorrected sign-flip test. Applied to the 5-fold split, CNN vs baseline gives p=0.0598 (not significant), ResNet1D vs baseline 0.030.
- 24 floors: mean 8.4754, SD 1.8122 (ddof=1), median 8.00, 10th percentile 6.50, 5th 5.65, all matching. Bootstrap 95% CI of the 10th percentile [5.47, 7.50]; leave-one-out range [6.50, 6.60]. 2/24 below 6.50 cm (Clopper-Pearson [0.010, 0.270]). The 6.50 cm value needs numpy-linear or nearest-rank interpolation; the Weibull (n+1)p rule gives 6.00.
- `anomalous_recovery`: 10 of 24 seeds (103, 105, 106, 108, 109, 115, 116, 117, 123, 124), matching the record.
- Per-depth p-values from the 24-seed sweep were never saved, so no per-seed significance table could be built. Lane 5 (09) added that logging for its rerun.

## Q6: BH-FDR family sensitivity

Question: how much does the clean-CNN floor depend on which tests are pooled for BH-FDR?

Method: `audit_fixes/q6_bh_fdr_families/q6_recompute_bh_fdr.py` re-implements BH step-up and applies it to the saved p-values under three families.

| family | n | rejected | floor |
|---|---|---|---|
| pooled 82 cells (4 frequencies) | 82 | 23 | 100 Hz: 4.30 cm |
| 100 Hz only | 24 | 0 | 0.00 cm |
| 23 depths at 27.05 dB | 23 | 0 | 0.00 cm |

The pooled family reproduces the manifest's 4.30 cm. The narrower families give 0.00 because with 200 shuffles the smallest attainable p is 1/201 = 0.00498, and the rank-1 BH threshold for 23 tests is 0.05/23 = 0.00217. No cell can pass whatever the real accuracy. This is a resolution limit of the test, and it comes back in Lane 5 (09) and in the battery runs (12) as floors of exactly 0.00 cm.

## Reproducibility

`python audit_fixes/lane2/analysis.py`; `python audit_fixes/q6_bh_fdr_families/q6_recompute_bh_fdr.py`. Reports in the same directories.

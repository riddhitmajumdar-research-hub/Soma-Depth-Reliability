# 07. Lane 4: SVM, random forest and GRU baselines on the clean task

## Question

Do other model families match the CNN and ResNet1D on the clean 3-class task (90 realizations, grouped 5-fold, same fold indices as `repeated_grouped_cv_log11.py`)?

## v1 (PREREG.md written before the run)

SVM-RBF and random forest on FFT-magnitude features, small GRU on raw windows, at most 4 hyperparameter configs each. 15x repeated grouped 5-fold (75 folds): SVM 47.98%, RF 48.41%, GRU 24.15%. Single split: 48.61 / 49.44 / 22.22. The GRU was below the 33.3% no-information rate and this was reported as run.

## Diagnosis (`diagnose_v1.py`)

The v1 inner hyperparameter CV used plain StratifiedKFold, so windows from one realization could sit on both sides of an inner split. The outer split was grouped correctly. The GRU had collapsed to a constant single-class prediction in all 5 folds (72 of 72 test windows in one class every time). SVM and RF were not collapsed; their low scores followed from magnitude-only features discarding arrival-time (phase) information.

## v2 (PREREG_v2.md written before the v2 run)

- SVM/RF features: raw window plus real and imaginary rfft (202 features).
- GRU: per-window standardization, gradient clipping at 5.0, restart on train-accuracy collapse (up to 5).
- Inner CV changed to StratifiedGroupKFold for all three models. The amendment asked for this on the GRU only; the same defect was in SVM and RF, so it was applied to all and flagged.

Result, 75 folds: SVM 90.50% ± 3.81, RF 91.72% ± 3.32, GRU 24.33% ± 5.41 (no change). The restart loop hit its limit of 5 in most folds (355 restarts over 80 fold-trainings). The GRU result is unexplained; the small size (1 layer, 16 units, 60 epochs) is the leading suspect and was not tested. This run took about 7100 s.

## Nadeau-Bengio correction

The v2 CSVs were not put through the Lane 2 correction when they were written. That was done afterwards, without retraining, using `nadeau_bengio_test` from `audit_fixes/lane2/analysis.py` unchanged (75 paired differences, n_test/n_train = 1/4, df = 74). The v2 folds use the same `grouped_kfold_indices` and `rep_seed` as the Log 11 CNN and ResNet1D runs, and the (repetition, fold) rows line up one to one, so the differences are paired.

| Comparison | Mean diff | t | p | 95% CI |
|---|---|---|---|---|
| SVM vs logistic baseline | +19.56 pp | 7.06 | 7.7e-10 | 14.04 to 25.08 |
| RF vs logistic baseline | +20.78 pp | 6.42 | 1.2e-08 | 14.33 to 27.24 |
| SVM vs CNN | +5.22 pp | 1.43 | 0.156 | -2.04 to 12.49 |
| RF vs CNN | +6.44 pp | 2.01 | 0.048 | 0.06 to 12.83 |
| SVM vs ResNet1D | +0.39 pp | 0.17 | 0.869 | -4.30 to 5.07 |
| RF vs ResNet1D | +1.61 pp | 0.56 | 0.579 | -4.15 to 7.37 |
| RF vs SVM | +1.22 pp | 0.71 | 0.478 | -2.19 to 4.64 |
| GRU vs logistic baseline | -46.61 pp | -12.58 | 4.7e-20 | -53.99 to -39.22 |

Mean accuracies over the 75 folds: baseline 70.94, CNN 85.28, ResNet1D 90.11, SVM 90.50, RF 91.72, GRU 24.33. SVM and RF clearly beat the logistic baseline. Neither is distinguishable from ResNet1D. SVM is not distinguishable from the CNN; RF vs CNN sits at the edge (p = 0.048, interval lower bound 0.06 pp) and should not be read as a firm difference, particularly with eight comparisons run here and no multiplicity adjustment. The GRU result is a training failure (below the 33.3% no-information rate), not a finding about GRUs.

## Link

The v2 SVM/RF result is consistent with the offset finding in Lane 6 (08): phase and timing carry the class information on clean signals.

## Reproducibility

`python audit_fixes/lane4/run_lane4_comparison.py` (v1), `diagnose_v1.py`, `run_lane4_v2_comparison.py`. Threads capped at 2. Raw logs and CSVs in `audit_fixes/lane4/`. `python audit_fixes/lane4/nb_v2.py` applies the Lane 2 Nadeau-Bengio function to the saved v2 CSVs (output in `nb_v2_output.txt`).

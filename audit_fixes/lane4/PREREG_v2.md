# Lane 4 Pre-registration v2 — amendment

Written and frozen before any v2 model is trained. v1 outputs
(`lane4_single_split.csv`, `lane4_repeated_grouped_cv_raw.csv`, `REPORT.md`,
`run_stdout.log`) are **not modified or deleted** — v1 stands as a documented
negative result. This document records only what changes for v2 and why.

## Step 1 diagnostic findings (from `diagnose_v1.py`, single grouped 5-fold split, rep_seed=1000)

Full output: `diagnose_v1_stdout.log`, `diagnose_v1_output.csv`.

1. **Inner-CV grouping — confirmed NOT grouped.** v1's `svm_inner_cv_select`,
   `rf_inner_cv_select`, and `gru_inner_cv_select` all call
   `StratifiedKFold(...).split(X_train, y_train)` with no `groups` argument
   (`run_lane4_comparison.py:123,144,163`). Inner-CV splits are stratified by
   label only; windows from the same realization (`group_id`) can appear on
   both sides of an inner train/validation split. The OUTER 5-fold split used
   for final scoring was correctly grouped throughout v1 — this issue is
   confined to hyperparameter *selection*, not the reported test-fold scores.
2. **The GRU collapsed to a constant single-class predictor in every one of
   the 5 folds.** Per-fold train accuracy exactly equals the train-fold's
   class proportion for one class, and every test-fold predicted-class
   histogram is 100% one class (72/72 windows), e.g. fold 1: all 72
   predictions are `significant_fibrosis`; fold 3: all 72 are `cirrhosis`.
   In 4 of 5 folds the collapsed class matches the train-fold majority class;
   in fold 2 it collapses to `cirrhosis` (train proportion 100/288=0.3472,
   matching the observed train accuracy 0.3472) even though
   `significant_fibrosis` was the train majority — consistent with the GRU
   converging to whichever single class its random init happened to favor,
   not necessarily the majority class. This is why the reported v1 GRU
   accuracy fell below the 33.3% no-information rate: it did worse than even
   a majority-class constant predictor in every fold except where the
   collapsed class happened to coincide with the test fold's plurality.
3. SVM and RF test-fold predicted-class histograms (`diagnose_v1_stdout.log`)
   are **not** collapsed — predictions are spread across all three classes in
   every fold, roughly matching test-fold class proportions. Their
   underperformance relative to CNN/ResNet1D/logistic-regression is not a
   collapse artifact; per amendment rationale (a), it is attributed to the
   FFT-magnitude feature set discarding time-shift (arrival-time) information
   that separates the classes.

## v2 changes (deviations from v1, with reasons)

### (a) SVM / RF feature set

**v1 used FFT-magnitude only**, which is invariant to time shift — exactly
the property that separates classes in this task (arrival-time encodes
stiffness via time-of-flight). v1's SVM/RF results stand as a documented
negative result explaining *why* magnitude-only features underperform, not as
a discarded run.

**v2 feature set**: concatenation of
  (i) the raw 100-sample window (unchanged, time-domain, preserves shift), and
  (ii) the real and imaginary parts of the full `rfft` (51 complex bins -> 102
  real values, preserves phase/arrival-time information magnitude-only
  discards).
Total v2 feature vector length: 100 + 102 = 202. `StandardScaler` fit on
TRAIN-fold features only, as in v1. SVM/RF model families, grid structure
(<=4 configs each), and inner-CV procedure otherwise unchanged in kind from
v1 (grid values below).

### (b) GRU

Step 1 confirms the collapse-to-constant-predictor condition, so v2 adds,
uniformly across the same 4-config grid used in v1 (`lr in {0.001, 0.01} x
dropout in {0.0, 0.3}` — grid itself is NOT re-tuned, only the training
procedure around it changes):

1. **Per-window standardization**: each raw window is standardized
   (zero-mean, unit-variance, `eps=1e-8` in the denominator) before being fed
   to the GRU. v1 fed raw, unnormalized amplitude values.
2. **Gradient clipping**: `torch.nn.utils.clip_grad_norm_(max_norm=5.0)`
   applied every optimizer step. Fixed value, not tuned.
3. **Train-accuracy collapse check with restart**: after training, if the
   final train-fold accuracy is <= the train-fold's own majority-class
   constant-predictor accuracy (i.e., the model did no better than always
   predicting the single most common class), retrain from a fresh random
   initialization, up to `max_restarts=5`, mirroring the existing restart
   safeguard already used for ResNet1D in
   `Source/repeated_grouped_cv_log11.py`'s `train_resnet_with_restarts`
   (same codebase precedent, reused in kind — not a new mechanism invented
   for this lane). The number of restarts used is logged per fold, exactly as
   the existing ResNet1D script logs `total_restarts`.

### (c) Inner-CV grouping — applied to all three models, not just the GRU

The amendment asks specifically for the GRU's inner CV to be
"grouped inner CV only." Step 1 found the ungrouped-inner-CV issue applies
identically to SVM and RF's inner CV. Fixing it only for the GRU while
leaving the same confirmed issue live for SVM/RF would leave a known problem
undocumented and unaddressed for 2 of 3 models in the same v2 run. v2
therefore switches all three models' inner-CV selectors from
`StratifiedKFold` to `StratifiedGroupKFold(n_splits=3)` (sklearn >=1.1,
confirmed available: sklearn 1.7.2), keyed on the TRAIN fold's own
`group_id` values. This is flagged here as a deviation beyond the literal
amendment text, not silently applied.

## What does NOT change from v1

- Outer split: same `grouped_kfold_indices` (imported from
  `Source/repeated_grouped_cv_log11.py`), same `rep_seed` values
  (1000 for single-split, `1000+rep` for 15x), same `N_FOLDS=5`,
  `N_REPETITIONS=15`.
- `n_jobs=2`, `torch.set_num_threads(2)`.
- CSV output format: `fold, svm_acc, rf_acc, gru_acc` (single-split) and
  `repetition, fold, svm_acc, rf_acc, gru_acc` (15x-repeated) — same column
  pattern as v1, written to new v2-suffixed files so v1 outputs are untouched.
- SVM grid: `C in {1.0, 10.0} x gamma in {"scale", 0.01}` (4 configs) —
  unchanged from v1.
- RF grid: `max_depth in {None, 10} x min_samples_leaf in {1, 4}`
  (4 configs), `n_estimators=300` — unchanged from v1.
- GRU grid: `lr in {0.001, 0.01} x dropout in {0.0, 0.3}` (4 configs),
  1 layer, 16 hidden units — unchanged from v1 (only the training procedure
  around each config changes, per (b) above).
- Reporting commitment: v1 and v2 both reported side by side in REPORT.md;
  no cherry-picking; v2 results reported whatever they are, including if the
  fixes do not help.

## Reporting commitment

v2's SVM/RF/GRU results will be reported beside v1's, with v1 kept intact as
a documented negative result explaining what the FFT-magnitude and
unstandardized/uncontrolled-GRU choices actually caused. No number is
retracted or silently replaced.

## Amendments

(none yet — this is PREREG for v2 itself, amending v1's PREREG.md)

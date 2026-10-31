# Lane 4 Report v2 -- amendment: diagnostics + feature/GRU fixes

v1 outputs and REPORT.md are unchanged and stand as a documented negative
result. This report adds Step 1 diagnostics and the v2 (amended) run, and
presents v1 and v2 side by side as the amendment specifies.

## What was done

1. Step 1 (diagnostics only, no tuning) -- diagnose_v1.py, raw output
   diagnose_v1_stdout.log / diagnose_v1_output.csv. Reran the exact v1
   single-split protocol (same fold splits, same grids, same feature sets) to
   capture per-fold train accuracy, test predicted-class histograms, test-fold
   ground-truth class counts, and train-majority constant-predictor accuracy --
   none of which v1 had saved.
2. Step 2 -- PREREG_v2.md, written before any v2 model was trained,
   documenting the Step 1 findings and the exact deviations made in response
   (full rationale there; summarized below).
3. Ran run_lane4_v2_comparison.py. Raw stdout: run_v2_stdout.log. Same outer
   split (grouped_kfold_indices, same seeds), same n_jobs=2 /
   torch.set_num_threads(2), same CSV column layout as v1, written to new
   v2-suffixed files:
   - lane4_v2_single_split.csv
   - lane4_v2_repeated_grouped_cv_raw.csv (75 rows, 15 reps x 5 folds)

No file outside audit_fixes/lane4/ was written; v1's files were not touched.

## Step 1 diagnostic findings

Inner-CV grouping -- confirmed NOT grouped in v1. All three of v1's inner-CV
selectors call StratifiedKFold(...).split(X_train, y_train) with no `groups`
argument (run_lane4_comparison.py lines 123, 144, 163). Windows from the same
realization could land on both sides of an inner train/validation split
during hyperparameter selection. The v1 OUTER 5-fold split (used for the
reported test-fold scores) was correctly grouped throughout -- this gap was
confined to hyperparameter selection.

The v1 GRU collapsed to a constant single-class predictor in all 5 folds of
the single split (per-fold detail below, from diagnose_v1_output.csv):

| Fold | Train majority class | Majority-baseline acc | GRU train acc | GRU test acc | GRU test-fold predictions |
|---|---|---|---|---|---|
| 1 | significant_fibrosis | 0.2222 | 0.3611 | 0.2222 | all 72 -> significant_fibrosis |
| 2 | significant_fibrosis | 0.2778 | 0.3472 | 0.2778 | all 72 -> cirrhosis |
| 3 | cirrhosis | 0.1667 | 0.3750 | 0.1667 | all 72 -> cirrhosis |
| 4 | healthy | 0.1667 | 0.3750 | 0.1667 | all 72 -> healthy |
| 5 | healthy | 0.2778 | 0.3472 | 0.2778 | all 72 -> healthy |

Train accuracy in every fold exactly equals the train-fold's own class
proportion for the single class the model collapsed to, and the test-fold
predicted-class histogram is 100% one class (72/72 windows) in every fold.
In 4 of 5 folds the collapsed class matches the train-fold majority class;
fold 2 collapsed to a non-majority class (cirrhosis, train proportion
0.3472) even though significant_fibrosis was the majority -- consistent with
the GRU locking onto whichever class its random init favored, not
necessarily the true majority. This is why v1's reported GRU accuracy fell
below the 33.3% no-information rate: the model did worse than even a
majority-class constant predictor in every fold except where the collapsed
class happened to coincide with the test fold's plurality (folds 1, 3, 4).

v1 SVM/RF were not collapsed. Test-fold predicted-class histograms
(diagnose_v1_output.csv) were spread across all three classes in every fold,
roughly tracking test-fold class proportions -- their underperformance is
attributed to the FFT-magnitude feature set discarding time-shift
(arrival-time) information, not a training failure.

## v2 changes (see PREREG_v2.md for full rationale)

- SVM/RF features: raw 100-sample window concatenated with real+imag parts
  of the full rfft (100 + 102 = 202 features), replacing v1's FFT-magnitude-
  only features. Restores the arrival-time information magnitude-only
  discarded.
- GRU: per-window standardization, gradient clipping (max_norm=5.0), and a
  train-accuracy collapse check with restart (up to max_restarts=5,
  mirroring the existing ResNet1D restart safeguard already used elsewhere
  in the codebase). Grid itself (lr in {0.001,0.01} x dropout in {0.0,0.3})
  unchanged.
- Inner-CV grouping: switched from StratifiedKFold to
  StratifiedGroupKFold(n_splits=3) for ALL THREE models (SVM, RF, and GRU)
  -- the amendment asked for grouped inner CV for the GRU specifically, but
  Step 1 found the same ungrouped-inner-CV issue applies identically to
  SVM/RF, so it was fixed for all three and flagged here as a deviation
  beyond the literal amendment text, not silently applied.

## Exact numbers -- v1 and v2 side by side

### Single grouped 5-fold split (rep_seed=1000)

| Model | v1 mean +/- SD | v2 mean +/- SD |
|---|---|---|
| SVM | 48.61% +/- 5.56% | 90.56% +/- 4.95% |
| Random forest | 49.44% +/- 4.35% | 91.11% +/- 2.88% |
| Small GRU | 22.22% +/- 5.56% | 22.22% +/- 5.56% (unchanged) |

(Existing baseline/CNN/ResNet1D for reference, not recomputed: 72.70% / 89.81% / 86.11%.)

### 15x repeated grouped 5-fold (n=75 paired fold accuracies)

| Model | v1 mean +/- SD | v2 mean +/- SD |
|---|---|---|
| SVM | 47.98% +/- 4.55% | 90.50% +/- 3.81% |
| Random forest | 48.41% +/- 4.45% | 91.72% +/- 3.32% |
| Small GRU | 24.15% +/- 4.70% | 24.33% +/- 5.41% (unchanged within noise) |

Full per-fold and per-repetition values: lane4_v2_single_split.csv,
lane4_v2_repeated_grouped_cv_raw.csv.

## What the v2 numbers show

- SVM and RF improved dramatically (v1 ~48-49% -> v2 ~90-92%), now in the
  same range as the existing CNN (89.81%) and ResNet1D (86.11%), and above
  the existing logistic-regression baseline (72.70%). This is consistent
  with amendment rationale (a): the arrival-time / phase information the
  FFT-magnitude-only features discarded was in fact the dominant signal for
  this task.
- The GRU did NOT improve (v1 22.22%/24.15% -> v2 22.22%/24.33%, differences
  within noise) despite all three fixes. run_v2_stdout.log shows the
  restart-on-collapse mechanism engaged in the large majority of folds
  (restarts=5, i.e. exhausted all 5 attempts) across both the single split
  and the 15x-repeated run (355 total restarts used across 80
  fold-trainings in the 15x-repeated run alone) -- meaning in most folds
  even 5 fresh random initializations never produced a model that beat its
  own train-fold's majority-class constant predictor. Where a fold succeeded
  without exhausting restarts (restarts=0), test accuracy was still modest
  (e.g. 0.2917, 0.3194, 0.3056, 0.5000 -- visible in run_v2_stdout.log), not
  reliably strong. This is reported as-is, unresolved: the evidence points
  to the architecture itself (1 layer, 16 hidden units, 60 epochs, ~4
  windows/realization x ~72 train realizations per fold) being too small or
  too briefly trained to learn this task from raw windows, rather than the
  three specific failure modes the amendment targeted (unnormalized
  amplitude, exploding gradients, or unlucky initialization alone) -- but
  this interpretation was not tested further, since doing so would mean
  widening the grid or architecture, which is out of scope for this
  pre-registered amendment.

## Open questions

1. Whether a still-larger GRU fix (more hidden units, more epochs, a
   different architecture entirely) would resolve the collapse is left open
   -- out of scope for this amendment's fixed protocol.
2. Whether the v2 SVM/RF numbers (~90-92%) constitute a real capacity
   advantage over the existing logistic-regression baseline (72.70%),
   comparable to or exceeding CNN/ResNet1D, is a significance-testing
   question for Lane 2's Nadeau-Bengio script, not resolved here. The v2
   CSVs are laid out in the same column pattern as v1's (and as
   Source/logs/log11_repeated_grouped_cv_raw.csv) so that script can be
   pointed at them directly.
3. Runtime: v2 single-split + 15x-repeated combined took ~7129s (~119 min),
   substantially longer than v1's ~2000s (~33 min), driven by the GRU's
   restart-on-collapse loop running inside grouped inner CV (up to
   5 restarts x 4 grid configs x 3 inner folds per outer fold).

## File list

- PREREG_v2.md
- diagnose_v1.py, diagnose_v1_stdout.log, diagnose_v1_output.csv
- run_lane4_v2_comparison.py, run_v2_stdout.log
- lane4_v2_single_split.csv
- lane4_v2_repeated_grouped_cv_raw.csv
- REPORT_v2.md -- this file
- v1 files, unchanged: PREREG.md, run_lane4_comparison.py, run_stdout.log,
  lane4_single_split.csv, lane4_repeated_grouped_cv_raw.csv, REPORT.md

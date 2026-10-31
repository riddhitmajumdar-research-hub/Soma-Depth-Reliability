# Lane 4 Pre-registration — Additional baseline architectures

Written and frozen **before** any model is trained. Deviations after this point
must be logged as amendments at the bottom of this file, not silently applied.

## Protocol reused unmodified from the existing comparison

Source: [Source/repeated_grouped_cv_log11.py](../../Source/repeated_grouped_cv_log11.py),
paired with [Source/logs/log11_grouped_paired_ttests.csv](../../Source/logs/log11_grouped_paired_ttests.csv)
(single-split) and `logs/log11_repeated_grouped_cv_raw.csv` (15x version).

- Data: clean (noiseless) synthetic signals, `physics_model.generate_signal`,
  `sample_rate=500`, `SEED = 42` (`Source/constants.py`).
- 90 realizations = 30 runs/class x 3 classes (`healthy`, `significant fibrosis`,
  `cirrhosis` per `stiffness_classes` in `physics_model.py`), grouped by
  realization (`group_id`).
- Raw windows: `cnn_classifier.extract_raw_windows`, `window_size=100` samples
  (200ms @ 500Hz), ~4 windows/realization.
- Split generator (reused verbatim, imported not reimplemented):
  `grouped_kfold_indices(groups, n_folds, rep_seed)` from
  `repeated_grouped_cv_log11.py` — group IDs shuffled with
  `np.random.default_rng(rep_seed)`, `fold_of_group[g] = i % n_folds`.
- `N_FOLDS = 5`. Single-repetition run uses `rep_seed = 1000` (repetition 0 of
  the existing 15x loop, so its folds are identical to the first repetition
  already on disk). 15x run uses `rep_seed = 1000 + rep` for `rep in range(15)`,
  identical to the existing script.
- Existing headline numbers this lane is compared against (single grouped
  5-fold split, from README.md): baseline logistic regression **72.70%**,
  CNN **89.81%**, ResNet1D **86.11%**.

## New models (fixed before running)

All three are trained/selected using **only TRAIN-fold data** for both
preprocessing (scaler fit) and hyperparameter selection (inner CV). Test folds
are touched only for final scoring. Metric: fold accuracy (mean over held-out
windows in that fold), matching the existing comparison's metric.

### (a) SVM — RBF kernel on FFT-magnitude features

- Feature extraction: per raw window (length 100 @ 500Hz), real FFT
  (`np.fft.rfft`), magnitude spectrum, bins up to Nyquist (250Hz) →
  51 features/window (`window_size // 2 + 1`).
- Preprocessing: `StandardScaler` fit on train-fold features only.
- Model: `sklearn.svm.SVC(kernel="rbf")`.
- Hyperparameter grid (<=4 configs), selected by 3-fold inner CV on the
  TRAIN fold only, scored by accuracy:
  1. `C=1.0, gamma="scale"`
  2. `C=10.0, gamma="scale"`
  3. `C=1.0, gamma=0.01`
  4. `C=10.0, gamma=0.01`

### (b) Random forest — 300 trees, same FFT-magnitude features

- Same feature extraction and scaling as (a) (tree models don't need scaling,
  but we reuse the same scaled features for a controlled, identical input
  across (a)/(b) rather than introduce an unscaled variant).
- Model: `sklearn.ensemble.RandomForestClassifier(n_estimators=300, random_state=SEED)`.
- Hyperparameter grid (<=4 configs), selected by 3-fold inner CV on TRAIN only:
  1. `max_depth=None, min_samples_leaf=1`
  2. `max_depth=10, min_samples_leaf=1`
  3. `max_depth=None, min_samples_leaf=4`
  4. `max_depth=10, min_samples_leaf=4`

### (c) Small GRU — 1 layer, 16 hidden units, on raw windows

- Input: raw window, shape `(batch, seq_len=100, input_size=1)` (no FFT).
- No feature scaling beyond what the existing CNN/ResNet1D pipeline already
  applies (raw signal values), for parity with those two.
- Model: `nn.GRU(input_size=1, hidden_size=16, num_layers=1, batch_first=True)`
  -> take final hidden state -> `nn.Linear(16, 3)`.
- Optimizer: `torch.optim.Adam`. Loss: `nn.CrossEntropyLoss`.
- Hyperparameter grid (<=4 configs), selected by 3-fold inner CV on TRAIN only,
  scored by inner-validation accuracy after a fixed 60 epochs (matching
  `train_cnn`'s epoch count in the existing script, for parity):
  1. `lr=0.001, dropout=0.0`
  2. `lr=0.001, dropout=0.3`
  3. `lr=0.01, dropout=0.0`
  4. `lr=0.01, dropout=0.3`
  - dropout applied to the GRU's final hidden state before the linear layer.

## Thread limits (per the protocol)

`n_jobs=2` for all sklearn grid/CV; `torch.set_num_threads(2)` before any GRU
training.

## Reporting commitment

All three models in this list will be reported in RESULTS regardless of
outcome — no post-hoc exclusion.

Per-fold accuracies saved as `audit_fixes/lane4/lane4_repeated_grouped_cv_raw.csv`
with columns `repetition, fold, svm_acc, rf_acc, gru_acc` — matching the column
pattern of `Source/logs/log11_repeated_grouped_cv_raw.csv`
(`repetition, fold, baseline_acc, cnn_acc, resnet1d_acc`), confirmed by reading
`audit_fixes/lane2/analysis.py` (uncommitted, found in the working tree), whose
`nadeau_bengio_test(diffs)` function operates on a plain array of paired
per-fold-differenced accuracies read from exactly this column layout. Lane 2's
function can be applied to lane4's output by substituting column names, with
no changes to Lane 2's file.

The single-repetition run additionally writes
`audit_fixes/lane4/lane4_single_split.csv` with columns
`fold, svm_acc, rf_acc, gru_acc`, mirroring
`Source/logs/log05_log11_grouped_cv_verification.csv`'s `item, fold, *_acc` rows
for the `log11_paired_grouped_cv` item (the single-split table Lane 2 reads
into `single_split`).

## Amendments

(none yet)

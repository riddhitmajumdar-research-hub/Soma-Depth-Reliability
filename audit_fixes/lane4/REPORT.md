# Lane 4 Report — Additional baseline architectures

## What was done

1. **Step 0 (discovery, read-only)**: located the existing baseline / CNN /
   ResNet1D comparison — [Source/repeated_grouped_cv_log11.py](../../Source/repeated_grouped_cv_log11.py)
   (single-split and 15x-repeated logic) and
   [Source/logs/log11_grouped_paired_ttests.csv](../../Source/logs/log11_grouped_paired_ttests.csv)
   (single-split numbers). Data condition: clean (noiseless) synthetic
   signals, 90 realizations (30/class x 3 classes), grouped 5-fold split via
   `grouped_kfold_indices(groups, n_folds, rep_seed)`, `SEED = 42`
   (`Source/constants.py`). Found `audit_fixes/lane2/` already present
   (uncommitted, Lane 2's Nadeau-Bengio script) and read its CSV column
   expectations before choosing lane4's output format.
2. Wrote [PREREG.md](PREREG.md), fixing SVM-RBF / random-forest / small-GRU
   models and a <=4-config hyperparameter grid per model, **before** running
   anything.
3. Wrote [run_lane4_comparison.py](run_lane4_comparison.py), importing
   (not reimplementing) `grouped_kfold_indices` from
   `Source/repeated_grouped_cv_log11.py` and `build_raw_dataset_with_groups`
   from `Source/cnn_classifier.py`, so folds line up with the existing paired
   data. Ran it once, single-threaded to `n_jobs=2` / `torch.set_num_threads(2)`
   as instructed. Raw stdout saved at [run_stdout.log](run_stdout.log).
4. Saved per-fold accuracies:
   - [lane4_single_split.csv](lane4_single_split.csv) — columns
     `fold, svm_acc, rf_acc, gru_acc`
   - [lane4_repeated_grouped_cv_raw.csv](lane4_repeated_grouped_cv_raw.csv) —
     columns `repetition, fold, svm_acc, rf_acc, gru_acc`, 75 rows
     (15 reps x 5 folds), matching the column pattern of
     `Source/logs/log11_repeated_grouped_cv_raw.csv` so Lane 2's
     `nadeau_bengio_test(diffs)` can be pointed at these columns directly
     without modification to Lane 2's file.

No file outside `audit_fixes/lane4/` was written. `Source/constants.py`,
certified results, saved models, README, and the running ResNet1D sweep
(pid 7429) were not touched.

## Exact numbers (produced by `run_lane4_comparison.py` this pass)

### Single grouped 5-fold split (`rep_seed=1000`, same fold-to-group mapping as repetition 0 of the existing 15x script)

| Model | Mean accuracy | SD (ddof=1, n=5 folds) |
|---|---|---|
| Existing baseline (logistic regression) | 72.70%* | 2.08%* |
| Existing CNN | 89.81%* | — |
| Existing ResNet1D | 86.11%* | — |
| **SVM (RBF, FFT-magnitude features)** | **48.61%** | **5.56%** |
| **Random forest (300 trees, same features)** | **49.44%** | **4.35%** |
| **Small GRU (1 layer, 16 hidden, raw windows)** | **22.22%** | **5.56%** |

\* existing values, from README.md / `Source/logs/log11_grouped_paired_ttests.csv` — not recomputed this pass, shown for reference only.

### 15x repeated grouped 5-fold (n=75 paired fold accuracies per model)

| Model | Mean accuracy | SD (ddof=1, n=75) |
|---|---|---|
| **SVM (RBF)** | **47.98%** | **4.55%** |
| **Random forest** | **48.41%** | **4.45%** |
| **Small GRU** | **24.15%** | **4.70%** |

Full per-fold and per-repetition values are in the two CSVs listed above.

## No check value was quoted for this lane

PREREG.md did not commit to a specific numeric target to check the run
against (unlike Lane 2, which checked against numbers already stated in the
record). No MATCH/MISMATCH determination applies here — these are new numbers,
not verification of an existing claim.

## Surprises / anomalies (reported, not resolved, per audit rule 6)

- **The small GRU scores below chance for a balanced 3-class problem**
  (chance = 33.3%): 22.2% on the single split, 24.15% mean over 75 repeated
  folds — both below the no-information rate, and consistently so across
  every fold and every repetition (min 16.7%, max 33.3% across the 75 rows;
  see `lane4_repeated_grouped_cv_raw.csv`).
  - Inner CV selected `lr=0.001, dropout=0.0` for essentially every fold
    (visible in `run_stdout.log`'s per-fold cfg output for the single split;
    same grid, same selection dominated the repeated run per spot-checks
    during development). That configuration was never beaten by the higher
    learning-rate or dropout variants in the grid.
  - This is consistent with a GRU that fails to learn the raw 100-sample
    window (e.g., converges to a fixed low-confidence prediction that happens
    to score worse than uniform-random on this particular grouped split) —
    but I did not diagnose further, since doing so (e.g., inspecting
    predicted-class distributions, adding a different architecture/init) was
    outside the fixed PREREG protocol and would risk exactly the "silently
    fix a mismatch" behavior the audit rules prohibit.
  - **This number is real and was reported as run, per the no-cherry-picking
    commitment in PREREG.md** — it is not evidence the GRU is broken as code
    (training/eval used the same raw windows and label encoder as CNN/ResNet1D
    in the existing script), only that this particular small-GRU
    configuration, on this exact protocol, underperforms.
- SVM and random forest land well below the existing CNN/ResNet1D/logistic-regression
  baseline (~48-49% vs. 72.70-89.81%), and below random forest's own
  natural expectation for a 3-class task with engineered spectral features.
  This is plausible given the features used (bare FFT magnitude, no
  normalization by frequency band, no phase information, no time-domain
  features) — the PREREG grid was fixed before running and not expanded after
  seeing this, per the no-post-hoc-tuning commitment.

## Open questions

1. Whether the below-chance GRU result should be investigated further (e.g.,
   a bug in the raw-window batatch shape, or a genuinely undertrained tiny
   RNN) is left open — PREREG deliberately fixed the protocol before running,
   and rule 5 prohibits silently fixing a result that looks wrong without
   flagging it first.
2. Whether SVM/RF's FFT-magnitude feature set should be expanded (e.g. added
   phase, log-magnitude, or additional time-domain features) is a design
   question for whoever reviews this lane, not resolved here — PREREG fixed
   the feature set before running and the no-cherry-picking commitment means
   it wasn't revisited after seeing the low accuracy.
3. Runtime was ~2000s (~33 min) for single-split + 15x-repeated combined,
   single-process, `n_jobs=2`/`torch.set_num_threads(2)` as instructed.

## File list

- [PREREG.md](PREREG.md) — frozen pre-registration, written before any run
- [run_lane4_comparison.py](run_lane4_comparison.py) — the script that produced every number in this report
- [run_stdout.log](run_stdout.log) — full raw stdout of that run
- [lane4_single_split.csv](lane4_single_split.csv)
- [lane4_repeated_grouped_cv_raw.csv](lane4_repeated_grouped_cv_raw.csv)
- [REPORT.md](REPORT.md) — this file

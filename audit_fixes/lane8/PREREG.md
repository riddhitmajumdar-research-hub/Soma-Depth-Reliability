# Lane 8 Pre-Registration: Random Forest Depth-Floor Certification

Status: PRE-REGISTERED before any run. Written before `lane8_harness.py` is
executed on any seed beyond the single-seed runtime benchmark explicitly
requested below. The 24-seed certification run itself is **not** launched
until explicit go-ahead is given (see REPORT.md for the launch record, once
that happens).

## 0. Question this lane answers

The certified depth floor (README's headline 8.48cm +/- 1.81cm, gate at
6.50cm) comes from one model family: a CNN (`SimpleCNN`,
`Source/degradation_aware_final_sweep.py`). Lane 8 asks whether that floor
is a **property of the signal/degradation model itself** (physics +
attenuation + sensor noise), or a **CNN-specific artifact** (e.g. of its
inductive bias, capacity, or training dynamics) -- by certifying the same
floor under an unrelated model family (Random Forest, no learned
convolutional structure, no gradient-based training) on the *identical*
data-generation and significance-testing pipeline.

## 1. Provenance (discovered in this pass, not assumed)

| Role | File | Function(s) |
|---|---|---|
| Certified 24-seed CNN sweep (seeds 101-124) | `Source/degradation_aware_final_sweep.py` | `build_degradation_aware_dataset`, `generate_degraded_windows`, `bh_fdr_correction`, `train_cnn` (CNN-specific, NOT reused), `EVAL_DEPTHS_CM`, `FREQ`, `N_SAMPLES_PER_CELL`, `N_SHUFFLES`, `TRAIN_DEPTHS_CM`, `N_SAMPLES_PER_DEPTH_PER_CLASS` |
| Windowing | `Source/cnn_classifier.py` | `extract_raw_windows(signal, time, heart_rate_hz, duration=3.0, window_size=100)` -- confirmed by direct call: `build_degradation_aware_dataset()` returns `X.shape == (375, 100)` (5 train depths x 3 classes x 25 realizations, 100-sample raw windows), `y.shape == (375,)` |
| Physics/classes | `Source/physics_model.py` | `generate_signal`, `stiffness_classes` |
| Sensor SNR constant | `Source/constants.py` | `SENSOR_SNR_DB = 27.05` (confirmed by grep; matches the task's stated 27.05 dB) |
| Eval grid | `Source/degradation_aware_final_sweep.py` lines 32-34 | `EVAL_DEPTHS_CM` = union of 12 depths linspace(0,15) and 11 fine depths near the floor; confirmed by direct computation: **23 unique depths**, matching the task's stated "23-depth eval grid" |

**Confirmed: `build_degradation_aware_dataset`, `generate_degraded_windows`,
and `bh_fdr_correction` are imported unchanged from
`degradation_aware_final_sweep.py`.** Only `train_cnn` is replaced --
Lane 8's actual contribution is a same-interface `train_rf` (train a model,
return an object with a `.predict(X) -> labels` interface) so the rest of
the pipeline (control shuffling, per-depth eval, BH-FDR, floor
determination) is bit-for-bit the same code path as the certified sweep,
just called with a different model constructor.

`Source/degradation_aware_resnet_final_sweep.py` (the ResNet1D thread,
already the subject of Lane 7) is a different, unrelated lane and is not
imported or touched. This lane is **read-only on `Source/` and on every
other `audit_fixes/lane*` directory**; it writes only inside
`audit_fixes/lane8/`.

## 2. What is reused unchanged vs. what is new

Unchanged (imported, not reimplemented):
- `build_degradation_aware_dataset()` -- training-set construction, 25
  realizations/class x 5 depths (`TRAIN_DEPTHS_CM = [0, 3, 6, 9, 12]`).
- `generate_degraded_windows()` -- physics + attenuation + sensor-noise
  generation, used identically for training data and for every eval cell.
- `EVAL_DEPTHS_CM` -- the same 23-depth grid, same values, same order.
- `bh_fdr_correction()` -- the same BH-FDR flagging logic, same default
  `alpha=0.05`, same one-family-per-seed convention as the original.
- `FREQ = 100` Hz, `SENSOR_SNR_DB = 27.05` dB (via `constants.py`,
  untouched), `N_SAMPLES_PER_CELL = 300` (task-specified, matches the
  original's already-raised value), `N_SHUFFLES = 200` shuffled-label
  controls per seed.

New (Lane 8's actual contribution, in `lane8_harness.py`):
- `train_rf(X, y_encoded)` -- trains `sklearn.ensemble.RandomForestClassifier`
  directly on the raw `(n_samples, 100)` window array (no reshaping to add
  a channel dimension -- that reshape in the original code is CNN-specific,
  needed for `nn.Conv1d`; RF's `.fit(X, y)` takes a 2D array natively, so
  the "raw 100-sample window as features" requirement is satisfied without
  any transform).
- Per-depth, per-seed persistence of every control's accuracy (not just the
  BH-flag/floor), so p-values can be recomputed offline -- same rationale
  and format as Lane 5's aggregate-depth-detail addendum.

## 3. Random Forest configuration (fixed, no tuning)

Per the protocol, all hyperparameters are fixed in
`lane8_config.py`, none are searched or tuned:

- `n_estimators = 300`
- `n_jobs = 2`
- All other `RandomForestClassifier` hyperparameters left at sklearn
  defaults (`criterion="gini"`, `max_depth=None`, `min_samples_split=2`,
  `min_samples_leaf=1`, `max_features="sqrt"`, `bootstrap=True`) --
  **explicitly not tuned**, since the task says "fixed hyperparameters...
  no tuning" and specifies only trees/features/n_jobs. Flagged as a
  judgment call: sklearn's default `max_features` for a classifier is
  `"sqrt"`, not "all features" -- stated here rather than silently assumed,
  since it materially affects how much of the 100-sample window each split
  can see.

**New reproducibility requirement not present in the CNN path (judgment
call, flagged):** `RandomForestClassifier` has its own internal randomness
(bootstrap resampling per tree, feature subsampling per split) controlled
by `random_state`, on top of whatever `np.random.seed(seed)` does upstream
for data generation. The CNN path has no equivalent knob (full-batch
gradient descent from a `torch.manual_seed`-initialized net is
deterministic given fixed data). To keep runs reproducible and exactly
re-derivable:
- Real model: `random_state = seed`.
- Control `i` (of 200) for seed `s`: `random_state = seed * 10_000 + i`
  (deterministic, collision-free across the seed x control grid used here:
  200 controls per seed, seed in [101,124], so `s*10000+i` is injective for
  `i < 10000`).
This is a new design decision RF requires that the CNN path did not; it is
disclosed here rather than left implicit, per the convention.

## 4. Protocol match to the CNN sweep (everything else identical)

- Seeds: 101-124 inclusive, same 24 as the CNN sweep and Lane 5/7.
- Training-set construction: `build_degradation_aware_dataset()`, unchanged
  -- 25 realizations/class x 5 depths = 375 training windows, same as CNN.
- Eval grid: `EVAL_DEPTHS_CM`, unchanged, 23 depths, at `FREQ = 100` Hz and
  `SENSOR_SNR_DB = 27.05` dB (both unchanged, via unmodified imports).
- `N_SAMPLES_PER_CELL = 300` per depth (100/class), unchanged.
- Controls: 200 shuffled-label RF models per seed, same
  `np.random.permutation(y_train_encoded)` shuffle protocol as the
  original, trained once per seed and reused across all 23 depths (matches
  the original's "train once per seed, evaluate at every depth" design).
- Per-depth p-value: `p = (1 + #{controls with acc_control >= acc_real}) /
  (1 + 200)`, identical formula to the original.
- FDR: one family per seed (all 23 depths), `bh_fdr_correction(...,
  alpha=0.05)`, unchanged -- same convention as the CNN sweep and as Lane
  5's aggregate-level analysis (this lane does not do per-class analysis;
  the task only asks for the aggregate floor).
- Floor convention: **first-failed-depth** -- the first depth (in
  increasing order) where the FDR flag is not significant, matching the
  CNN sweep's `naive_floor_cm` definition exactly (same variable name is
  not reused to avoid confusion with the CNN's certified CSV; this lane's
  output column is `naive_floor_cm_rf`).

## 5. Runtime estimate (per the protocol, before launching the full run)

`lane8_benchmark.py` times ONE seed end-to-end (train real RF + 200 control
RFs + full 23-depth eval sweep) and linearly projects the 24-seed total.
This benchmark is run as part of this pre-registration step (it is a
single-seed timing measurement, not "the run" -- the 24-seed certification
run itself is not started here). Result recorded in `logs/lane8_benchmark_result.json`
and reported in REPORT.md before any go-ahead is requested for the full sweep.

Expectation stated before measuring (to avoid post-hoc rationalizing a
surprising number either way): RF training (fit 300 trees on a 375x100
array) is expected to be much faster per-model than 60 full-batch gradient
steps on a CNN, so wall-clock per seed should be dominated by data
generation (`generate_degraded_windows`, physics simulation) shared
identically with the CNN path, not by model fitting -- but this is a
prediction to be checked against the actual benchmark, not assumed.

## 6. Artifacts saved per seed

- Real model: not persisted as a pickled sklearn object by default (unlike
  the CNN's `state_dict`, an RF's picked object can be large with 300 deep
  trees x 200 controls x 24 seeds); instead, per-depth **accuracies** for
  the real model and all 200 controls are saved in full, which is
  sufficient to recompute every p-value, FDR flag, and floor exactly
  offline without needing the fitted models themselves. Flagged as a
  judgment call, symmetric with the protocol's own rule ("save the
  seed's real-model and control accuracies per depth so p-values can be
  recomputed" -- the task asks for accuracies, not model objects).
- Per-seed accuracy detail CSV (real + 200 control accuracies per depth):
  `audit_fixes/lane8/logs/lane8_depth_detail_seed{seed}.csv`.
- Aggregate per-seed summary row (floor, timings):
  `audit_fixes/lane8/logs/lane8_results.csv`.
- Checkpoint/resume: identical pattern to Lane 5/the original driver --
  CSV existence + already-done seed set determines what to skip.
- Log file: `audit_fixes/lane8/logs/lane8_run.log` (progress + ETA).

## 7. Planned report (once the 24-seed run is authorized and completes)

Per the protocol, `lane8_analysis.py` (written now, not run until data
exists) will compute, mirroring Lane 7's pre-registered statistics exactly
so the two lanes are comparable:

1. RF's 24 floors: mean, SD (ddof=1, matching the record's
   correction), median, 10th percentile under three conventions (numpy
   `linear`, `weibull`, nearest-rank), range.
2. Overlap with the CNN's 24 floors (`Source/logs/degradation_aware_final_results.csv`):
   count/fraction of seeds where RF floor == CNN floor exactly, and where
   RF floor is within +/-1 grid step of CNN floor (grid is non-uniform, so
   "1 step" is stated per-seed, not a fixed cm value).
3. Unpaired test: Mann-Whitney U (`scipy.stats.mannwhitneyu`,
   two-sided) plus a manual permutation-test cross-check, same as Lane 7
   section 47(2).
4. Paired test: Wilcoxon signed-rank on per-seed (RF - CNN) differences
   (both models share all 24 seeds, so no n-mismatch caveat Lane 7 had to
   carry for the incomplete ResNet1D sweep), plus paired Hodges-Lehmann
   with a 10,000-resample bootstrap 95% CI, `np.random.default_rng(seed=20260101)`
   for reproducibility, same as Lane 7.
5. Fraction of seeds with RF floor < 6.50 cm gate, Clopper-Pearson 95% CI.

No number from this section is computed before the 24-seed run exists; this
section only locks the analysis plan in advance.

## 8. Disagreement policy

Any Lane 8 number that conflicts with a number already stated in
README.md or in Lane 5/7's reports will be listed under "Flagged
disagreements" in REPORT.md with both values shown side by side, per
the convention -- not resolved or adjudicated here.

## 9. Explicit stop

Per the protocol, this pre-registration and the single-seed runtime
benchmark are the only executions performed in this step. The full 24-seed
`lane8_harness.py` run is **not launched** until explicitly authorized.

## 10. Addendum: measured single-seed benchmark (seed 101)

`lane8_benchmark.py` was run once, per section 5 (raw output:
`logs/lane8_benchmark_result.json`):

- Wall-clock for seed 101, end-to-end: **410.4s** (~6.8 min) --
  train real RF: 0.65s, train 200 control RFs: 142.2s, full 23-depth eval
  sweep: 267.3s.
- **The expectation stated in section 5 was wrong and is corrected here,
  not silently revised:** RF model-fitting is indeed cheap (0.65s for the
  real model), but eval is the dominant cost (267s, 65% of wall-clock), not
  data generation shared with the CNN path being negligible either way --
  most of eval time is `generate_degraded_windows` (physics simulation)
  called fresh per depth (23 calls x 300 samples), not the 4,623
  RF `.predict()` calls, which are cheap. This is the same eval-data-generation
  cost the CNN sweep also pays per depth (unchanged, imported code), so it
  is not RF-specific -- but it means total wall-clock is not
  train-dominated as guessed.
- Linear projection for the full 24-seed run: **9,850s = 164.2 min = 2.74
  hr**. Caveat, stated per the convention: this is a linear
  extrapolation from a single seed; RF training/control cost could vary
  seed-to-seed with class-shuffle balance, and is not itself validated
  against a second seed in this step.
- Seed 101's RF floor from this single benchmark run: naive_floor_cm_rf =
  13.64cm, anomalous_recovery_rf = False. **This is one seed, not a
  certified result** -- stated here only as a benchmark byproduct, not as
  evidence toward the 24-seed question this lane exists to answer. For
  comparison (not yet a paired analysis, just an orientation number): CNN
  seed 101's naive_floor_cm in the certified CSV should be checked in
  REPORT.md once the full run exists, not asserted here from memory.
- This benchmark seed's per-depth detail is saved at
  `logs/lane8_depth_detail_seed101.csv` as a normal side effect of calling
  `run_one_seed` (same file `main()` would also produce); it is not yet
  appended to `logs/lane8_results.csv`, since only `main()` writes that
  file and `main()` was not invoked in this step.

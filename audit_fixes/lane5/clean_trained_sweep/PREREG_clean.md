# Lane 5 Pre-Registration — Clean (Non-Degradation-Aware) Comparison Sweep

Status: PRE-REGISTERED before any run of this sweep. Written as specified
in the protocol: "prepare now, run later." **Does not touch the running main
job** (pid of `lane5_harness.py`'s 24-seed run, nor pid 7429's ResNet1D
sweep). Separate output directory, separate script, chained only after the
main run finishes (see section 5).

## 0. Provenance (discovered, not assumed)

- "Clean CNN" training definition: `Source/floor_characterization_cnn.py`
  line 92-93 (`"Training the real CNN (clean signals, exactly as Stage
  3c)..."`) calls `build_raw_dataset()` from `Source/cnn_classifier.py`.
  `build_raw_dataset` generates the raw physics signal via
  `generate_signal` and windows it via `extract_raw_windows` — **no
  `attenuate_signal` call, no depth, no added sensor noise at all**. This
  is genuinely noiseless/depth-0 training data, not depth=0cm run through
  the noisy degradation pipeline.
- `train_cnn` and `bh_fdr_correction` in `floor_characterization_cnn.py`
  are **byte-identical** (checked via `inspect.getsource`, diff confirmed
  in this pass) to the same-named functions already imported by the main
  Lane 5 harness from `degradation_aware_final_sweep.py`. This sweep
  reuses the already-imported `degradation_aware_final_sweep.train_cnn`
  / `.bh_fdr_correction` rather than importing a second, functionally
  duplicate copy — same unmodified logic either way, confirmed not
  assumed.
- Evaluation (per the protocol: "identical to the main run") reuses
  `generate_degraded_windows` and `EVAL_DEPTHS_CM` from
  `degradation_aware_final_sweep.py` unchanged: 100Hz, 27.05dB
  (`SENSOR_SNR_DB`), the same 23-point depth grid, `N_SAMPLES_PER_CELL=300`,
  200 shuffled-label controls, BH-FDR within each seed (family = one
  seed, all depths — matching the main run's own aggregate-floor family
  definition, section 1 of `PREREG.md`).

## 1. What varies vs. the main run

| | Main run (degradation-aware) | Clean sweep (this addendum) |
|---|---|---|
| Training data | `build_degradation_aware_dataset()`: 5 depths (0,3,6,9,12cm) x attenuated x 27.05dB noise | `build_raw_dataset()`: clean, noiseless, no attenuation (Stage 3c protocol) |
| Training function | `train_cnn` (imported) | same `train_cnn` (imported, identical function object) |
| Evaluation | `generate_degraded_windows`, 23-depth grid, 100Hz, 27.05dB, N=300/cell | identical, unchanged |
| Controls | 200 shuffled-label, BH within seed | identical, unchanged |
| Seeds | 101-124 | 101-124 (same 24) |

This isolates exactly one variable: whether training was degradation-aware
or clean, holding the entire evaluation protocol fixed. This is the same
logic `noise_robustness_stress_test.py` already used for isolating
evaluation-SNR as the one varied condition (PREREG.md section 0 table).

## 2. TRAIN_MODE flag

`audit_fixes/lane5/lane5_config.py` gains `TRAIN_MODE` with two legal
values: `"degradation_aware"` (the main run, default, unchanged behavior)
and `"clean"` (this addendum). The clean sweep is a **separate script**
(`lane5_harness_clean.py`), not a code path inside the already-running
`lane5_harness.py` — per the protocol, not to touch the running job, and
because the running process has already imported and pickled its own
module state; editing the shared file it's running from risks affecting
a live interpreter's already-loaded bytecode/globals in ways that are
hard to reason about safely. `lane5_harness_clean.py` imports
`lane5_config` for shared constants (thresholds, N_SHUFFLES, alpha) but
does not import or execute anything from `lane5_harness.py` itself, to
guarantee zero interaction with the running process.

## 3. Output location

Entirely separate from the main run's artifacts:
- `audit_fixes/lane5/clean_sweep/logs/lane5_clean_per_class_results.csv`
- `audit_fixes/lane5/clean_sweep/logs/lane5_clean_run.log`
- `audit_fixes/lane5/clean_sweep/logs/lane5_clean_aggregate_depth_detail_seed{seed}.csv`
- `audit_fixes/lane5/clean_sweep/seed_weights/clean_seed{seed}.pt` (real model)
- `audit_fixes/lane5/clean_sweep/seed_weights/controls_seed{seed}/control{000-199}.pt`
- `audit_fixes/lane5/clean_sweep/controls/seed{seed}_depth{depth}_{class}.npy`

Same checkpoint/resume convention as the main harness (CSV row presence =
done).

## 4. No prior "clean" 24-seed floor exists to check against

Unlike the main run (which had `Source/logs/degradation_aware_final_results.csv`
to diff against), there is no existing 24-seed clean-CNN floor sweep in
the repo to reproduce — `floor_characterization_cnn.py` runs a
single-seed (`SEED=42`) certification across 4 frequencies, not a 24-seed
sweep at one frequency. **So this sweep has no original-CSV MATCH/MISMATCH
check available; its own internal consistency (e.g. BASIC_CNN_CORRECTED_FLOOR_CM=4.30cm
from `constants.py`, which is `floor_characterization_cnn.py`'s single-seed
100Hz corrected-noise floor) is the only available cross-reference, and
even that is a different seed/protocol axis (single seed vs. 24-seed
distribution) — flagged as a genuine limitation, not resolved here.**

## 5. Launch sequencing (per the protocol: "when the main run finishes, chain")

Explicitly authorized to auto-chain. **Revised after the main run died
silently for ~3h45m with no PID to watch (see `PREREG.md` section 10):**
`audit_fixes/lane5/run_chain.py` (v2) now polls
`lane5_per_class_results.csv` for all 24 pre-registered seed rows,
rather than waiting on an OS PID — a PID disappearing does not reliably
mean "all 24 seeds finished" (it could just as easily mean "died at seed
3," as actually happened). Once all 24 rows are present, it launches, in
order: (1) `lane5_harness_clean.py` (this addendum's clean sweep, 24
seeds), (2) `multiplicity_robustness.py --run main`, (3)
`multiplicity_robustness.py --run clean`. "Prepare now, run later" is
satisfied because no training/compute for any step executes until the
poll succeeds — the chain script is inert (one CSV read per minute)
while waiting. Not launched here; the launch command is
to be run manually.

## 6. Threads

`torch.set_num_threads(4)`, matching the main run and benchmark.

## 7. Addendum (per the protocol, write-only prep while the main
run's resume is launched separately): confusion matrices

`lane5_harness_clean.py` now also saves, per seed and per depth, a 3x3
confusion matrix (rows=true class, cols=predicted class, order =
`CLASS_NAMES_SORTED` = [cirrhosis, healthy, significant_fibrosis]) for
the real model and for all 200 controls (stacked into one
`(200, 3, 3)` array per depth, to bound file count), to
`audit_fixes/lane5/clean_sweep/confusion_matrices/`. Computed by a small
new `confusion_matrix_3x3()` helper (plain double-loop over predictions,
no external dependency) — this is new bookkeeping code, not a
reimplementation of any imported training/eval/FDR function; the
predictions it consumes still come only from the imported `train_cnn`
model's own `forward()` output. This was added specifically so
`perclass_balacc_eval.py` (see `PREREG_perclass.md`) has a matched,
independently-computed confusion-matrix source to cross-check against
for the clean sweep, and so per-class balanced accuracy / F1 can be
derived post-hoc for the clean sweep without any retraining. Real +
control weight saving (`seed_weights/clean_seed{seed}.pt`,
`seed_weights/controls_seed{seed}/control{000-199}.pt`) was already
implemented in the first version of this script — confirmed present,
not newly added.

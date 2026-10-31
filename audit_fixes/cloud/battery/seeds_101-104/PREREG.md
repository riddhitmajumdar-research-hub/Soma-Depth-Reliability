# Battery-A pre-registration (seeds 101-104)

Lane: battery-A. Branch: run battery-101-104. Written BEFORE any run;
git history on this branch is the timestamp for this document.

## Goal
Measure how far the certified depth floor (from
`Source/degradation_aware_final_sweep.py`) moves when specific simulator
assumptions change, for seeds 101-104, across 7 condition families (with
sub-variants), using the exact real-CNN-vs-200-shuffled-controls /
BH-FDR-per-seed protocol the certified sweep uses.

## What is imported unchanged (read-only)
- `Source/degradation_aware_final_sweep.py`: `build_degradation_aware_dataset`,
  `train_cnn`, `bh_fdr_correction`, `generate_degraded_windows`,
  `EVAL_DEPTHS_CM`, `FREQ`, `N_SAMPLES_PER_CELL`, `TRAIN_DEPTHS_CM`.
- `Source/physics_model.py`: `generate_signal`, `stiffness_classes`,
  `shear_wave_velocity_m_s`, `time_of_flight_s`.
- `Source/attenuation_model.py`: `attenuate_signal`, `attenuation_coefficient`,
  `LIVER_VISCOSITY_BY_CLASS`.
- `Source/cnn_classifier.py`: `SimpleCNN`, `extract_raw_windows`.
- `Source/constants.py`: `SENSOR_SNR_DB`.
- `audit_fixes/lane5/lane5_harness.py` `run_one_seed` is read for the
  training/control/eval flow pattern (per-depth p-value: (1+#controls>=real)/201,
  BH within seed) but not imported (lane5/lane8 are off-limits to edit; we
  re-implement the same pattern in our own harness file, calling only the
  Source functions above).

Nothing under `Source/` or `audit_fixes/lane5`/`audit_fixes/lane8` is edited.
All new code lives under `audit_fixes/cloud/battery/`.

## Confirmed facts about the existing code (checked by reading, before any run)
1. **Grid**: `EVAL_DEPTHS_CM = sort(unique(concat(linspace(0,15,12), [3.5,4.0,4.3,4.5,5.0,5.5,6.0,6.5,7.0,7.5,8.0])))`
   has 23 points (12 base + 11 fine, no exact duplicates) at 100 Hz. This is
   the "23-depth grid" referenced in the task.
2. **SNR->noise convention**: `SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)`,
   applied as `attenuated + np.random.normal(0, SENSOR_NOISE_STD, ...)` in
   `generate_degraded_windows`. Signal amplitude is order-1 (pulse peak ~1
   before attenuation), so this is a fixed-noise-std convention, not a
   per-signal-power SNR — noise std does NOT depend on the attenuated
   signal's own amplitude. At 27.05 dB, std = 10**(-27.05/20) = 0.04443.
   Condition 5 (SNR sweep) will recompute std the same way for each target dB.
3. **Geometric spreading**: `attenuate_signal` = `signal * exp(-alpha * depth_m)`
   only. There is NO 1/distance or spherical/cylindrical spreading term in
   the original code. Condition 3 adds one as a wrapper multiplier
   `min(1, (x0/x)^p)` applied on top of the existing exponential decay,
   x0 = 1 cm (0.01 m), x = depth_m (with a floor at a small epsilon to avoid
   div-by-zero at depth_m=0 — depth 0 cm is in the eval grid).
4. **Beat index**: `generate_signal` places one pulse per
   `beat_times = arange(0, duration, 1/heart_rate_hz)`, and
   `extract_raw_windows` returns one window per beat time in the same order
   (`enumerate(beat_times)`, skipping only windows that run past the
   signal's end). So window order within one generated signal directly
   corresponds to beat index k=0,1,2,... in chronological order. Condition 1
   will tag each window with its k at generation time (a thin bookkeeping
   wrapper around `extract_raw_windows`, not a change to its logic) and
   report accuracy for k=0 vs k>=1 separately, in addition to the pooled
   baseline.
5. **Viscosity dict**: `LIVER_VISCOSITY_BY_CLASS = {"healthy": 0.5,
   "significant_fibrosis": 0.5, "cirrhosis": 1.77}` Pa.s, looked up in
   `attenuation_coefficient` via `_eta_for_kpa` (exact-kPa dict match against
   `stiffness_classes`).
6. **Stiffness classes / kPa**: `{"healthy": 4.5, "significant_fibrosis": 7.5,
   "cirrhosis": 12.0}` kPa.
7. **G = E/3 relation**: confirmed in both `shear_wave_velocity_m_s`
   (`G_pa = E_pa/3`) and `attenuation_coefficient` (`G_pa = E_pa/3`) —
   consistent, so condition 6b's `G = E/3` assumption matches the code.
8. **Existing (weak-damping) alpha formula**: `alpha = (omega**2 * eta) /
   (2 * G_pa * c)`, `c = shear_wave_velocity_m_s(E)`. This is the small-loss
   approximation to the exact complex-wavenumber attenuation, not the exact
   value itself — condition 6 computes the exact value
   `alpha_exact = |Im(k)|`, `k = omega * sqrt(rho / (G + i*omega*eta))`,
   `rho = 1060`, and reports MATCH/MISMATCH against the task's quoted
   100 Hz reference values (healthy 53.85, fibrosis 25.46, cirrhosis 42.92 /m)
   as its own logged output, not assumed.

## Wrapper design (all in `audit_fixes/cloud/battery/wrappers.py`)
Each condition is implemented as a wrapper function that copies the
relevant original function's body verbatim except the one stated change,
never imports-and-monkeypatches, and never edits the original file. A
regression test (`audit_fixes/cloud/battery/test_regression.py`) asserts
that the "baseline" wrapper path (condition 1 with no shift applied)
reproduces `generate_degraded_windows` / `attenuate_signal` / `generate_signal`
bit-for-bit (`np.array_equal`) for a fixed seed, run before seed 101 is
launched. If this test fails, the run stops and it is reported, not
silently patched.

## Protocol per seed (matches audit_fixes/lane5/lane5_harness.py's pattern)
1. Train the real CNN once via `build_degradation_aware_dataset` +
   `train_cnn` (unchanged).
2. Train 200 shuffled-label controls the same way, keep all 201 models in
   memory for the whole seed.
3. For each condition (in the priority order given in the task), build one
   test set per depth in the 23-point grid (N_SAMPLES_PER_CELL=300 total,
   100/class, matching the original) using that condition's wrapper
   generation path, evaluate the real model and all 200 controls on the
   SAME test set.
4. Per depth: `p = (1 + #controls_with_acc>=real_acc) / 201`; BH across the
   23 depths within (seed, condition) using the imported
   `bh_fdr_correction` (same alpha=0.05 default as the original sweep).
   Floor = first depth (in ascending order) that is NOT BH-significant; if
   every depth is significant, floor = grid max, flagged
   `censored=True` (right-censored).
5. Persisted per (seed, condition, depth): real accuracy, raw p, BH-adjusted
   p (recomputed via the same monotone step-up formula lane5 uses, since
   the imported `bh_fdr_correction` only returns booleans), significance
   flag, the real model's 3x3 confusion matrix (`results/confmat_<seed>.npz`),
   and min/median/max/std of the 200 controls' accuracy at that depth (to
   flag degenerate nulls, e.g. std ~ 0).

## Condition 8 — pure-noise probe (added during the run, after
## conditions 1-7 were already drafted; appended here rather than
## renumbering, per rule 4's "report, never silently fix, any mismatch")
Runs LAST for each seed, after conditions 1-7 finish and are committed. For
each seed: build 300 test windows (window_size=100, matching WINDOW) that
are pure Gaussian noise, no pulse — `np.random.normal(0, SENSOR_NOISE_STD,
size=(300, 100))` using the identical SNR->std convention as condition 5 /
the original code (`10 ** (-SENSOR_SNR_DB/20)` at the baseline 27.05 dB).
Evaluate the real model and all 200 controls; record the predicted-class
distribution (fraction of the 300 windows assigned to each class) for the
real model, and the MEAN predicted-class distribution across the 200
controls. Class order is the LabelEncoder's `classes_` order fit on
`sorted(stiffness_classes.keys())`-style names in
`build_degradation_aware_dataset`/`run_one_seed` — alphabetical:
`["cirrhosis", "healthy", "significant_fibrosis"]` (confirmed: sklearn's
`LabelEncoder.fit` sorts unique labels lexicographically, and lane5's own
`CLASS_NAMES_SORTED = sorted(stiffness_classes.keys())` uses the same
order). No floor/p-value computed for this condition — counts/fractions
only. Output: `results/noise_probe_<seeds>.csv` with columns
`seed, source, class, fraction` (source in {real, control_mean}).

## Deviations / open judgment calls (disclosed up front)
- The task's SNR condition 5 list includes 13.58 dB, which is exactly
  `constants.SENSOR_SNR_DB_WORST_CASE` — noted as a cross-check, not treated
  specially.
- N_SHUFFLES=200 controls trained per seed, not per (seed, condition): the
  condition only changes the TEST distribution, never training, matching
  the requirement to evaluate the real model AND all 200 (baseline-
  trained) controls on the same shifted test set. This is a judgment call:
  it isolates "does the floor move because the *deployment* distribution
  shifted" from "does the floor move because retraining under the shift
  would change what both real and control models learn." Flagged as a
  limitation in REPORT.md; the alternative (retrain per condition) would
  multiply compute by ~7x-14x and is out of scope for a 4-seed battery.
- Condition 6b changes TOF via `kpa_eff = 3*rho*c_exact^2` in kPa, which is
  passed to `generate_signal` in place of the class's nominal kPa — this
  changes the pulse's timing but not the classification label (label stays
  the class's true kPa/eta identity). Disclosed as a design choice.
- Per task rule 6, no significance test is run ACROSS only these 4 seeds;
  REPORT.md reports per-seed numbers, per-seed shift vs baseline, and the
  raw median shift only (no test statistic on n=4).
- Any other flaw found while running is added here or to REPORT.md's
  deviations section as it is found, per rule 4.

## Cross-platform check
The Mac replicate baseline floors quoted in the task (seeds 101,102,103,104:
9.55, 6.50, 9.55, 7.50 cm) will be compared against this Linux battery's own
condition-1 (baseline) floors for the same seeds and reported as
agreement/disagreement in REPORT.md — not treated as ground truth to match.

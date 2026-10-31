# Battery-B pre-registration (seeds 105-108)

Lane: battery-B. Branch: run battery-105-108. Written BEFORE any run;
git history on this branch is the timestamp for this document.

## Goal
Replicate the `run battery-101-104` battery (branch `run battery-101-104`,
its `PREREG.md`/`REPORT.md` read via `git show`, never merged or checked out)
for seeds 105-108, with the exact same 19 conditions (1, 2, 3a/3b, 4a/4b,
5a-5e, 6a/6b, 7a-7f, 8 -- the pure-noise probe), same priority order, same
per-seed/per-condition checkpointing and CSV column schema, so results
concatenate cleanly with the 101-104 set for a combined n=8 analysis. ONE
CORRECTION is applied (see below), condition 2 only.

## Correction to condition 2 (the only intentional deviation from
## 101-104's protocol)
101-104's condition 2 drew the amplitude gain as
`g = exp(U(ln 0.5, ln 2.0))` (log-uniform), which has `E[g] ~ 1.08`
(biased high) -- a design error in the original PREREG.md, found after
that run (101-104 disclosed this in their own REPORT.md addendum and
added a separate condition "2b" rerun to correct it while leaving their
original condition-2 results untouched).

On THIS branch (seeds 105-108), condition 2 is run ONLY in the corrected
form: `g ~ Uniform(0.5, 1.5)` drawn in LINEAR space (`E[g] = 1.0` exactly),
applied at the identical point in the signal chain as 101-104's condition 2
(after attenuation, before noise). There is no "2 vs 2b" split here --
seeds 105-108 only ever see the corrected draw, still labeled condition id
`2_amp_randomization` (not `2b`) since this branch has no uncorrected
version to distinguish it from. This means condition 2's rows in
`results/floors_105-108.csv` are NOT bit-for-bit comparable to 101-104's
own `2_amp_randomization` rows (those used the biased log-uniform draw) --
they ARE comparable to 101-104's `2b_amp_randomization_linear` rows (same
corrected distribution). REPORT.md will flag this explicitly when framing
any 101-104-vs-105-108 comparison for condition 2.

All other 18 conditions (1, 3a, 3b, 4a, 4b, 5a-5e, 6a, 6b, 7a-7f, 8) use
IDENTICAL wrapper logic, shift parameters, condition ids and labels as
101-104's `config.py`/`wrappers.py` (copied then verified unchanged except
condition 2's entry and the seed list) -- see "What is imported unchanged"
and "Confirmed facts" below, which restate 101-104's PREREG.md verbatim
(re-verified against `Source/` on this branch, not assumed).

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
- `audit_fixes/lane5/lane5_harness.py` `run_one_seed` is read (not imported)
  for the training/control/eval flow pattern, same as 101-104. lane5/lane8
  are off-limits to edit; our own harness re-implements the pattern calling
  only the Source functions above.

Nothing under `Source/` or `audit_fixes/lane5`/`audit_fixes/lane8` is
edited. All new code lives under `audit_fixes/cloud/battery/` (this branch's
copy, forked from 101-104's files at the point they were read via
`git show`, not merged).

## Confirmed facts about the existing code (re-checked by reading, before
## any run -- restated from 101-104's PREREG.md, not assumed identical)
1. **Grid**: `EVAL_DEPTHS_CM` (23-point grid, 12 base linspace(0,15,12) +
   11 fine points, sorted unique) at 100 Hz.
2. **SNR->noise convention**: `SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)`,
   applied as `attenuated + np.random.normal(0, SENSOR_NOISE_STD, ...)`.
   Fixed-noise-std convention, not per-signal-power SNR.
3. **Geometric spreading**: `attenuate_signal = signal * exp(-alpha*depth_m)`
   only, no 1/distance term in the original code. Condition 3 adds
   `min(1, (x0/x)^p)`, x0=1cm, on top.
4. **Beat index**: window order within a signal corresponds to beat index
   k=0,1,2,... in chronological order (condition 1 tags k at generation
   time).
5. **Viscosity dict**: `LIVER_VISCOSITY_BY_CLASS = {healthy: 0.5,
   significant_fibrosis: 0.5, cirrhosis: 1.77}` Pa.s.
6. **Stiffness classes**: `{healthy: 4.5, significant_fibrosis: 7.5,
   cirrhosis: 12.0}` kPa.
7. **G = E/3** confirmed in `shear_wave_velocity_m_s` and
   `attenuation_coefficient`.
8. **Existing (weak-damping) alpha**: `alpha = (omega**2*eta)/(2*G_pa*c)`.
   Condition 6 computes the exact `alpha_exact = |Im(k)|`,
   `k = omega*sqrt(rho/(G+i*omega*eta))`, rho=1060, checked against the
   task's reference values (healthy 53.85, fibrosis 25.46, cirrhosis
   42.92 /m at 100Hz) as its own logged output.

## Wrapper design
`audit_fixes/cloud/battery/wrappers.py` on this branch is 101-104's
`wrappers.py` file, byte-identical except nothing (the `amp_gain` /
log-uniform function is left in place, unused by this branch's `config.py`,
so the file matches 101-104's for easy diffing; only `config.py`'s
condition-2 entry points at `amp_gain_linear` instead of `amp_gain`).
`audit_fixes/cloud/battery/test_regression.py` (also unchanged) asserts
the baseline wrapper path reproduces `generate_degraded_windows`
bit-for-bit (`np.array_equal`) for 3 seeds x 3 classes x 3 depths, run
before seed 105 is launched. If it fails, the run stops and is reported.

## Protocol per seed (identical to 101-104)
1. Train the real CNN once via `build_degradation_aware_dataset` +
   `train_cnn` (unchanged).
2. Train 200 shuffled-label controls the same way, keep all 201 models in
   memory for the whole seed.
3. For each condition (priority order below), build one test set per depth
   in the 23-point grid (N_SAMPLES_PER_CELL=300, 100/class) using that
   condition's wrapper generation path, evaluate the real model and all
   200 controls on the SAME test set.
4. Per depth: `p = (1 + #controls_with_acc>=real_acc) / 201`; BH across the
   23 depths within (seed, condition) using the imported
   `bh_fdr_correction` (alpha=0.05). Floor = first depth (ascending) that
   is NOT BH-significant; if every depth is significant, floor = grid max,
   `censored=True`.
5. Persisted per (seed, condition, depth): real accuracy, raw p, BH-adjusted
   p, significance flag, the real model's 3x3 confusion matrix
   (`results/confmat_<seed>.npz`), and min/median/max/std of the 200
   controls' accuracy (degenerate-null check).

## Condition priority order (19 total, matching 101-104's numbering except
## condition 2 is the corrected version only)
1 baseline; 2 amplitude randomisation (CORRECTED, linear
g~U(0.5,1.5)); 3a spreading p=0.5; 3b spreading p=1.0; 4a stiffness CV
per-class; 4b stiffness CV=0.15 uniform; 5a-5e SNR sweep (20, 13.58, 10, 5,
0 dB); 6a exact attenuation; 6b exact attenuation + exact TOF; 7a-7c
healthy/fibrosis viscosity x1.4/1.8/2.6; 7d-7f cirrhosis viscosity
1.2/2.18/2.5; 8 pure-noise probe (runs last per seed, no floor/p-value,
counts/fractions only, identical to 101-104's condition 8).

## Deviations / open judgment calls (disclosed up front)
- Condition 2's correction is the only intentional protocol deviation
  from 101-104, described above.
- Controls trained once per seed on the unshifted training distribution,
  reused across all conditions -- same judgment call as 101-104, same
  rationale (isolates deployment-distribution shift from retraining
  effects; retraining per condition would multiply compute ~7-14x).
- Per rule 6, no significance test is run across only these 4 seeds;
  REPORT.md reports per-seed numbers, per-seed shift vs baseline, and the
  raw median shift only. A combined n=8 analysis (concatenating with
  101-104, condition 2 caveat noted) is left for a follow-on report, not
  computed here.
- Any other flaw found while running is added here or to REPORT.md's
  deviations section as it is found, per rule 4.

## Expected per-seed timing (from 101-104's REPORT.md, verified by
## reading it, not assumed)
101-104's REPORT.md states: "Total wall-clock for all 4 seeds x 19
conditions (18 depth-sweep conditions + noise probe): approx 9 minutes
(seed 101: 136s, seed 104: 137s; well under the 3x-first-seed stop
threshold at every seed)." This branch expects roughly the same
~130-140s/seed on comparable hardware; the actual first-seed (105) time
measured here is the basis for this run's own 3x stop threshold (rule 7),
not the 101-104 figure directly (different container, no guarantee of
identical CPU).

## Environment
numpy==2.2.6, scipy, scikit-learn, matplotlib, torch==2.14.0 installed
from plain PyPI (CPU-only container, CUDA build of torch runs fine on
CPU). `torch.set_num_threads(4)` as in 101-104. See RUNLOG.md for the
exact installed versions and setup log.

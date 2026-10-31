# Battery-A REPORT (seeds 101-104)

Branch: `run battery-101-104`. All code in `audit_fixes/cloud/battery/`.
`Source/` and `audit_fixes/lane5`/`lane8` were never edited (read-only
imports only). PREREG.md was committed before any run (git history is the
timestamp). Regression test (`test_regression.py`) passed before launch:
the baseline wrapper path reproduces `generate_degraded_windows` bit-for-
bit for 3 seeds x 3 classes x 3 depths.

## Self-diagnosis during the run (requested at the time, resolved BEFORE this
## report was finalized)
It was flagged that "real model trained in 0-1s" while 200
controls took ~55-66s looked implausible and could indicate a
silently-untrained real model. Investigated and confirmed NOT a bug:

1. Code path check: `train_cnn` (imported unchanged from
   `degradation_aware_final_sweep.py`) is called identically for the real
   model and every control -- same function, same call site, fresh
   `SimpleCNN()` instantiated inside it every call, 60 epochs, full-batch
   gradient descent (no minibatching). That is the original, unmodified
   design, not something this lane changed.
2. Direct re-run with per-epoch loss logged (same seed-101 training set,
   same architecture, same 60 epochs): loss went 1.1020 (epoch 0, ~ln(3)=
   1.0986, i.e. random-guessing start) -> 1.0943 -> 1.0784 -> 1.0484 ->
   0.9930 -> 0.9158 -> 0.8551 (epoch 59), total wall time 0.306s, resulting
   train accuracy 67.5%. A real, monotonically-improving training curve.
3. Accuracy-vs-depth sanity check (seed 101 baseline, real model): 77.3%
   at 0cm, rising to a peak of 93.3% at 4.5cm, then decaying smoothly to
   chance (~32-38%) by 9.5-15cm -- same shape as the certified sweep
   reports. All 200 shuffled-label controls sit flat at chance (~33%) at
   every depth, as expected.
4. Timing is simply explained: the training set is tiny (375 raw windows)
   and full-batch, so 60 epochs run in a fraction of a second on 4 CPU
   threads; 200 such calls taking ~55-66s (~0.28-0.33s/call) matches a
   direct 10-call microbenchmark (3.17s/10 = 0.317s/call) run before the
   real batch launch.

Conclusion: real-model training is genuine and fast by design (original
`train_cnn`'s full-batch/60-epoch/tiny-dataset architecture), confirmed
with an actual loss curve and accuracy-vs-depth curve, not asserted.

## Condition 3b / 4a / 4b floor=0.00cm anomalies (also investigated)
Three (seed, condition) cells came back with floor_cm=0.00, censored=False
even though real accuracy was clearly and consistently above chance
(67-79%) at the shallowest 1-2 depths: seed 102/3b_spreading_p1.0, seed
104/4a_stiffness_cv_perclass, seed 104/4b_stiffness_cv_uniform.

Root cause, confirmed by inspecting depth_detail_<seed>.csv: a structural
limitation of BH-FDR at N_SHUFFLES=200 combined with a 23-depth grid, not
a degenerate null and not a training bug. The minimum achievable one-sided
permutation p-value is (0+1)/201 = 0.0049751. The strictest BH-FDR
threshold (rank 1 of 23, alpha=0.05) is 1/23*0.05 = 0.002174, which the
floor p-value already exceeds. BH's step-up rule only lets rank-1 pass if
p_(1) itself is <= that threshold, or enough depths cluster at the same
minimal p to raise the rank threshold (k depths at the floor need k >=
ceil(0.0049751*23/0.05) = 3 to pass BH at any of them). In all three
flagged cells only 1-2 depths hit the p-floor, so none of the 23 depths --
including the shallowest, where real accuracy is genuinely far above
chance -- survive BH correction, and the reported "floor" collapses to the
shallowest grid point (0cm) even though the underlying signal is real.
Disclosed as a genuine methodological caveat of this specific
(N_SHUFFLES=200, 23-depth-grid) configuration -- reported, not silently
fixed (raising N_SHUFFLES would change the certified protocol and is out
of scope here).

## Cross-platform check vs the Mac replicate baseline floors
Mac (given): 101=9.55, 102=6.50, 103=9.55, 104=7.50 cm.
This Linux battery's own condition-1 (baseline) floors:
  101: Linux 9.55 vs Mac 9.55 (diff 0.00)
  102: Linux 6.00 vs Mac 6.50 (diff -0.50)
  103: Linux 8.18 vs Mac 9.55 (diff -1.37)
  104: Linux 8.00 vs Mac 7.50 (diff +0.50)
Seed 101 matches exactly. Seeds 102-104 differ by 0.5-1.4cm -- plausible
given cross-platform floating-point nondeterminism (different BLAS/CPU,
torch 2.14.0+cu130 here vs whatever build the Mac run used) propagating
through 201 independently-trained models per seed with the same np/torch
seeds but not bit-identical arithmetic. Not investigated further (reported,
not diagnosed as a mismatch to fix).

## Anomalous recovery (pre-existing phenomenon, also present here)
Seed 103 baseline: depth 8.18cm is NOT BH-significant but depth 9.55cm
immediately after IS significant again (real_acc 0.41 at 8.18cm vs 0.407
at 9.55cm -- both near chance, order flips on noise). The original
degradation_aware_final_sweep.py has its own anomalous_recovery flag for
exactly this pattern; this lane's "floor = first non-significant depth"
convention (matching the task spec) reports 8.18cm as seed 103's baseline
floor. Flagged as the same known phenomenon, not a new bug.

## Per-seed floors by condition (cm)
Full data: results/floors_101-104.csv, results/depth_detail_<seed>.csv
(all 23 depths x raw/BH-adjusted p / significance / control
min-median-max-std), results/confmat_<seed>.npz (3x3 confusion matrices
per condition per depth), results/noise_probe_101-104.csv.

condition                              | 101   | 102   | 103   | 104
1 baseline                             | 9.55  | 6.00  | 8.18  | 8.00
2 amplitude randomisation              | 10.91 | 6.50  | 9.55  | 9.55
3a spreading p=0.5                     | 6.00  | 3.50  | 6.50  | 6.50
3b spreading p=1.0                     | 4.30  | 0.00* | 5.50  | 6.50
4a stiffness CV per-class              | 15.00c| 15.00c| 7.00  | 0.00*
4b stiffness CV=0.15 all               | 15.00c| 5.00  | 4.30  | 0.00*
5a SNR 20dB                            | 8.18  | 5.45  | 8.18  | 9.55
5b SNR 13.58dB                         | 7.50  | 8.00  | 6.00  | 5.50
5c SNR 10dB                            | 8.00  | 6.50  | 5.50  | 5.00
5d SNR 5dB                             | 5.50  | 4.50  | 5.50  | 6.82
5e SNR 0dB                             | 4.30  | 4.00  | 3.50  | 4.00
6a exact attenuation                   | 10.91 | 6.50  | 10.91 | 9.55
6b exact attenuation + exact TOF       | 9.55  | 6.50  | 9.55  | 8.00
7a visc healthy/fib x1.4               | 10.91 | 5.45  | 9.55  | 12.27
7b visc healthy/fib x1.8               | 10.91 | 6.50  | 10.91 | 9.55
7c visc healthy/fib x2.6               | 10.91 | 6.50  | 3.50  | 2.73
7d visc cirrhosis=1.2                  | 15.00c| 8.18  | 13.64 | 15.00c
7e visc cirrhosis=2.18                 | 6.82  | 5.00  | 7.50  | 7.00
7f visc cirrhosis=2.5                  | 6.82  | 4.30  | 7.50  | 7.00

c = right-censored (BH-significant all the way to the 15cm grid max).
* = the BH-structural artifact described above, not a genuine shallow-
    depth collapse (real accuracy at 0-1cm was 67-79%, far above chance).

## Per-seed shift vs baseline (condition floor minus that seed's own
## baseline floor, cm), and plain median shift (n=4, no significance test
## per rule 6)
condition                          | 101   | 102   | 103   | 104   | median
2 amp randomisation                | +1.36 | +0.50 | +1.36 | +1.55 | +1.36
3a spreading p=0.5                 | -3.55 | -2.50 | -1.68 | -1.50 | -2.09
3b spreading p=1.0                 | -5.25 | -6.00*| -2.68 | -1.50 | -3.96 (unreliable, see *)
4a stiffness CV per-class          | +5.45c| +9.00c| -1.18 | -8.00*| unreliable (bimodal)
4b stiffness CV=0.15               | +5.45c| -1.00 | -3.88 | -8.00*| unreliable
5a SNR 20dB                        | -1.36 | -0.55 | 0.00  | +1.55 | -0.28
5b SNR 13.58dB                     | -2.05 | +2.00 | -2.18 | -2.50 | -2.12
5c SNR 10dB                        | -1.55 | +0.50 | -2.68 | -3.00 | -2.12
5d SNR 5dB                         | -4.05 | -1.50 | -2.68 | -1.18 | -2.09
5e SNR 0dB                         | -5.25 | -2.00 | -4.68 | -4.00 | -4.34
6a exact attenuation               | +1.36 | +0.50 | +2.73 | +1.55 | +1.46
6b exact atten+TOF                 | 0.00  | +0.50 | +1.36 | 0.00  | +0.25
7a visc h/f x1.4                   | +1.36 | -0.55 | +1.36 | +4.27 | +1.36
7b visc h/f x1.8                   | +1.36 | +0.50 | +2.73 | +1.55 | +1.46
7c visc h/f x2.6                   | +1.36 | +0.50 | -4.68 | -5.27 | -2.09
7d visc cirr=1.2                   | +5.45c| +2.18 | +5.45 | +7.00c| +5.45
7e visc cirr=2.18                  | -2.73 | -1.00 | -0.68 | -1.00 | -1.00
7f visc cirr=2.5                   | -2.73 | -1.70 | -0.68 | -1.00 | -1.35

## Headline direction of effect (n=4 seeds, directional only, no test statistic)
- Amplitude randomisation (2): consistently DEEPER (+0.5 to +1.55cm, median
  +1.36), all 4 seeds agree in sign -- a random overall gain factor doesn't
  destroy the relative class-dependent attenuation/timing pattern the CNN
  uses.
- Geometric spreading (3a/3b): consistently SHALLOWER (median -2.1cm at
  p=0.5, -3.96cm at p=1.0 though contaminated by one BH-artifact seed) --
  added spreading loss on top of the existing exponential decay
  accelerates signal loss with depth, as expected.
- SNR sweep (5): floor moves shallower as SNR degrades (median -0.3 at
  20dB, roughly -2.1 at 13.58/10/5dB, -4.3 at 0dB), directionally
  consistent though not perfectly monotonic in magnitude across seeds.
- Exact complex-wavenumber attenuation (6): pushes the floor DEEPER in all
  4 seeds (median +1.46cm for 6a, +0.25 for 6b) -- the exact alpha values
  (53.85/25.46/42.92 /m at 100Hz, confirmed MATCH against the task's
  reference numbers) are higher than the weak-damping approximation for
  significant_fibrosis and cirrhosis in this regime, so attenuation-model
  choice alone moves the certified floor by roughly a centimeter, deeper
  not shallower.
- Viscosity (7): healthy/fibrosis eta up to 1.8x pushes DEEPER (median
  +1.36 to +1.46cm); at 2.6x the direction flips negative for 2/4 seeds
  (median -2.09, non-monotonic across seeds -- flagged, not resolved, n=4
  too small to say more). Lowering cirrhosis eta to 1.2 (from 1.77) pushes
  substantially DEEPER (median +5.45cm, 2/4 seeds right-censored at grid
  max). Raising cirrhosis eta (2.18, 2.5) pushes shallower by ~1-1.35cm
  median, the opposite direction, as expected (more viscosity -> more
  attenuation -> harder to classify at depth).
- Within-class stiffness CV (4): reported floors are unreliable per-seed
  (bimodal: some right-censored at grid max, one seed hit the
  BH-structural-artifact floor=0). Real accuracy at shallow-mid depths
  under stiffness variation is still high (67-93%, control std 0.12-0.21,
  not degenerate), so the qualitative finding is "within-class stiffness
  noise did not obviously destroy classifiability" but a reliable
  quantitative floor shift cannot be reported from this configuration for
  this condition.

## Condition 8 -- pure-noise probe (added during the run, run last per seed)
300 pure-Gaussian-noise (no pulse) windows per seed, sigma from the same
SNR->std convention (27.05dB), evaluated by the real model and the 200
controls. Class order (LabelEncoder, alphabetical, confirmed against
sorted(stiffness_classes.keys())): cirrhosis, healthy, significant_fibrosis.

seed | source       | cirrhosis | healthy | significant_fibrosis
101  | real         | 0.06      | 0.94    | 0.00
101  | control mean | 0.326     | 0.329   | 0.345
102  | real         | 0.00      | 1.00    | 0.00
102  | control mean | 0.387     | 0.329   | 0.283
103  | real         | 0.233     | 0.767   | 0.00
103  | control mean | 0.317     | 0.336   | 0.347
104  | real         | 0.52      | 0.48    | 0.00
104  | control mean | 0.433     | 0.252   | 0.315

Full data: results/noise_probe_101-104.csv. Finding: the real (genuinely
trained) model, given pure noise with no pulse at all, never once predicts
significant_fibrosis across any of the 4 seeds, and is heavily biased
toward healthy in 3/4 seeds (94-100%) and split healthy/cirrhosis in seed
104. The 200 shuffled-label controls' mean prediction distribution stays
close to uniform (~0.25-0.43 per class), consistent with each control
having learned an arbitrary decision boundary from its own random label
permutation. A real, disclosable finding about the deployed model's
failure mode on pulse-absent input.

## Addendum: condition 2b -- corrected amplitude-gain distribution
## (rerun on branch run battery-101-104, appended after the run above)

Condition 2's original gain draw was **log-uniform**,
`g = exp(U(ln 0.5, ln 2.0))`, which has `E[g] ~ 1.08 Pa.s` (biased high),
not a mean-1.0 gain randomization -- a design error in the original
PREREG.md, not a bug in the run that executed it faithfully. Condition
**2b** replaces this with `g ~ Uniform(0.5, 1.5)` drawn in **linear**
space (`E[g] = 1.0` exactly), applied at the identical point in the
signal chain (after attenuation, before noise), everything else held
fixed. See `wrappers.py`'s `amp_gain_linear` shift and `config.py`'s
`2b_amp_randomization_linear` condition. Condition 2's own results are
untouched in `results/`; 2b is appended alongside as new rows/columns,
never overwriting.

Model weights from the original run were not persisted to disk (by
design -- only accuracy stats/confusion matrices were kept), so 2b
retrains the real model + 200 shuffled controls per seed from scratch,
using the same `Source/` functions and the same per-seed
`torch.manual_seed(seed)` / `np.random.seed(seed)` convention as
`harness.run_one_seed` (`rerun_condition_2b.py`). The condition-1
baseline regression test (`test_regression.py`) was re-run before this
rerun and still passes bit-for-bit.

### Floors, condition 2 vs 2b, side by side (cm)

condition                              | 101   | 102   | 103   | 104
1 baseline                             | 9.55  | 6.00  | 8.18  | 8.00
2 amp rand, log-uniform (E[g]~1.08)    | 10.91 | 6.50  | 9.55  | 9.55
2b amp rand, linear U(0.5,1.5) (E[g]=1)| 10.91 | 6.00  | 9.55  | 9.55

shift vs baseline (cm)                 | 101   | 102   | 103   | 104   | median
2  (log-uniform)                       | +1.36 | +0.50 | +1.36 | +1.55 | +1.36
2b (linear, corrected)                 | +1.36 | 0.00  | +1.36 | +1.55 | +1.36

### Does "amplitude randomization deepens the floor" survive the correction?

**Mostly, but not unanimously.** Under the corrected (unbiased, mean-1.0)
gain distribution:
- 3 of 4 seeds (101, 103, 104) land on the **exact same floor** as the
  original biased condition 2 -- the discrete 23-point depth grid means
  the reduced gain spread wasn't enough to move the BH-significance
  boundary for these seeds, so the deepening (+1.36 to +1.55cm) holds.
- 1 of 4 seeds (102) shows **no deepening at all** under 2b: floor =
  6.00cm, identical to baseline, versus condition 2's +0.50cm shift. The
  original condition-2 deepening at seed 102 was the smallest of the
  four seeds (+0.50cm, i.e. one grid step) and did not survive removing
  the E[g]~1.08 upward bias.
- The plain median shift is numerically unchanged (+1.36cm for both 2 and
  2b) because it is driven by the majority seeds, not seed 102.

So the finding is directionally robust for 3/4 seeds but its claim to
generalize across "all 4 seeds agree in sign" (as stated in the headline
section above for condition 2) does **not** survive the correction --
under g~U(0.5,1.5) linear, only 3/4 seeds show a positive shift and one
shows zero. With n=4 seeds this is not resolvable further without more
seeds or repeated draws per seed; flagged here rather than silently
patched into the original condition-2 headline text above, which is left
as-is (it correctly describes condition 2 as run, on its own biased
distribution).

### Files (2b-specific)
- `rerun_condition_2b.py` -- retrain + run script for 2b only.
- `results/floors_101-104.csv` -- 2b rows appended (condition id
  `2b_amp_randomization_linear`).
- `results/depth_detail_<seed>.csv` -- 2b's 23-depth rows appended.
- `results/confmat_<seed>.npz` -- 2b's per-depth confusion matrices added
  under `2b_amp_randomization_linear__depth<D>` keys.
- `logs/battery_2b_run.log` -- full 2b rerun log (4 seeds, ~85-95s each
  for real+200 controls retrain + eval, well within the original
  per-seed timing).

## Deviations from the pre-registered protocol
1. Controls are trained once per seed on the ORIGINAL (unshifted) training
   distribution and reused across all conditions (condition changes only
   the test distribution) -- disclosed in PREREG.md before the run.
2. The three floor=0.00cm cells are a genuine BH-FDR structural artifact
   at N_SHUFFLES=200 + 23 depths (described above) -- a limitation
   inherited from the certified sweep's own N_SHUFFLES=200, not a bug
   introduced by this lane's wrappers, reported rather than patched.
3. Per the addition made during the run, condition 8 (pure-noise
   probe) was appended to the protocol and PREREG.md AFTER conditions 1-7
   were already drafted; it ran last for every seed, after conditions 1-7
   were checkpointed and committed, so it never delayed the primary
   conditions.
4. No significance test was run across only these 4 seeds, per rule 6.

## Files
- PREREG.md, RUNLOG.md -- written/committed before any run.
- config.py, wrappers.py, harness.py, test_regression.py -- all new code;
  Source/ and lane5/lane8 untouched.
- results/floors_101-104.csv, results/depth_detail_<seed>.csv,
  results/confmat_<seed>.npz, results/noise_probe_101-104.csv.
- logs/battery_run.log, logs/harness_stdout.log -- full run logs.

Total wall-clock for all 4 seeds x 19 conditions (18 depth-sweep conditions
+ noise probe): approx 9 minutes (seed 101: 136s, seed 104: 137s; well
under the 3x-first-seed stop threshold at every seed).

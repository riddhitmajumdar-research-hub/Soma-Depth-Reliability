# Battery-B REPORT (seeds 105-108)

Branch: run battery-105-108. All code in audit_fixes/cloud/battery/.
Source/ and audit_fixes/lane5/lane8 were never edited (read-only
imports only). PREREG.md was committed before any run (git history is the
timestamp). Regression test (test_regression.py) passed before launch:
27/27 (3 seeds x 3 classes x 3 depths) reproduced
generate_degraded_windows bit-for-bit.

## Protocol
Replicates run battery-101-104's 19-condition protocol (1, 2, 3a/3b,
4a/4b, 5a-5e, 6a/6b, 7a-7f, 8), same priority order, same per-seed/
per-condition checkpointing, same CSV schema, for seeds 105-108. The ONE
intentional deviation (as specified) is condition 2: this branch
uses the CORRECTED linear-space gain draw g ~ Uniform(0.5, 1.5)
(E[g] = 1.0 exactly) in place of 101-104's original condition-2 draw
g = exp(U(ln 0.5, ln 2)) (log-uniform, E[g] ~ 1.08, a design error
101-104 found and corrected in its own "2b" addendum). Seeds 105-108 only
ever ran the corrected version, labeled condition id 2_amp_randomization
(no "2 vs 2b" split needed on this branch).

Important comparability caveat: this branch's condition-2 rows are
comparable to 101-104's 2b_amp_randomization_linear rows (same
distribution), NOT to 101-104's own 2_amp_randomization rows (biased
log-uniform draw). Any n=8 concatenation of condition 2 across both
branches must use 101-104's 2b rows, not its original condition-2 rows,
or the pooled sample mixes two different gain distributions.

## Timing
First seed (105): 208s (train=17s, controls=98s, 18 depth-sweep conditions
+ noise probe = ~93s). Subsequent seeds: 106=176s, 107=172s, 108=190s. All
well under the 3x-first-seed stop threshold (624s); no STATUS.md was
written, no seed failed, no retries were needed.

101-104's own REPORT.md reports "seed 101: 136s, seed 104: 137s" (~9 min
total for 4 seeds). This run's per-seed times (172-208s, ~12.4 min total)
run noticeably slower on this container -- real-model training alone took
17s for seed 105 vs 101-104's reported <1s, though it dropped to ~1s for
seeds 106-108 (likely first-call JIT/cache warmup rather than a real
per-seed difference; controls training, the dominant cost, tracked close
to 101-104's pace: ~77-98s here vs ~55-66s there for 200 controls,
consistent with a somewhat busier/slower CPU in this container, not a
qualitatively different regime). Disclosed as an observed deviation from
the 101-104 figure, not treated as evidence of a bug.

## BH-FDR floor=0.00cm structural artifact (same phenomenon as 101-104,
## reproduced here, verified not silently assumed)
Two (seed, condition) cells came back floor_cm=0.00, censored=False even
though real accuracy is well above chance (68-75%) at the shallowest
depths: seed 106/4b_stiffness_cv_uniform, seed 107/3b_spreading_p1.0.
Checked depth_detail_<seed>.csv directly for both:

- seed 106, 4b, depth 0cm: real_acc=0.727, raw_p=0.00498 (the minimum
  achievable p at N_SHUFFLES=200), BH-adjusted p=0.053, NOT significant.
- seed 107, 3b, depth 0cm: real_acc=0.750, raw_p=0.00995, BH-adjusted
  p=0.114, NOT significant.

Same root cause as 101-104's REPORT.md: the minimum one-sided permutation
p-value at N_SHUFFLES=200 is (0+1)/201=0.0049751, and BH's step-up rule at
rank 1 of 23 (alpha=0.05) requires p_(1) <= 1/23*0.05=0.002174 (or >=3
depths tied at the p-floor) to survive -- neither cell has enough depths
at the minimal p, so none of the 23 depths pass BH even though the
shallowest depths show genuinely far-above-chance accuracy. Reported, not
silently fixed (raising N_SHUFFLES is out of scope and would break
comparability with 101-104 and the certified sweep).

## Per-seed floors by condition (cm)
Full data: results/floors_105-108.csv, results/depth_detail_<seed>.csv
(all 23 depths x raw/BH-adjusted p / significance / control
min-median-max-std), results/confmat_<seed>.npz,
results/noise_probe_105-108.csv.

condition                              | 105   | 106   | 107   | 108
1 baseline                             | 8.00  | 9.55  | 7.50  | 10.91
2 amp randomisation (corrected linear) | 8.18  | 10.91 | 8.00  | 13.64
3a spreading p=0.5                     | 6.50  | 6.00  | 1.36  | 15.00c
3b spreading p=1.0                     | 4.30  | 4.00  | 0.00* | 7.50
4a stiffness CV per-class              | 15.00c| 4.00  | 15.00c| 15.00c
4b stiffness CV=0.15 all               | 3.50  | 0.00* | 6.82  | 15.00c
5a SNR 20dB                            | 8.00  | 7.50  | 8.00  | 7.50
5b SNR 13.58dB                         | 7.50  | 6.82  | 7.00  | 6.82
5c SNR 10dB                            | 7.50  | 6.00  | 5.00  | 6.00
5d SNR 5dB                             | 5.00  | 5.45  | 4.30  | 4.30
5e SNR 0dB                             | 4.50  | 3.50  | 3.50  | 3.50
6a exact attenuation                   | 8.18  | 8.18  | 8.00  | 13.64
6b exact attenuation + exact TOF       | 8.00  | 9.55  | 7.50  | 9.55
7a visc healthy/fib x1.4               | 10.91 | 9.55  | 6.50  | 13.64
7b visc healthy/fib x1.8               | 10.91 | 10.91 | 8.00  | 12.27
7c visc healthy/fib x2.6               | 10.91 | 9.55  | 9.55  | 13.64
7d visc cirrhosis=1.2                  | 15.00c| 15.00c| 10.91 | 15.00c
7e visc cirrhosis=2.18                 | 7.00  | 6.82  | 6.50  | 7.50
7f visc cirrhosis=2.5                  | 6.50  | 6.50  | 1.36  | 7.50

c = right-censored (BH-significant all the way to the 15cm grid max).
* = the BH-structural artifact described above, not a genuine shallow-
    depth collapse (real accuracy at 0cm was 68-75%, far above chance).

## Per-seed shift vs baseline (condition floor minus that seed's own
## baseline floor, cm), plain median shift (n=4, no significance test
## per rule 6)
condition                          | 105   | 106   | 107   | 108   | median
2 amp randomisation (corrected)    | +0.18 | +1.36 | +0.50 | +2.73 | +0.93
3a spreading p=0.5                 | -1.50 | -3.55 | -6.14 | +4.09 | -2.52
3b spreading p=1.0                 | -3.70 | -5.55 | -7.50 | -3.41 | -4.62
4a stiffness CV per-class          | +7.00c| -5.55 | +7.50c| +4.09c| +5.55 (unreliable, bimodal)
4b stiffness CV=0.15               | -4.50 | -9.55*| -0.68 | +4.09c| -2.59 (unreliable)
5a SNR 20dB                        | 0.00  | -2.05 | +0.50 | -3.41 | -1.02
5b SNR 13.58dB                     | -0.50 | -2.73 | -0.50 | -4.09 | -1.61
5c SNR 10dB                        | -0.50 | -3.55 | -2.50 | -4.91 | -3.02
5d SNR 5dB                         | -3.00 | -4.09 | -3.20 | -6.61 | -3.65
5e SNR 0dB                         | -3.50 | -6.05 | -4.00 | -7.41 | -5.02
6a exact attenuation               | +0.18 | -1.36 | +0.50 | +2.73 | +0.34
6b exact atten+TOF                 | 0.00  | 0.00  | 0.00  | -1.36 | 0.00
7a visc h/f x1.4                   | +2.91 | 0.00  | -1.00 | +2.73 | +1.36
7b visc h/f x1.8                   | +2.91 | +1.36 | +0.50 | +1.36 | +1.36
7c visc h/f x2.6                   | +2.91 | 0.00  | +2.05 | +2.73 | +2.39
7d visc cirr=1.2                   | +7.00c| +5.45c| +3.41 | +4.09c| +4.77
7e visc cirr=2.18                  | -1.00 | -2.73 | -1.00 | -3.41 | -1.86
7f visc cirr=2.5                   | -1.50 | -3.05 | -6.14 | -3.41 | -3.23

Note: seed 108's baseline floor (10.91cm) is itself the highest of the
four seeds, so its condition shifts, especially 3a (+4.09) and 4b (+4.09,
right-censored), diverge in sign from the other three seeds more often
than in 101-104's battery. This is disclosed as a per-seed effect, not
adjusted for.

## Headline direction of effect (n=4 seeds, directional only, no test
## statistic, consistent with 101-104's qualitative findings)
- Amplitude randomisation (2, corrected): DEEPER in all 4 seeds
  (+0.18 to +2.73cm, median +0.93), same sign as 101-104's original
  condition 2 (median +1.36) and its own corrected 2b rerun (median
  +1.36, 3/4 seeds positive, 1 zero). Combined with 101-104's 2b result,
  an n=8 pooled view (using 101-104's 2b rows, not its original
  condition-2 rows, per the caveat above) would show 7/8 seeds with a
  nonnegative shift, 1/8 exactly zero, none negative.
- Geometric spreading (3a/3b): mostly SHALLOWER (median -2.52 at
  p=0.5, -4.62 at p=1.0), matching 101-104's direction, but NOT unanimous
  here -- seed 108's 3a shift is +4.09 (right-censored, opposite sign),
  the only spreading-condition sign flip across both batteries' seeds.
  Flagged, not resolved.
- SNR sweep (5): floor moves shallower as SNR degrades, monotonically
  in the median (-1.02 at 20dB down to -5.02 at 0dB), matching 101-104's
  shape closely.
- Exact complex-wavenumber attenuation (6): pushes DEEPER in the
  median for 6a (+0.34) though only 3/4 seeds positive (106 is -1.36);
  6b's median shift is exactly 0.00 (3/4 seeds show no shift at all under
  the 23-point grid's resolution, 1 seed -1.36). Weaker and less
  unanimous than 101-104's 6a median +1.46 / 6b +0.25, both of which
  agreed in sign for all 4 seeds there.
- Viscosity (7): healthy/fibrosis eta increases (7a-7c) push DEEPER
  in the median (+1.36, +1.36, +2.39), matching 101-104's direction and
  NOT showing the sign flip 101-104 saw at 7c for 2/4 seeds -- here all
  4 seeds are nonnegative at 7c. Lowering cirrhosis eta to 1.2 (7d)
  pushes substantially DEEPER (median +4.77, 3/4 seeds right-censored),
  matching 101-104's strong deepening (+5.45, 2/4 censored). Raising
  cirrhosis eta (7e, 7f) pushes SHALLOWER (median -1.86, -3.23), same
  direction as 101-104 (-1.00, -1.35) but a noticeably larger magnitude
  here, especially seed 107's 7f (-6.14).
- Within-class stiffness CV (4): unreliable per-seed here too
  (bimodal: some right-censored, two seeds hit the BH-structural
  artifact), consistent with 101-104's own disclosed limitation for this
  condition -- no reliable quantitative floor shift reportable from this
  configuration.

## Condition 8 -- pure-noise probe (run last per seed, same convention as
## 101-104: 300 pure-Gaussian-noise windows, sigma from SNR->std at
## 27.05dB, class order alphabetical: cirrhosis, healthy,
## significant_fibrosis)

seed | source       | cirrhosis | healthy | significant_fibrosis
105  | real         | 0.140     | 0.860   | 0.000
105  | control mean | 0.360     | 0.338   | 0.302
106  | real         | 0.077     | 0.923   | 0.000
106  | control mean | 0.321     | 0.400   | 0.279
107  | real         | 0.003     | 0.997   | 0.000
107  | control mean | 0.318     | 0.344   | 0.338
108  | real         | 0.117     | 0.883   | 0.000
108  | control mean | 0.296     | 0.341   | 0.364

Full data: results/noise_probe_105-108.csv. Finding replicates
101-104's exactly: the real, genuinely trained model never once predicts
significant_fibrosis on pure noise across any of the 4 seeds, and is now
even MORE heavily biased toward healthy than 101-104 was (86.0-99.7% here
vs 48-100% there, with 101-104's seed 104 being a near-even
healthy/cirrhosis split at 0.48/0.52 -- no seed here shows anywhere near
that even a split; the closest, seed 105, is still 86% healthy). The 200
shuffled-label controls again stay close to uniform (~0.28-0.40 per
class). Combined across both batteries (n=8 seeds), the real model has
NEVER once predicted significant_fibrosis on pure noise, a consistent
finding across 8 independently seeded/trained real models.

## Cross-check: 105-108 baseline floors vs 101-104's own baseline floors
(no external Mac replicate baseline was given for seeds 105-108 in the
task, unlike 101-104; this section compares this battery's baseline
spread to 101-104's own Linux baseline spread, for completeness only)
101-104 baseline (this Linux container, per 101-104's REPORT.md): 9.55,
6.00, 8.18, 8.00cm (mean 7.93, range 3.55).
105-108 baseline (this run): 8.00, 9.55, 7.50, 10.91cm (mean 8.99, range
3.41). Comparable spread and magnitude to 101-104's baseline; no anomaly.

## Deviations from the pre-registered protocol
1. Condition 2 uses the corrected linear-space gain draw only, as
   specified -- the intended, disclosed deviation from 101-104's
   original condition 2 (comparable instead to 101-104's own 2b).
2. Controls are trained once per seed on the ORIGINAL (unshifted) training
   distribution and reused across all conditions -- same as 101-104,
   disclosed in this branch's PREREG.md before the run.
3. Two floor=0.00cm cells (seed 106/4b, seed 107/3b) are the same
   BH-FDR-at-N_SHUFFLES=200 structural artifact 101-104 found -- not a
   bug in this lane's wrappers, verified directly against
   depth_detail_<seed>.csv rather than assumed to match 101-104's finding.
4. Per-seed timing (172-208s) ran somewhat slower than 101-104's reported
   136-137s, well inside the 3x-first-seed stop threshold in every case;
   attributed to container/CPU variability, not investigated further
   (disclosed, not diagnosed as a bug).
5. Condition 3a's sign flip at seed 108 (+4.09cm, opposite direction from
   the other 3 seeds and from all of 101-104's 4 seeds) is flagged as an
   open finding, not resolved -- n=4 (or even n=8 pooled) is too small to
   say whether this is noise or a real effect specific to seed 108's
   particular real-model/control-population draw.
6. No significance test was run across only these 4 seeds, per rule 6. No
   combined n=8 statistical test was run either (out of scope per the
   task's "Stop after REPORT.md"); the n=8 pooled observations stated
   above (condition 2 and condition 8) are directional/descriptive only.

## Files
- PREREG.md, RUNLOG.md -- written/committed before any run.
- config.py, wrappers.py, harness.py, test_regression.py -- config.py
  edited for seeds 105-108 and the corrected condition 2; wrappers.py,
  harness.py (except SEED_TAG), test_regression.py copied unchanged from
  run battery-101-104. Source/ and lane5/lane8 untouched.
- results/floors_105-108.csv, results/depth_detail_<seed>.csv,
  results/confmat_<seed>.npz, results/noise_probe_105-108.csv.
- logs/battery_run.log, logs/harness_stdout.log -- full run logs.

Total wall-clock for all 4 seeds x 19 conditions (18 depth-sweep conditions
+ noise probe): 208 + 176 + 172 + 190 = 746s (~12.4 minutes). Every seed
finished well under the 3x-first-seed (624s) stop threshold; no seed
failed; no STATUS.md was written.

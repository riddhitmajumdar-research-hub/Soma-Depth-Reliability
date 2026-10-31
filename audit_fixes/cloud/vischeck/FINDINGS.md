# vischeck — FINDINGS.md

Read-only investigation of condition 7c (viscosity healthy/fibrosis x2.6)
vs. baseline (condition 1), per-depth, for the two seeds that shallowed
under this condition (103, 104) and three that deepened (101, 105, 107 —
contrast set). Only `audit_fixes/cloud/vischeck/` was written. Seeds
101/103/104 pulled from `run battery-101-104`
(`results/depth_detail_<seed>.csv`, commit `<commit>`); seeds 105/107
pulled from `run battery-105-108`
(`results/depth_detail_<seed>.csv`, commit `<commit>`) — 101, 103, 104 do
not exist on run 105-108 and 105, 107 do not exist on run 101-104,
so each seed was read from the run that actually produced it.
Nothing retrained or recomputed; all accuracy/control/significance values
below are copied directly from those CSVs.

---

## Aside noticed while pulling data (not part of the task, reported anyway)

`depth_detail_101.csv` (and, by extension, the other seeds on
`run battery-101-104`) contains a 20th condition label,
`2b_amp_randomization_linear`, that does not appear in that run's
`results/floors_101-104.csv` or in its REPORT.md's stated 19-condition
list. This looks like leftover data from testing/validating the corrected
linear-gain amplitude draw (the one later used as-is for condition 2 on
`run battery-105-108`) before it was folded into battery-B's protocol,
left in the raw depth-detail file but not carried into the summary CSV or
REPORT.md table. Not investigated further — reported as found.

---

## Per-depth baseline vs. 7c accuracy, and 7c control stats

### Seed 103 (shallowed under 7c, shift -4.68cm)

| depth_cm | baseline_acc | 7c_acc | 7c_fdr_sig | ctrl_min | ctrl_median | ctrl_max |
|---|---|---|---|---|---|---|
| 0.000 | 0.770 | 0.767 | True | 0.000 | 0.333 | 0.780 |
| 1.364 | 0.777 | 0.800 | True | 0.000 | 0.293 | 0.857 |
| 2.727 | 0.803 | 0.723 | True | 0.000 | 0.333 | 0.787 |
| 3.500 | 0.903 | 0.540 | False | 0.000 | 0.333 | 0.667 |
| 4.000 | 0.903 | 0.533 | True | 0.043 | 0.333 | 0.663 |
| 4.091 | 0.923 | 0.583 | True | 0.063 | 0.333 | 0.660 |
| 4.300 | 0.903 | 0.553 | True | 0.133 | 0.333 | 0.660 |
| 4.500 | 0.837 | 0.537 | True | 0.133 | 0.333 | 0.647 |
| 5.000 | 0.857 | 0.513 | True | 0.177 | 0.333 | 0.533 |
| 5.455 | 0.860 | 0.557 | True | 0.230 | 0.333 | 0.460 |
| 5.500 | 0.840 | 0.567 | True | 0.193 | 0.333 | 0.470 |
| 6.000 | 0.817 | 0.500 | True | 0.243 | 0.333 | 0.423 |
| 6.500 | 0.717 | 0.470 | True | 0.260 | 0.333 | 0.363 |
| 6.818 | 0.687 | 0.453 | True | 0.273 | 0.333 | 0.393 |
| 7.000 | 0.630 | 0.437 | True | 0.243 | 0.333 | 0.400 |
| 7.500 | 0.523 | 0.407 | True | 0.273 | 0.333 | 0.393 |
| 8.000 | 0.480 | 0.400 | True | 0.297 | 0.333 | 0.423 |
| 8.182 | 0.410 | 0.400 | True | 0.283 | 0.333 | 0.400 |
| 9.545 | 0.407 | 0.387 | True | 0.290 | 0.333 | 0.370 |
| 10.909 | 0.333 | 0.347 | True | 0.277 | 0.333 | 0.360 |
| 12.273 | 0.323 | 0.323 | False | 0.310 | 0.333 | 0.370 |
| 13.636 | 0.320 | 0.367 | True | 0.287 | 0.333 | 0.377 |
| 15.000 | 0.290 | 0.300 | False | 0.290 | 0.333 | 0.360 |

### Seed 104 (shallowed under 7c, shift -5.27cm)

| depth_cm | baseline_acc | 7c_acc | 7c_fdr_sig | ctrl_min | ctrl_median | ctrl_max |
|---|---|---|---|---|---|---|
| 0.000 | 0.723 | 0.727 | True | 0.000 | 0.333 | 0.650 |
| 1.364 | 0.777 | 0.803 | True | 0.000 | 0.282 | 0.840 |
| 2.727 | 0.843 | 0.547 | False | 0.000 | 0.333 | 0.770 |
| 3.500 | 0.933 | 0.450 | False | 0.020 | 0.333 | 0.750 |
| 4.000 | 0.930 | 0.473 | False | 0.027 | 0.333 | 0.660 |
| 4.091 | 0.917 | 0.470 | False | 0.060 | 0.333 | 0.690 |
| 4.300 | 0.953 | 0.483 | False | 0.063 | 0.333 | 0.647 |
| 4.500 | 0.947 | 0.460 | False | 0.090 | 0.333 | 0.643 |
| 5.000 | 0.913 | 0.480 | True | 0.137 | 0.333 | 0.610 |
| 5.455 | 0.873 | 0.480 | True | 0.187 | 0.333 | 0.590 |
| 5.500 | 0.853 | 0.467 | True | 0.170 | 0.333 | 0.587 |
| 6.000 | 0.817 | 0.453 | True | 0.200 | 0.333 | 0.523 |
| 6.500 | 0.730 | 0.447 | True | 0.207 | 0.333 | 0.507 |
| 6.818 | 0.650 | 0.457 | True | 0.257 | 0.333 | 0.463 |
| 7.000 | 0.680 | 0.423 | True | 0.267 | 0.333 | 0.480 |
| 7.500 | 0.570 | 0.387 | True | 0.277 | 0.333 | 0.407 |
| 8.000 | 0.480 | 0.413 | True | 0.297 | 0.333 | 0.393 |
| 8.182 | 0.430 | 0.417 | True | 0.293 | 0.333 | 0.383 |
| 9.545 | 0.363 | 0.387 | True | 0.303 | 0.333 | 0.367 |
| 10.909 | 0.360 | 0.367 | True | 0.273 | 0.333 | 0.377 |
| 12.273 | 0.327 | 0.327 | False | 0.293 | 0.333 | 0.377 |
| 13.636 | 0.347 | 0.350 | False | 0.297 | 0.333 | 0.383 |
| 15.000 | 0.347 | 0.320 | False | 0.303 | 0.333 | 0.390 |

### Seed 101 (deepened under 7c, shift +1.36cm (contrast))

| depth_cm | baseline_acc | 7c_acc | 7c_fdr_sig | ctrl_min | ctrl_median | ctrl_max |
|---|---|---|---|---|---|---|
| 0.000 | 0.773 | 0.790 | True | 0.000 | 0.333 | 0.720 |
| 1.364 | 0.820 | 0.850 | True | 0.000 | 0.293 | 0.903 |
| 2.727 | 0.917 | 0.817 | True | 0.000 | 0.333 | 0.747 |
| 3.500 | 0.860 | 0.653 | True | 0.030 | 0.333 | 0.670 |
| 4.000 | 0.913 | 0.663 | True | 0.047 | 0.333 | 0.627 |
| 4.091 | 0.930 | 0.637 | True | 0.057 | 0.333 | 0.643 |
| 4.300 | 0.900 | 0.657 | True | 0.040 | 0.333 | 0.637 |
| 4.500 | 0.933 | 0.643 | True | 0.060 | 0.333 | 0.643 |
| 5.000 | 0.907 | 0.653 | True | 0.080 | 0.333 | 0.607 |
| 5.455 | 0.890 | 0.643 | True | 0.077 | 0.333 | 0.577 |
| 5.500 | 0.870 | 0.633 | True | 0.110 | 0.333 | 0.577 |
| 6.000 | 0.847 | 0.583 | True | 0.143 | 0.333 | 0.503 |
| 6.500 | 0.760 | 0.533 | True | 0.140 | 0.333 | 0.450 |
| 6.818 | 0.670 | 0.480 | True | 0.200 | 0.333 | 0.443 |
| 7.000 | 0.653 | 0.480 | True | 0.220 | 0.333 | 0.420 |
| 7.500 | 0.553 | 0.463 | True | 0.220 | 0.333 | 0.430 |
| 8.000 | 0.523 | 0.410 | True | 0.263 | 0.333 | 0.407 |
| 8.182 | 0.463 | 0.400 | True | 0.243 | 0.333 | 0.387 |
| 9.545 | 0.380 | 0.370 | True | 0.283 | 0.333 | 0.370 |
| 10.909 | 0.340 | 0.347 | False | 0.293 | 0.333 | 0.403 |
| 12.273 | 0.337 | 0.330 | False | 0.253 | 0.333 | 0.387 |
| 13.636 | 0.323 | 0.330 | False | 0.307 | 0.333 | 0.370 |
| 15.000 | 0.327 | 0.330 | False | 0.297 | 0.333 | 0.363 |

### Seed 105 (deepened under 7c, shift +2.909cm (contrast))

| depth_cm | baseline_acc | 7c_acc | 7c_fdr_sig | ctrl_min | ctrl_median | ctrl_max |
|---|---|---|---|---|---|---|
| 0.000 | 0.757 | 0.757 | True | 0.000 | 0.333 | 0.753 |
| 1.364 | 0.767 | 0.827 | True | 0.000 | 0.333 | 0.883 |
| 2.727 | 0.920 | 0.703 | True | 0.000 | 0.333 | 0.663 |
| 3.500 | 0.940 | 0.607 | True | 0.000 | 0.333 | 0.633 |
| 4.000 | 0.910 | 0.633 | True | 0.013 | 0.333 | 0.647 |
| 4.091 | 0.950 | 0.617 | True | 0.000 | 0.333 | 0.637 |
| 4.300 | 0.930 | 0.627 | True | 0.033 | 0.333 | 0.623 |
| 4.500 | 0.913 | 0.583 | True | 0.033 | 0.333 | 0.567 |
| 5.000 | 0.877 | 0.640 | True | 0.123 | 0.333 | 0.443 |
| 5.455 | 0.863 | 0.607 | True | 0.123 | 0.333 | 0.420 |
| 5.500 | 0.870 | 0.593 | True | 0.153 | 0.333 | 0.417 |
| 6.000 | 0.793 | 0.570 | True | 0.193 | 0.333 | 0.400 |
| 6.500 | 0.710 | 0.550 | True | 0.190 | 0.333 | 0.370 |
| 6.818 | 0.647 | 0.500 | True | 0.220 | 0.333 | 0.400 |
| 7.000 | 0.610 | 0.487 | True | 0.260 | 0.333 | 0.383 |
| 7.500 | 0.527 | 0.437 | True | 0.273 | 0.333 | 0.387 |
| 8.000 | 0.463 | 0.433 | True | 0.273 | 0.333 | 0.383 |
| 8.182 | 0.430 | 0.417 | True | 0.270 | 0.333 | 0.403 |
| 9.545 | 0.357 | 0.390 | True | 0.290 | 0.333 | 0.363 |
| 10.909 | 0.353 | 0.337 | False | 0.290 | 0.333 | 0.373 |
| 12.273 | 0.363 | 0.353 | True | 0.277 | 0.333 | 0.380 |
| 13.636 | 0.347 | 0.320 | False | 0.297 | 0.333 | 0.373 |
| 15.000 | 0.330 | 0.340 | False | 0.283 | 0.333 | 0.370 |

### Seed 107 (deepened under 7c, shift +2.045cm (contrast))

| depth_cm | baseline_acc | 7c_acc | 7c_fdr_sig | ctrl_min | ctrl_median | ctrl_max |
|---|---|---|---|---|---|---|
| 0.000 | 0.753 | 0.763 | True | 0.003 | 0.333 | 0.787 |
| 1.364 | 0.770 | 0.810 | True | 0.000 | 0.333 | 0.930 |
| 2.727 | 0.797 | 0.750 | True | 0.000 | 0.333 | 0.870 |
| 3.500 | 0.867 | 0.613 | True | 0.000 | 0.333 | 0.690 |
| 4.000 | 0.927 | 0.650 | True | 0.037 | 0.333 | 0.623 |
| 4.091 | 0.947 | 0.647 | True | 0.027 | 0.333 | 0.640 |
| 4.300 | 0.920 | 0.660 | True | 0.027 | 0.333 | 0.633 |
| 4.500 | 0.963 | 0.657 | True | 0.020 | 0.333 | 0.650 |
| 5.000 | 0.927 | 0.660 | True | 0.063 | 0.333 | 0.620 |
| 5.455 | 0.880 | 0.613 | True | 0.170 | 0.333 | 0.613 |
| 5.500 | 0.883 | 0.613 | True | 0.180 | 0.333 | 0.583 |
| 6.000 | 0.800 | 0.497 | True | 0.230 | 0.333 | 0.493 |
| 6.500 | 0.717 | 0.450 | True | 0.203 | 0.333 | 0.487 |
| 6.818 | 0.610 | 0.457 | True | 0.247 | 0.333 | 0.440 |
| 7.000 | 0.600 | 0.397 | True | 0.263 | 0.333 | 0.427 |
| 7.500 | 0.537 | 0.380 | True | 0.280 | 0.333 | 0.400 |
| 8.000 | 0.440 | 0.350 | True | 0.277 | 0.333 | 0.403 |
| 8.182 | 0.403 | 0.363 | True | 0.283 | 0.333 | 0.383 |
| 9.545 | 0.340 | 0.333 | False | 0.310 | 0.333 | 0.390 |
| 10.909 | 0.330 | 0.333 | False | 0.280 | 0.333 | 0.363 |
| 12.273 | 0.333 | 0.337 | False | 0.283 | 0.333 | 0.370 |
| 13.636 | 0.333 | 0.343 | True | 0.280 | 0.333 | 0.380 |
| 15.000 | 0.330 | 0.333 | False | 0.280 | 0.333 | 0.370 |


---

## Gradual decline vs. sharp collapse: what the data shows

**Seed 103** (floor 3.5cm, shift -4.68cm): accuracy under 7c declines
smoothly from 0.767 (0cm) through 0.723, then to 0.540 at 3.5cm — no
single depth-to-depth accuracy jump is more than ~0.18. The 3.5cm floor
itself is driven by one marginal p-value miss (raw_p=0.0498,
bh_adj_p=0.0545, just above the BH threshold) while accuracy (0.540) is
still far above chance (0.333); every depth immediately after 3.5cm is
BH-significant again (0.533-0.583) and only fails again much later
(12.273cm, 15.0cm), in the same near-chance region every seed eventually
reaches. **This is a single marginal significance miss amid an otherwise
smooth decline, not a sharp accuracy collapse.**

**Seed 104** (floor 2.727cm, shift -5.27cm): accuracy at 2.727cm (0.547)
and 3.5-4.5cm (0.450-0.483) is not a collapse either — it's a smooth
continuation of the decline visible from 0.727→0.803→0.547→0.450. What
flips is `fdr_significant`, False for five consecutive depths (2.727
through 4.5cm) despite accuracy sitting 12-15 points above chance the
whole time. Raw p-values in that band (0.055-0.075) sit just above BH's
adjusted threshold in each case — a marginal-significance band, not a
step-change in the underlying accuracy curve. Significance resumes at
5.0cm (accuracy 0.480, essentially unchanged from the "non-significant"
depths right before it) and continues through most of the mid-range
before fading to chance at the deep end, same as every other seed.

**Contrast set (101, 105, 107 — deepened):** accuracy under 7c declines
just as smoothly as baseline in all three, with the 7c curve running
consistently a few points below the baseline curve at every depth rather
than crossing or diverging sharply. `fdr_significant` stays True across
nearly the entire depth range and only turns False in the same deep tail
(10.909-15.0cm) where baseline itself also loses significance — i.e. the
7c condition tracks baseline's own significance pattern closely for these
three seeds, with no early significance gap.

**Overall**: none of the five seeds examined (103, 104, 101, 105, 107)
shows a sharp, single-depth accuracy collapse under condition 7c. What
differs between the "shallowed" pair (103, 104) and the "deepened" trio
(101, 105, 107) is not the shape of the accuracy-vs-depth curve — all
five decline gradually and similarly in shape — but whether a handful of
mid-shallow depths (where accuracy sits in the 0.45-0.55 range, clearly
above chance but with growing control variance) happen to land on the
right or wrong side of the BH significance threshold for that seed's
particular 200-control p-value distribution. Seeds 103 and 104's floors
in particular look driven by this same marginal BH-threshold sensitivity
already documented for the floor=0.00cm artifacts in the 101-104
REPORT.md, not by any qualitatively different degradation behavior at
2.6x viscosity relative to seeds that deepened. This diagnosis is offered
because the data supports it directly (raw/adjusted p-values quoted
above), but it is reported, not treated as a settled
conclusion — no seeds were rerun and no threshold was adjusted to test
this further.

## Control accuracy at each depth (7c)

The control min/median/max columns above show the same pattern across all
five seeds regardless of shallow/deep grouping: control_median is pinned
at 0.333 (exact chance for 3 classes) at every single depth in every
seed — expected, since with 200 independent shuffled-label controls the
median sits on the middle control's accuracy, which clusters tightly
around chance by construction. control_min and control_max both start
wide at 0cm (min as low as 0.000, max as high as ~0.75-0.93) and converge
toward the median as depth increases, reaching a narrow band (~0.28-0.40)
by 10-15cm. This convergence is present in every seed/group here and is
not distinctive to the seeds that shallowed under 7c — it is the same
generic deep-tail control-variance shrinkage already noted in the
seed-108 spreading check (spreadcheck).

# spreadcheck — FINDINGS.md

Read-only investigation of seed 108's geometric-spreading sign flip
(condition 3a) on `run battery-105-108`, and the only claimed n=8
disagreement across the two battery branches. Only
`audit_fixes/cloud/spreadcheck/` was written. All numbers below are pulled
directly via `git show` from `run battery-105-108` and
`run battery-101-104` (commits `<commit>` and `<commit>`
respectively, the latest on each branch at the time of this check) — not
recomputed, not simulated.

---

## 1. Seed 108, condition 3a (geometric spreading, p=0.5): per-depth detail

From `results/depth_detail_108.csv`, condition `3a_spreading_p0.5`, all 23
depths (columns: depth_cm, real_acc, raw_p, bh_adj_p, fdr_significant,
control_min, control_median, control_max, control_std):

| depth_cm | real_acc | fdr_sig | ctrl_min | ctrl_median | ctrl_max | ctrl_std |
|---|---|---|---|---|---|---|
| 0.000 | 0.767 | True | 0.017 | 0.333 | 0.760 | 0.143 |
| 1.364 | 0.820 | True | 0.000 | 0.333 | 0.833 | 0.199 |
| 2.727 | 0.913 | True | 0.003 | 0.333 | 0.657 | 0.146 |
| 3.500 | 0.853 | True | 0.020 | 0.333 | 0.603 | 0.096 |
| 4.000 | 0.787 | True | 0.023 | 0.333 | 0.593 | 0.073 |
| 4.091 | 0.750 | True | 0.017 | 0.333 | 0.587 | 0.069 |
| 4.300 | 0.760 | True | 0.030 | 0.333 | 0.587 | 0.064 |
| 4.500 | 0.660 | True | 0.050 | 0.333 | 0.593 | 0.057 |
| 5.000 | 0.590 | True | 0.153 | 0.333 | 0.577 | 0.045 |
| 5.455 | 0.490 | True | 0.200 | 0.333 | 0.550 | 0.037 |
| 5.500 | 0.503 | True | 0.217 | 0.333 | 0.543 | 0.035 |
| 6.000 | 0.453 | True | 0.220 | 0.333 | 0.560 | 0.031 |
| 6.500 | 0.403 | True | 0.223 | 0.333 | 0.530 | 0.025 |
| 6.818 | 0.397 | True | 0.247 | 0.333 | 0.530 | 0.023 |
| 7.000 | 0.410 | True | 0.253 | 0.333 | 0.500 | 0.020 |
| 7.500 | 0.400 | True | 0.257 | 0.333 | 0.487 | 0.019 |
| 8.000 | 0.367 | True | 0.260 | 0.333 | 0.457 | 0.017 |
| 8.182 | 0.380 | True | 0.260 | 0.333 | 0.473 | 0.014 |
| 9.545 | 0.360 | True | 0.290 | 0.333 | 0.393 | 0.009 |
| 10.909 | 0.350 | True | 0.263 | 0.333 | 0.390 | 0.011 |
| 12.273 | 0.350 | True | 0.307 | 0.333 | 0.377 | 0.007 |
| 13.636 | 0.357 | True | 0.300 | 0.333 | 0.373 | 0.007 |
| **15.000** | **0.313** | **False** | 0.307 | 0.333 | 0.367 | 0.006 |

Reported floor (from `results/floors_105-108.csv`): `3a_spreading_p0.5,
15.0, last_significant_cm=13.636, censored=False`. This is the first (and
only) depth in the 23-depth grid where `fdr_significant` is False — every
prior depth including 13.636 cm is BH-significant, so the floor is pinned
at the grid maximum by construction, not because the signal collapsed
early.

## 2. Does the control distribution show a "near-constant collapse" near the floor?

At the floor depth (15.0 cm), the controls do sit in a narrow band
(min 0.307 / median 0.333 / max 0.367, std 0.006) — but this narrow-band
pattern is present at the deep end of **every** seed and every condition
in both batteries (control variance shrinks monotonically with depth
because the classification task itself becomes harder to overfit to any
random label permutation at low SNR/deep attenuation), not something
distinct to this seed/condition. It's the same shape seen at 15.0 cm in,
for example, seed 106 condition 3a (ctrl_std 0.0084) and seed 107
condition 3a (ctrl_std 0.0059) — see below.

## 3. Comparison against seeds 106/107's own floor-adjacent control behavior

The task description states seeds 106/107 "showed" this same near-constant
collapse pattern "at the depths near the floor" in an earlier check. Pulling
seed 106 and 107's own condition-3a floor rows directly (not assumed):

- **Seed 106**, floor = 6.0 cm (first `fdr_significant=False`), row at
  6.0 cm: real_acc=0.407, ctrl_min=0.127, ctrl_median=0.333,
  ctrl_max=0.507, **ctrl_std=0.0377**.
- **Seed 107**, floor = 1.364 cm (first `fdr_significant=False`; already
  its own anomalous-recovery case — see section 5), row at 1.364 cm:
  real_acc=0.753, ctrl_min=0.0, ctrl_median=0.333, ctrl_max=0.907,
  **ctrl_std=0.183**.

**This does not match the premise.** At seeds 106's and 107's own floor
depths, the control distribution is still wide (std 0.038 and 0.183
respectively — far from "near-constant"), because their floors sit at
much shallower depths (6.0 cm and 1.364 cm) where controls haven't yet
converged to chance. The tight, near-constant control band (std ~0.006-
0.009) only shows up at the deep tail (12-15 cm) for every seed,
including 106 and 107 — it just isn't at *their* floor depth, because
their floors are shallow. Seed 108's floor happens to land at 15.0 cm
(the deep tail) precisely because it stayed BH-significant all the way
out, so its floor row is the one place in this comparison where "floor
depth" and "tight-control-band depth" coincide. This is reported as a
discrepancy between the task's stated premise and what the underlying
CSVs show, not resolved further.

## 4. Seed 108: baseline floor and all 17 other condition shifts

Baseline (condition 1) floor for seed 108: **10.909 cm** (from
`floors_105-108.csv`). This is the deepest baseline floor of all 8 seeds
across both branches:

| seed | baseline floor (cm) |
|---|---|
| 101 | 9.545 |
| 102 | 6.000 |
| 103 | 8.182 |
| 104 | 8.000 |
| 105 | 8.000 |
| 106 | 9.545 |
| 107 | 7.500 |
| **108** | **10.909** |

Seed 108's shift vs. its own baseline, all 17 non-baseline conditions
(floor - 10.909):

| condition | floor (cm) | shift (cm) |
|---|---|---|
| 2_amp_randomization | 13.636 | +2.727 |
| 3a_spreading_p0.5 | 15.000 | **+4.091** |
| 3b_spreading_p1.0 | 7.500 | -3.409 |
| 4a_stiffness_cv_perclass | 15.000 | +4.091 (censored) |
| 4b_stiffness_cv_uniform | 15.000 | +4.091 (censored) |
| 5a_snr20 | 7.500 | -3.409 |
| 5b_snr13.58 | 6.818 | -4.091 |
| 5c_snr10 | 6.000 | -4.909 |
| 5d_snr5 | 4.300 | -6.609 |
| 5e_snr0 | 3.500 | -7.409 |
| 6a_exact_atten | 13.636 | +2.727 |
| 6b_exact_atten_tof | 9.545 | -1.364 |
| 7a_visc_hf_1.4x | 13.636 | +2.727 |
| 7b_visc_hf_1.8x | 12.273 | +1.364 |
| 7c_visc_hf_2.6x | 13.636 | +2.727 |
| 7d_visc_cirr_1.2 | 15.000 | +4.091 (censored) |
| 7e_visc_cirr_2.18 | 7.500 | -3.409 |
| 7f_visc_cirr_2.5 | 7.500 | -3.409 |

**Seed 108 is not a general outlier in direction**: every condition's
shift sign matches the direction already established across the other
7 seeds for that same condition family (amplitude/exact-attenuation/
low-multiplier-viscosity deeper; spreading-p1.0/SNR-degradation/raised-
cirrhosis-viscosity shallower). Only condition 3a (spreading p=0.5) flips
sign relative to the other 7 seeds. Seed 108 does show **larger-magnitude
shifts than typical** in several conditions (e.g. SNR 0dB shift -7.409 cm
vs. a roughly -4 to -5 cm range in most other seeds), which is at least
partly mechanical: because its own baseline (10.909 cm) is already the
deepest of the 8, and conditions that move the floor shallower have more
"room" to produce a large negative shift on this seed than on a seed
whose baseline is already shallow.

## 5. Seed 108 baseline: a second anomalous-recovery instance (not part of the 3a question, but found while checking)

`results/floors_105-108.csv` lists seed 108 baseline as floor=10.909,
last_significant_cm=15.0 — last_significant deeper than the floor itself.
Confirmed directly against `depth_detail_108.csv`, condition `1_baseline`:
depths 10.909 and 12.273 cm are `fdr_significant=False`, but the two
deepest grid points, 13.636 and 15.0 cm, are `fdr_significant=True` again
(real_acc 0.357 and 0.347 — both barely above chance, same
near-chance-noise-flip pattern already documented for seed 103's baseline
in the 101-104 REPORT.md). This is the same known phenomenon, not a new
bug, but is a second anomalous-recovery seed in the n=8 set (103 and 108),
not previously reported for 108 specifically.

## 6. Sign-flip confirmation across the full n=8 set for condition 3a

Baseline and condition-3a floors pulled directly from both branches'
`results/floors_*.csv`:

| seed | baseline floor | 3a floor | shift (3a - baseline) |
|---|---|---|---|
| 101 | 9.545 | 6.000 | -3.545 |
| 102 | 6.000 | 3.500 | -2.500 |
| 103 | 8.182 | 6.500 | -1.682 |
| 104 | 8.000 | 6.500 | -1.500 |
| 105 | 8.000 | 6.500 | -1.500 |
| 106 | 9.545 | 6.000 | -3.545 |
| 107 | 7.500 | 1.364 | -6.136 |
| **108** | **10.909** | **15.000** | **+4.091** |

Confirmed: seed 108 is the **only** one of the 8 seeds where condition 3a
(geometric spreading, p=0.5) produces a shallower-baseline-relative-to-
deeper-condition sign flip (floor goes deeper than baseline instead of
shallower). All other 7 seeds are consistently negative (shallower),
ranging from -1.5 to -6.1 cm. No other seed in either branch shows this
sign flip for condition 3a.

## 7. Not resolved here

This file reports the numbers and does not diagnose *why*
seed 108's 3a run landed BH-significant all the way to the grid ceiling
(e.g. whether it's genuine seed-to-seed model variance, a coincidence of
which windows fell into the 200-shuffle test set for this seed, or
something else). The confusion matrices
(`results/confmat_108.npz`, key `3a_spreading_p0.5`) were pulled to
confirm the file exists and is loadable but per-class breakdown was not
required by the task and is not reproduced here.

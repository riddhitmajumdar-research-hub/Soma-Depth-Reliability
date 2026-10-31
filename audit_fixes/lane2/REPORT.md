# Lane 2 — Repeated-CV significance correction & safety-gate stability

**Scope:** read-only on the rest of the repo; all work in `audit_fixes/lane2/`.
No training was rerun. Every number below comes from
`audit_fixes/lane2/analysis.py`, run once in this pass; full stdout is saved
verbatim in `audit_fixes/lane2/raw_output.txt`, machine-readable results in
`audit_fixes/lane2/results.json`.

## Files used (all read-only)

- `Source/logs/log05_log11_grouped_cv_verification.csv` — single grouped-5-fold split, per-fold baseline/CNN/ResNet1D accuracies (`log11_paired_grouped_cv` rows)
- `Source/logs/log11_grouped_paired_ttests.csv` — original single-split paired-t-test numbers (used only as check values)
- `Source/logs/log11_repeated_grouped_cv_raw.csv` — 15 reps × 5 folds = 75 rows of per-fold accuracies (the repeated-CV check)
- `Source/logs/log11_repeated_grouped_cv_summary.csv` — original repeated-CV summary (sign-flip permutation p-values)
- `Source/logs/degradation_aware_final_results.csv` — 24 certified floors (seeds 101–124), columns `seed, naive_floor_cm, last_significant_depth_cm, anomalous_recovery`
- `Source/constants.py` — imported `MAX_VALID_DEPTH_CM` (6.50 cm); no constant was hardcoded
- `Source/degradation_aware_final_sweep.py` — read only, to recover the exact `anomalous_recovery` definition (lines ~155–231)

## Task 1 — locating per-fold data

All five log files above exist and were used as listed.

**Per-depth p-values / per-seed BH-FDR flags for the 24-seed sweep are NOT
SAVED anywhere in the repo.** `degradation_aware_final_sweep.py::run_one_seed()`
computes a permutation p-value and a BH-FDR flag for every depth internally,
but only writes the three derived columns (`naive_floor_cm`,
`last_significant_depth_cm`, `anomalous_recovery`) to CSV — the per-depth
intermediates are discarded in memory once each seed finishes. This directly
determines the answer to Task 6 (below): that table cannot be reconstructed
without rerunning the 24-seed training sweep, which this lane is not
permitted to do.

## Task 2 — single-split paired t-tests (n=5 folds)

Recovered fold accuracies:

| fold | baseline | CNN | ResNet1D |
|---|---|---|---|
| 1 | 0.7083 | 0.9444 | 0.9583 |
| 2 | 0.6806 | 0.9028 | 0.9028 |
| 3 | 0.8056 | 0.8056 | 0.8472 |
| 4 | 0.7042 | 0.8889 | 0.8889 |
| 5 | 0.7500 | 0.9444 | 0.9444 |

| comparison | mean diff | t | p (computed) | p (record) | p check | d (computed, paired-diff SD, ddof=1) | d (record) | d check |
|---|---|---|---|---|---|---|---|---|
| CNN vs baseline | 0.1675 | 3.906 | 0.0175 | 0.0175 | **MATCH** | 1.747 | 1.54 | **MISMATCH** |
| ResNet1D vs baseline | 0.1786 | 4.948 | 0.0078 | 0.0078 | **MATCH** | 2.213 | 2.76 | **MISMATCH** |
| ResNet1D vs CNN | 0.0111 | 1.373 | 0.2418 | 0.2420 | MATCH | 0.614 | 0.59 | MATCH |

**p-values all reproduce exactly.** Two of three Cohen's d values do not.
Per rule 6, this is reported and NOT resolved. For transparency, two other
common d conventions were also checked (not to pick a "right" answer, just to
show the discrepancy isn't explained by an obvious alternate convention):

| comparison | d (ddof=0 paired) | d (pooled-SD, independent-samples formula) |
|---|---|---|
| CNN vs baseline | 1.953 | 3.148 |
| ResNet1D vs baseline | 2.474 | 3.804 |
| ResNet1D vs CNN | 0.686 | 0.217 |

None of these alternate conventions reproduce 1.54 / 2.76 either. **The
source of the record's stated d values for these two comparisons is not
identified — flagging as an open question, not resolving.**

Repeated-CV p=0.0002 check: the record's stated repeated-CV p=0.0002 for all
three comparisons **matches** `log11_repeated_grouped_cv_summary.csv`'s
`permutation_p_value` (0.0002 for all three, rounds from ≈0.00019996). This is
a **sign-flip permutation p-value**, not a Nadeau-Bengio corrected t-test —
see Task 3.

## Task 3 — Nadeau-Bengio corrected resampled t-test

`t = mean / sqrt((1/J + n_test/n_train) * s²)`, `n_test/n_train = 1/4`, `df = J−1`.

**75 differences (repeated 15×5-fold CV):**

| comparison | J | df | mean diff | t | p | 95% CI |
|---|---|---|---|---|---|---|
| CNN vs baseline | 75 | 74 | 0.1434 | 3.005 | 0.003624 | [0.0483, 0.2384] |
| ResNet1D vs baseline | 75 | 74 | 0.1917 | 5.385 | 0.000001 | [0.1208, 0.2626] |
| ResNet1D vs CNN | 75 | 74 | 0.0483 | 1.155 | 0.251959 | [-0.0351, 0.1317] |

**5 differences (single split):**

| comparison | J | df | mean diff | t | p | 95% CI |
|---|---|---|---|---|---|---|
| CNN vs baseline | 5 | 4 | 0.1675 | 2.604 | 0.059805 | [-0.0111, 0.3461] |
| ResNet1D vs baseline | 5 | 4 | 0.1786 | 3.298 | 0.029976 | [0.0283, 0.3289] |
| ResNet1D vs CNN | 5 | 4 | 0.0111 | 0.915 | 0.411893 | [-0.0226, 0.0448] |

**Finding, not a record-number contradiction (new analysis, no prior claim to
check against):** once the Nadeau-Bengio variance-inflation correction is
applied to the *single-split* n=5 comparison, **CNN vs baseline is no longer
significant at α=0.05** (p=0.0598, CI crosses 0), versus the naive paired-t
p=0.0175 currently reported. ResNet1D vs baseline remains significant
(p=0.030) but far more marginal than the naive p=0.0078. The repeated-CV
(J=75) result stays clearly significant for both baseline comparisons even
under the conservative correction; ResNet1D vs CNN is not significant either
way. This is a legitimate statistical-rigor finding for the audit response,
not a record mismatch — the record never claimed a Nadeau-Bengio number.

## Task 4 — 24 certified floors (seeds 101–124)

| statistic | computed | record | check |
|---|---|---|---|
| mean | 8.4754 | 8.48 | MATCH |
| SD (ddof=1) | 1.8122 | 1.81 | MATCH |
| median | 8.0000 | 8.00 | MATCH |
| 10th pct (numpy linear) | 6.5000 | 6.50 | MATCH |
| 5th pct (numpy linear) | 5.6500 | 5.65 | MATCH |

**(a) Bootstrap 95% CI, 10,000 resamples of the 24 individual floors:**
- 5th percentile: point 5.65, 95% CI [5.4545, 7.5000]
- 10th percentile: point 6.50, 95% CI [5.4682, 7.5000]

These CIs are wide relative to the point estimates — expected at n=24 — and
both include values well above the adopted 6.50cm gate, i.e. under resampling
uncertainty the 10th-percentile point estimate could plausibly have landed
anywhere from ~5.47 to 7.5cm.

**(b) Leave-one-out range of the 10th percentile (n=23, 24 omissions):**
range = [6.5000, 6.6000] — i.e. dropping any single seed moves the 10th
percentile by at most 0.10cm. The statistic is not being driven by one
seed.

**(c) Seeds with floor < MAX_VALID_DEPTH_CM (6.50cm, imported from
`Source/constants.py`):** 2/24 = 0.0833. Clopper-Pearson 95% CI on this
fraction: [0.0103, 0.2700]. (Note this CI is wide enough that the true
below-gate rate could plausibly be as high as ~27% at this sample size —
flagging as context for the gate's stated safety margin, not a mismatch.)

**(d) Percentile interpolation conventions:**

| percentile | numpy-linear | Weibull (n+1)p | nearest-rank |
|---|---|---|---|
| 10th | 6.5000 | 6.0000 | 6.5000 |
| 5th | 5.6500 | 5.4659 | 5.5000 |

The record's 6.50cm gate value is produced by **numpy-linear and
nearest-rank** conventions, not by the Weibull `(n+1)p` convention (which
gives 6.00cm instead). This should be stated explicitly wherever the gate
derivation is documented, since "10th percentile" is ambiguous without
naming the interpolation rule.

## Task 5 — `anomalous_recovery` flag

**Definition** (from `Source/degradation_aware_final_sweep.py`,
`run_one_seed()`, ~lines 155–231): for a given seed, a permutation p-value is
computed at each evaluated depth as `(#label-shuffled control CNNs with
accuracy ≥ the real model's accuracy, + 1) / 201`; Benjamini-Hochberg FDR
(α=0.05) is applied across that seed's per-depth p-values to flag which
depths are significant; the seed's "naive floor" is the first (shallowest→
deepest) depth that is *not* flagged significant; and `anomalous_recovery`
is `True` if any depth *strictly deeper* than that naive floor is still
flagged significant — i.e., the model appears to regain FDR-significant,
above-chance accuracy at a depth beyond where it first became
indistinguishable from chance.

**Count:** 10/24 seeds have `anomalous_recovery=True` — **MATCH** with the
record's stated 10/24. Seeds: 103, 105, 106, 108, 109, 115, 116, 117, 123, 124.

## Task 6 — per-depth per-seed BH-FDR significance table

**Not possible from existing files.** As established in Task 1, the
per-depth p-values and per-depth significance flags computed inside
`run_one_seed()` are never written to disk — only the three summary columns
per seed are saved. Producing this table would require rerunning the 24-seed
training sweep, which is out of scope for this lane (read-only, no
retraining). No fabricated or approximated numbers are reported here.

## Summary of MATCH / MISMATCH

| item | result |
|---|---|
| Single-split p-values (all 3 comparisons) | MATCH |
| Single-split Cohen's d, CNN vs baseline (1.54) | **MISMATCH** (computed 1.747, paired-diff SD ddof=1) |
| Single-split Cohen's d, ResNet1D vs baseline (2.76) | **MISMATCH** (computed 2.213) |
| Single-split Cohen's d, ResNet1D vs CNN (0.59) | MATCH (0.614) |
| Repeated-CV p=0.0002 (all 3) | MATCH (as a sign-flip permutation p; see caveat above) |
| 24-seed floor mean/SD/median/p10/p5 | MATCH (all 5) |
| anomalous_recovery 10/24 | MATCH |
| 10th-pct interpolation producing 6.50 | numpy-linear / nearest-rank (not Weibull) |

## Open questions / flagged for the researcher (not resolved here)

1. **Cohen's d discrepancy** (CNN vs baseline, ResNet1D vs baseline): neither
   the standard paired-diff formula (ddof=1 or ddof=0) nor an
   independent-samples pooled-SD formula reproduces the record's 1.54 / 2.76.
   Source of those two numbers is unidentified.
2. **Nadeau-Bengio vs naive t-test divergence at n=5**: CNN vs baseline drops
   below α=0.05 significance (p=0.060) once the repeated-CV-style variance
   correction is applied to the single split, even though it is the correct
   test to use whenever training/test sets share data across "independent"
   splits (not applicable in the n=5 single-split case in the strict sense,
   but requested by the task as a sensitivity check — the corrected 5-diff
   number is *not* the primary claim, the 75-diff repeated-CV result is, and
   that one stays significant for both baseline comparisons).
3. **Clopper-Pearson CI on below-gate fraction is wide** ([0.010, 0.270] on a
   point estimate of 0.083) — worth noting anywhere the 6.50cm gate's safety
   margin is characterized as tight, given n=24.

## File list

- `audit_fixes/lane2/analysis.py` — the script that produced every number above
- `audit_fixes/lane2/raw_output.txt` — full stdout from running it
- `audit_fixes/lane2/results.json` — same results, machine-readable
- `audit_fixes/lane2/REPORT.md` — this file

# Lane 7 Pre-Registration v2 — censoring, seed-pairing, seed-110, sign-flip test

Follow-up to `PREREG.md`/`REPORT.md`. Written before any v2 result was
inspected. Covers the 6 items in the v2 follow-up request.

## Status check (unchanged blocker, re-confirmed before writing this plan)

`ps -p 7429` still shows the ResNet1D sweep running; the CSV still has 20
rows (seeds 101–120); the log tail is still on "Seed 121 starting". Item 6
below is therefore deferred again — everything else in this document
proceeds on the same PROVISIONAL n=20 (ResNet1D) / n=24 (CNN) basis as v1,
labeled accordingly.

## 1. Censoring

**Plan:** grep both sweep scripts' floor-assignment logic verbatim (not
paraphrased), quote it in the report. A seed is **right-censored** if its
floor equals `EVAL_DEPTHS_CM[-1]` (15.0 cm) via the `if floor_depth is
None: floor_depth = EVAL_DEPTHS_CM[-1]` fallback — meaning every depth on
the grid was still FDR-significant, so the true floor is unknown and lies
at or beyond 15.0 cm, not "exactly 15.0 cm" as a real observation.

Count censored seeds per model from the CSVs already loaded (`naive_floor_cm
== 15.0` with `EVAL_DEPTHS_CM[-1] == 15.0`, confirmed via the imported
constant, not hardcoded).

If any seed is censored in either group: re-run the independent-sample
comparison as a **Gehan-Wilcoxon test** (via `lifelines.statistics.
logrank_test` with `weightings="wilcoxon"`, which implements exactly this)
with the standard right-censoring convention (`event_observed=0` for
censored seeds, `1` otherwise), and separately a plain **log-rank test**
(same call, `weightings=None`/default). Both reported alongside, not
instead of, the uncensored Mann-Whitney/HL numbers from v1 — explicitly
labeled which is which. If `lifelines` is not installed, state that and
fall back to a hand-rolled Gehan generalized-Wilcoxon statistic (Mann/
Whitney-style score comparison under right-censoring, Gehan 1965 scoring
rule) implemented directly in the script, with the formula shown in-script.

## 2. Seed pairing

**Plan:** grep both scripts' `run_one_seed` seeding lines verbatim.
Empirically test data-identity for seed 101 (the smallest, cheapest check)
by replaying — not retraining — each script's exact pre-training call
sequence (seed the RNGs the way each script does, call the identical
imported `build_degradation_aware_dataset()`, then the identical
`N_SHUFFLES` label-permutation loop, then one `generate_degraded_windows`
eval-cell call) and diffing the resulting arrays with `np.array_equal`.
Training itself (which touches only torch's RNG, independent of numpy's
RNG used for data generation) is not replayed — retraining 40+ models to
check a data-equality question is out of scope and unnecessary once the
call-sequence identity is established.

If arrays are NOT bit-identical: the paired Wilcoxon/HL results in
`REPORT.md` §4d are to be re-labeled "index-matched only (same seed number,
data not verified identical)" rather than "paired on identical inputs."

## 3. Seed 110

Re-state (from files already read in the v1 follow-up, re-quoted here for
this report's self-containedness): full CSV row, matching log line,
`anomalous_recovery`, `real_model_collapse_count`,
`real_model_escaped_on_attempt`, `control_test_mean_overall`,
`control_sanity_ok`. If `Source/seed_weights_resnet/
degradation_aware_resnet_seed110.pt` exists, reuse the already-produced
`eval_seed110.py` per-depth accuracy table (already run and saved to
`seed110_eval_raw.txt` in the prior follow-up — re-cited, not re-run,
since nothing about seed 110's saved weights or the standard grid has
changed). CNN seed-110 weights: already confirmed absent by repo-wide
`find`; re-confirmed here, not re-searched from scratch, since no new CNN
weights could have appeared without someone running that sweep, which
nothing in this pass did.

## 4. Paired sign-flip permutation test

**Plan:** on the same 20 matched-seed differences (ResNet1D − CNN) used for
the v1 Wilcoxon test, run a paired sign-flip (permutation) test: for each of
100,000 iterations, independently flip the sign of each difference with
probability 0.5 (`rng.choice([1, -1], size=n)`), compute the mean of the
flipped differences, and compare the distribution of these to the observed
mean difference — two-sided p-value = fraction of `|flipped_mean| >=
|observed_mean|`. Zero differences are *kept* in this test (unlike
Wilcoxon's `zero_method='wilcox'`, which drops them) since sign-flipping a
zero is a no-op and doesn't bias anything — stated explicitly since this is
a deliberate difference from the Wilcoxon test's handling. Same
`np.random.default_rng(seed=20260101)` reproducibility seed as v1's
bootstrap, for consistency across this lane's outputs.

## 5. Seeds below 6.50 cm

List (not just count) every seed in each model's floor set with
`naive_floor_cm < 6.50`, using the same imported `MAX_VALID_DEPTH_CM`
constant as v1.

## 6. Full 24-seed re-run trigger

Deferred — confirmed blocked in the status check above. No action beyond
noting the deferral.

## Disagreement policy

Unchanged from v1: any number here that conflicts with an existing record
number is listed side by side in REPORT_v2.md, not resolved.

## Outputs

- `run_lane7_v2_analysis.py` — script for items 1, 4, 5 (censoring +
  sign-flip + below-gate lists), reusing the same CSV-loading logic as v1.
- `verify_seed_pairing.py` — already written for item 2 (pre-dates this
  PREREG_v2 only in file-creation order, not in being run — it has not
  been executed yet; run now under this plan).
- `eval_seed110.py` / `seed110_eval_raw.txt` — already produced in the
  prior follow-up for item 3; cited, re-verified present, not re-run.
- `lane7_v2_output_raw.txt` — raw stdout of `run_lane7_v2_analysis.py`.
- `seed_pairing_raw.txt` — raw stdout of `verify_seed_pairing.py`.
- `REPORT_v2.md` — narrative summary, exact numbers only from the above
  raw-output files.

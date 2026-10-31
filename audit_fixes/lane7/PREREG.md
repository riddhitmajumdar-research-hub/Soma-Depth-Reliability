# Lane 7 Pre-Registration — ResNet1D vs CNN 24-seed floor comparison

Written before any result was inspected. Locks the analysis plan (tests, CIs,
outputs) so nothing downstream is chosen after seeing the numbers.

## Status check performed before writing this plan (data availability)

- CNN input: `Source/logs/degradation_aware_final_results.csv` — 24 rows,
  seeds 101–124. Complete.
- ResNet1D input: `Source/logs/degradation_aware_resnet_final_results.csv` —
  **20 rows, seeds 101–120 only**, as of this pass. The generating
  process, `degradation_aware_resnet_final_sweep.py` (pid 7429), is still
  running (confirmed via `ps`) and was mid-way through seed 121 in its log
  tail at inspection time.
- README.md line 1004 already flags this open item: *"ResNet1D 24-seed
  retest only thread still open."* This is a known incomplete task, not a
  new contradiction — but it means the analysis below **cannot be run at
  n=24 for ResNet1D in this pass**.

## Decision (fixed before looking at any statistic)

Run the full pre-registered plan below twice in the same script:
1. On the 20 ResNet1D seeds currently available (101–120), matched against
   the same 20 CNN seeds, labeled **PROVISIONAL — n=20, sweep incomplete**.
2. A second block that will simply report "NOT AVAILABLE — sweep in
   progress" for anything requiring all 24 ResNet1D seeds (i.e. the true
   24-seed comparison this lane is meant to close).

No number from block 1 is to be described as the certified 24-seed result
anywhere in the report. This is stated here, before results exist, to
prevent silent scope-narrowing later.

## Inputs (paths, to be confirmed unchanged by the script itself)

- CNN: `Source/logs/degradation_aware_final_results.csv`
  columns: `seed, naive_floor_cm, last_significant_depth_cm, anomalous_recovery`
- ResNet1D: `Source/logs/degradation_aware_resnet_final_results.csv`
  columns: `seed, naive_floor_cm, last_significant_depth_cm, anomalous_recovery,
  real_model_collapse_count, real_model_escaped_on_attempt,
  control_test_mean_overall, control_sanity_ok`
- Gate threshold: `MAX_VALID_DEPTH_CM = 6.50` from `Source/constants.py`
  (imported, not hardcoded).

The floor variable analyzed throughout is `naive_floor_cm` for each model,
matching the record's stated floor definition (first failure point).

## Tests and statistics, fixed in advance

1. **Descriptive stats per model floor set**: mean, SD (sample, ddof=1 —
   matching the record's correction to ddof=1), median, 5th
   percentile, 10th percentile, range (min, max). Percentiles computed
   under three interpolation methods (numpy `linear`, `weibull`,
   `nearest-rank` via `closest_observation` equivalent) since the record's
   gate derivation (constants.py line 85) depends on percentile method
   choice — report all three, do not pick one silently.

2. **Mann-Whitney U** (independent-samples, CNN floors vs ResNet1D floors):
   `scipy.stats.mannwhitneyu(..., alternative='two-sided', method='auto')`,
   which uses exact computation when feasible and falls back to a normal
   approximation with continuity correction under ties otherwise; report
   which method scipy actually used. Also report a manual permutation-test
   p-value (all `C(n1+n2, n1)` label permutations if computationally
   feasible at this n, else 100,000-permutation Monte Carlo) as a
   cross-check. **MATCH/MISMATCH** stated explicitly between exact and
   permutation p.

3. **Hodges-Lehmann shift estimator** (median of all pairwise differences
   between the two independent samples) with a 10,000-resample bootstrap
   95% CI (percentile method, resampling within each group with
   replacement, fixed `np.random.default_rng(seed=20260101)` for
   reproducibility — seed value stated here, not chosen after seeing
   results).

4. **Paired Wilcoxon signed-rank test** on per-seed differences
   (ResNet1D − CNN, seeds 101–120 for the provisional block — the paired
   design requires seeds present in both files; will explicitly report if
   any seed differs) via `scipy.stats.wilcoxon(..., zero_method='wilcox',
   correction=False, mode='auto')`. Also report the paired Hodges-Lehmann
   (median of paired differences) with the same 10,000-resample bootstrap
   95% CI (paired resampling — resample seed-difference pairs together).

5. **Fraction of seeds with floor < 6.50 cm**, per model: count, n,
   fraction, and exact Clopper-Pearson 95% CI (`scipy.stats.beta.ppf`
   construction, standard exact binomial CI formula).

6. **`anomalous_recovery` counts**: raw count and fraction True, per model,
   over whatever n is available for that model.

7. **ResNet1D restart data**: per-seed `real_model_collapse_count` and
   `real_model_escaped_on_attempt`, printed as a table; overall restart
   rate = fraction of seeds with `real_model_collapse_count > 0` over
   available n, with Clopper-Pearson 95% CI. (CNN file has no equivalent
   columns — will state N/A for CNN, not fabricate zero.)

8. Any seed where ResNet1D floor exactly equals CNN floor: stated plainly
   by seed number, both values.

## Disagreement policy

Any number produced here that conflicts with a number already stated in
README.md (e.g. 8.48cm ± 1.81cm certified mean, 6.50cm gate, restart/escape
counts, anomalous-recovery counts) will be listed under "Flagged
disagreements" in REPORT.md with both values shown side by side. Per the
protocol rules, these are **not** resolved or adjudicated here.

## Outputs

- `run_lane7_analysis.py` — the single script producing every number below.
- `lane7_output_raw.txt` — full stdout of that script, unedited.
- `REPORT.md` — narrative summary, exact numbers copied from
  `lane7_output_raw.txt` only, file list, disagreements, open questions.

No number is to appear in REPORT.md that is not traceable to
`lane7_output_raw.txt`.

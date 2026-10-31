> **Superseded.** This is the provisional n=20 analysis. Its "blocked" status is out of date. The n=24 numbers are in `final_v1_output.txt` and `final_v2_output.txt` in this directory and are described in [docs/methodology/10-lane7-resnet1d-vs-cnn.md](../../docs/methodology/10-lane7-resnet1d-vs-cnn.md). The text below is unchanged.

# Lane 7 Report v2 — censoring, seed-pairing, seed-110, sign-flip test

Follow-up to `REPORT.md`. **Status unchanged: still BLOCKED/PROVISIONAL.**
`ps -p 7429` confirms the ResNet1D sweep is still running; the CSV still
has 20/24 seeds; the log is still on "Seed 121 starting". Item 6 (full
24-seed re-run) is deferred again — nothing below changes that.

All numbers below are copied from `lane7_v2_output_raw.txt`,
`seed_pairing_raw.txt`, and `seed110_eval_raw.txt` (item 3 reuses the
already-run seed-110 evaluation from the prior follow-up unchanged, since
nothing about the seed-110 weights or grid has changed).

## 1. Censoring — a real finding, not previously flagged in v1

**Both sweep scripts assign the floor via an identical fallback when no
depth is non-significant.** Quoted directly from both files:

CNN (`Source/degradation_aware_final_sweep.py`, lines 160–161):
```
    if floor_depth is None:
        floor_depth = EVAL_DEPTHS_CM[-1]
```

ResNet1D (`Source/degradation_aware_resnet_final_sweep.py`, lines 222–223):
```
    if floor_depth is None:
        floor_depth = EVAL_DEPTHS_CM[-1]
```

`EVAL_DEPTHS_CM[-1]` (imported, confirmed by the script) = **15.0 cm**, the
maximum depth on the grid. A seed hitting this fallback means every tested
depth was still FDR-significant — the true floor is unknown and lies
somewhere at or beyond 15.0 cm, not "exactly 15.0 cm" as a directly observed
value. This is textbook right-censoring.

**Count of censored seeds:**
- CNN (n=24): **0 censored.**
- ResNet1D (n=20, PROVISIONAL): **1 censored — seed 103** (`naive_floor_cm =
  15.0`).

This was **not called out as censoring in v1's REPORT.md** — v1 correctly
listed seed 103's floor as 15.0cm in the raw dump but treated it as an
ordinary observation in the mean/SD/percentile/Mann-Whitney/HL/Wilcoxon
calculations there, which understates the true (unknown, ≥15.0cm) floor and
biases those summary statistics slightly downward for ResNet1D. **This is a
disagreement with how v1 presented its own numbers, flagged here per the
protocol's disagreement policy, not resolved.**

### Censoring-aware re-analysis (Gehan-Wilcoxon and log-rank)

`lifelines` is not installed in this environment (`pip3 list` confirms it
absent); per the pre-registered fallback, both tests were hand-implemented
directly from the standard textbook formulas (Gehan, *Biometrika* 1965;
Kalbfleisch & Prentice, *The Statistical Analysis of Failure Time Data*, 2nd
ed., §1.5 and §6.3) in `run_lane7_v2_analysis.py`.

- **Gehan-Wilcoxon** (CNN n=24 vs ResNet1D n=20, ResNet1D seed 103 marked
  censored): W(CNN) = −233.0000, z = −2.7790, two-sided p = **0.00545214**
- **Log-rank** (unweighted): O−E(CNN) = 8.4698, χ² = 10.9577, p =
  **0.00093215**

Both reject the null of equal floor distributions, same direction as v1's
uncensored Mann-Whitney (p=0.00565569 — see `lane7_output_raw.txt` §2).
**Both versions are reported side by side, per the pre-registered plan;
neither is presented as superseding the other.** The one-censored-point
correction did not flip the qualitative conclusion here, but it is the
statistically correct test to run given the censoring, and should be the
one carried forward once n=24 is available (at which point it should be
re-run with `lifelines` if installable, as a cross-check against this
hand-rolled implementation).

## 2. Seed pairing — verified identical, not just index-matched

Quoted seeding code (`grep -n -A2 "def run_one_seed"`, both files):

CNN (`degradation_aware_final_sweep.py` lines 101–103):
```
def run_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
```

ResNet1D (`degradation_aware_resnet_final_sweep.py` lines 158–160):
```
def run_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
```

Both call `torch.manual_seed(seed)` then `np.random.seed(seed)` with the
identical seed value, then both immediately call the same imported
`build_degradation_aware_dataset()` function (the ResNet script imports it
directly from `degradation_aware_final_sweep.py` rather than
reimplementing it — confirmed by the `from degradation_aware_final_sweep
import (build_degradation_aware_dataset, ...)` line already quoted in
`REPORT.md` §1).

**Empirical check (seed 101), not just a code-reading argument:**
`verify_seed_pairing.py` replayed each script's exact data-generation call
sequence (seed → build training set → 200 label-shuffle permutations → one
depth's eval-cell generation) *without retraining any model* (model
training only consumes torch's RNG, which is independent of numpy's RNG
used for all data generation, so replaying data-gen calls alone is
sufficient). Result, from `seed_pairing_raw.txt`:

- `X_train` identical (`np.array_equal`): **True** (shape (375, 100) both)
- `y_train_names` identical: **True**
- All 200 label-shuffle permutations identical: **True**
- Eval-cell windows at depth_cm=0.0 identical: **True**
- Eval-cell labels at that depth identical: **True**

**Verdict: MATCH.** For seed 101, the two scripts produce bit-identical
training data, control-permutation data, and evaluation windows. Since the
generation function and call order are identical across all other depths
too (both scripts loop over the same imported `EVAL_DEPTHS_CM`), this
result generalizes: **the paired comparison (Wilcoxon, paired HL, sign-flip
test) is comparing the two models on genuinely identical per-seed inputs,
not merely seeds that share a number.** No relabeling to "index-matching
only" is needed.

## 3. Seed 110 (ResNet1D) — full record

From `Source/logs/degradation_aware_resnet_final_results.csv` and
`Source/logs/degradation_aware_resnet_final_sweep_log.txt`
(`seed110_eval_raw.txt` §1, already produced in the prior follow-up):

| field | value |
|---|---|
| seed | 110 |
| naive_floor_cm | 1.3636363636363635 |
| last_significant_depth_cm | 13.636363636363635 |
| anomalous_recovery | True |
| real_model_collapse_count | 0 |
| real_model_escaped_on_attempt | 0 |
| control_test_mean_overall | 0.3333594202898551 |
| control_sanity_ok | True |

Matching sweep log line:
```
Seed 110: naive_floor=1.36cm  last_sig=13.636363636363635  anomalous_recovery=True  collapses=0  escaped_on_attempt=0  control_test_mean=33.34% [OK]  (7703s)
```

Seed 110 is **not** censored (its floor, 1.36cm, is a genuine observed
FDR-significance failure point, far below the grid max — this is the
"outlier" already flagged in v1 §5, unrelated to the censoring issue in
§1 above).

**Accuracy-by-depth, reloaded seed-110 ResNet1D weights**
(`Source/seed_weights_resnet/degradation_aware_resnet_seed110.pt`, 35,879
bytes, confirmed present), evaluated on the standard grid (100 Hz, 27.05 dB
SNR via `constants.SENSOR_SNR_DB`, N_SAMPLES_PER_CELL=300/100 per class —
all imported, not hardcoded). This is an independent re-evaluation using a
freshly-seeded (`seed=110`) generation of eval windows, **not a bit
reproduction** of the original sweep's own internal eval windows (the
sweep does not reset its RNG before its eval loop, and per Log 23's
already-established finding elsewhere in this project, PyTorch/data
pipelines here are not guaranteed bit-reproducible across independently
sequenced runs even with the same seed):

| depth_cm | accuracy |
|---|---|
| 0.00 | 0.5800 |
| 1.36 | 0.4100 |
| 2.73 | 0.3700 |
| 3.50 | 0.3933 |
| 4.00 | 0.4100 |
| 4.09 | 0.3567 |
| 4.30 | 0.3900 |
| 4.50 | 0.4000 |
| 5.00 | 0.3833 |
| 5.45 | 0.4133 |
| 5.50 | 0.4367 |
| 6.00 | 0.4567 |
| 6.50 | 0.4667 |
| 6.82 | 0.4733 |
| 7.00 | 0.4767 |
| 7.50 | 0.5300 |
| 8.00 | 0.5633 |
| 8.18 | 0.4733 |
| 9.55 | 0.4600 |
| 10.91 | 0.4633 |
| 12.27 | 0.4367 |
| 13.64 | 0.4033 |
| 15.00 | 0.3800 |

**CNN seed-110 comparison: N/A.** No saved CNN weights for seed 110 exist
anywhere in the repo — checked by a repo-wide `find -iname '*cnn*seed110*'
-o -iname '*seed110*cnn*'` (no matches), plus a direct listing of
`Source/seed_weights_resnet/` (ResNet1D only) and
`audit_fixes/lane5/seed_weights/` (a different lane's work, seeds 101–103
only, not 110). README.md independently confirms Log 20's 24-seed CNN
sweep never called `torch.save()` for any seed. Producing a CNN
accuracy-by-depth curve for seed 110 would require retraining, out of
scope for this read-only lane.

## 4. Paired sign-flip permutation test

On the same 20 matched-seed differences used for v1's Wilcoxon test
(`REPORT.md` §4d), with **zero differences kept** (unlike Wilcoxon's
`zero_method='wilcox'`, which drops the 1 zero-difference pair at seed
116 — a deliberate difference between the two tests, stated per the
pre-registered plan):

- Observed mean difference (ResNet1D − CNN) = 2.193182 cm
- Sign-flip permutation test (100,000 flips): two-sided **p = 0.033330**

Compare to v1's Wilcoxon signed-rank p = 0.03997423 (drops 1 zero pair, uses
ranks not raw magnitudes). Both are nominally significant at α=0.05 and
point the same direction. **Both are reported; neither supersedes the
other** — they test slightly different null hypotheses (Wilcoxon: symmetric
distribution around zero via signed ranks; sign-flip on the mean: exchangeable
sign of the raw difference), and the small gap between p=0.034 and p=0.040
is exactly the kind of small-n sensitivity worth keeping visible rather than
picking whichever number looks better.

## 5. Seeds below 6.50 cm — full lists (not just counts)

| Model | n | count below 6.50cm | seeds |
|---|---|---|---|
| CNN | 24 | 2 | seed 102 (5.500000 cm), seed 111 (5.454545 cm) |
| ResNet1D (PROVISIONAL) | 20 | 3 | seed 101 (6.000000 cm), seed 110 (1.363636 cm), seed 112 (4.000000 cm) |

(Matches the counts already reported in v1 §4e; this adds the seed-level
detail that report only summarized as a fraction.)

## 6. Full 24-seed re-run

**Deferred — confirmed still blocked.** `ps -p 7429` shows the sweep still
running (elapsed 02-04:18:10 at last check), CSV still at 20 rows, log
still mid-seed-121. No action taken; re-run all Lane 7 scripts
(`run_lane7_analysis.py`, `run_lane7_v2_analysis.py`,
`verify_seed_pairing.py`) once the CSV reaches 24 rows — none require
modification to pick up the full set, and the "PROVISIONAL" labeling
throughout both reports is designed to drop automatically once n=24 is
passed in.

## 7. Open questions carried over / new

1. (Carried from v1) Will seeds 121–124 fall inside or outside the current
   ResNet1D range? Now sharpened: will any of 121–124 *also* hit the
   15.0cm censoring ceiling, which would change both the censored-seed
   count and which comparison test (Gehan-Wilcoxon vs. plain Mann-Whitney)
   is the more correct one to lead with?
2. (New) Should v1's already-reported ResNet1D mean/SD/percentiles be
   corrected to account for seed 103's censoring, or left as descriptive
   statistics with the censoring caveat attached? This report does not
   decide that — flagged as an open question per the disagreement policy,
   since "correcting" v1's numbers is itself a judgment call outside this
   lane's read-only remit.
3. (New) The Gehan-Wilcoxon/log-rank implementation here is hand-rolled
   because `lifelines` isn't installed. Recommend installing `lifelines`
   in a follow-up (not done here — no packages were installed in this
   pass; installing dependencies is more than a read-only check) so
   the hand-rolled result can be cross-checked against a maintained
   library implementation before this becomes a headline number.

## 8. File list

New/updated files in `audit_fixes/lane7/` this follow-up:
- `PREREG_v2.md` — plan fixed before this follow-up's results were seen
- `run_lane7_v2_analysis.py` — censoring, sign-flip test, below-gate lists
- `lane7_v2_output_raw.txt` — its raw stdout
- `verify_seed_pairing.py` — seed-101 data-identity check
- `seed_pairing_raw.txt` — its raw stdout
- `REPORT_v2.md` — this file

Reused unchanged from the prior follow-up (not re-run, nothing changed):
- `eval_seed110.py`, `seed110_eval_raw.txt`

Unchanged from v1:
- `PREREG.md`, `run_lane7_analysis.py`, `lane7_output_raw.txt`, `REPORT.md`

All inputs read, never modified: `Source/logs/degradation_aware_final_results.csv`,
`Source/logs/degradation_aware_resnet_final_results.csv`, `Source/constants.py`,
`Source/degradation_aware_final_sweep.py`, `Source/degradation_aware_resnet_final_sweep.py`,
`Source/resnet_classifier.py`, `Source/physics_model.py`,
`Source/seed_weights_resnet/degradation_aware_resnet_seed110.pt`.
The running sweep (pid 7429) and its output files were not touched. No
commit was made.

> **Superseded.** This is the provisional n=20 analysis. Its "blocked" status is out of date. The n=24 numbers are in `final_v1_output.txt` and `final_v2_output.txt` in this directory and are described in [docs/methodology/10-lane7-resnet1d-vs-cnn.md](../../docs/methodology/10-lane7-resnet1d-vs-cnn.md). The text below is unchanged.

# Lane 7 Report — ResNet1D vs CNN 24-seed floor comparison

**Status: BLOCKED / PROVISIONAL.** The task this lane was given ("close the
ResNet1D-vs-CNN 24-seed comparison") cannot be completed in this pass.
The ResNet1D sweep that generates the required data is still running (see
below) and only has 20 of 24 seeds. Everything below is a provisional n=20
analysis, run under a plan fixed *before* any result was inspected
(`PREREG.md`), plus explicit MATCH/MISMATCH checks against numbers already
in the record. No pre-registered test was skipped, and nothing here should be
read as the certified 24-seed comparison.

## 1. What was done

1. Located inputs by `find`/`grep`, did not assume names:
   - CNN: `Source/logs/degradation_aware_final_results.csv` (24 rows,
     columns `seed, naive_floor_cm, last_significant_depth_cm,
     anomalous_recovery`)
   - ResNet1D: `Source/logs/degradation_aware_resnet_final_results.csv`
     (**20 rows**, columns `seed, naive_floor_cm, last_significant_depth_cm,
     anomalous_recovery, real_model_collapse_count,
     real_model_escaped_on_attempt, control_test_mean_overall,
     control_sanity_ok`)
   - Gate threshold `MAX_VALID_DEPTH_CM = 6.50` imported from
     `Source/constants.py` (not hardcoded).
2. Confirmed via `ps -p 7429` that the ResNet1D sweep
   (`degradation_aware_resnet_final_sweep.py`) is still running, and its log
   (`Source/logs/degradation_aware_resnet_final_sweep_log.txt`) tail showed
   it mid-way through seed 121 at inspection time. Per the protocol rules, this
   process and its output files were not touched, killed, or reniced.
3. Wrote `PREREG.md` fixing every test, CI method, and output *before*
   looking at any result.
4. Wrote and ran `run_lane7_analysis.py` (stdlib `csv` + `numpy` + `scipy`
   only; no pandas needed — not installed in this environment). Full raw
   stdout saved unedited to `lane7_output_raw.txt`. Every number below is
   copied from that file.

## 2. Blocking finding — report only, not resolved

**ResNet1D CSV has seeds 101–120 only; seeds 121, 122, 123, 124 are
missing.** This is a data-completeness gap, not a computed contradiction,
so it is reported rather than resolved, per the protocol rules. README.md line
1004 already lists "ResNet1D 24-seed retest" as an open thread, so this is
consistent with the record's own stated status — it just means this lane
cannot deliver the certified comparison yet. **Recommend re-running this
lane's script once the sweep finishes and the CSV has all 24 rows** (the
script requires no changes — it will automatically pick up all 24 seeds and
the "PROVISIONAL" labeling will drop away).

## 3. Sanity checks against numbers already in the record (MATCH/MISMATCH)

| Record claim | Source | Our computed value (n=24, CNN) | Verdict |
|---|---|---|---|
| Certified floor mean 8.48cm | README.md L803 | 8.475379 cm | **MATCH** (rounds to 8.48) |
| Certified floor SD 1.81cm (ddof=1) | commit an earlier run | 1.812193 cm (ddof=1) | **MATCH** (rounds to 1.81) |
| 10 of 24 seeds show anomalous_recovery | README.md L799 | 10/24 = 0.4167 | **MATCH** |
| Gate 6.50cm = 10th percentile of 24-seed floor distribution | constants.py L85 | 10th pct, linear method = 6.500000 cm | **MATCH** (confirms "linear"/default numpy interpolation is the method actually used) |
| 5th-percentile candidate 5.65cm (rejected in favor of 6.50cm gate) | constants.py L87 | 5th pct, linear method = 5.650000 cm | **MATCH** |
| Floor range 5.45–12.27cm | README.md L1025 | range = [5.454545, 12.272727] | **MATCH** |

No mismatch found on the CNN side. The ResNet1D side has no n=24 record
number to check against yet (that number does not exist until the sweep
finishes), so no check could be run there.

## 4. Provisional n=20 results (ResNet1D seeds 101–120 vs. matched CNN seeds)

All numbers from `lane7_output_raw.txt`.

### 4a. Descriptive stats — floor sets

| | CNN (n=24, full/final) | ResNet1D (n=20, PROVISIONAL) |
|---|---|---|
| mean | 8.475379 cm | 10.386364 cm |
| SD (ddof=1) | 1.812193 cm | 3.436857 cm |
| median | 8.000000 cm | 12.272727 cm |
| 5th pct (linear / weibull / nearest-rank) | 5.650000 / 5.465909 / 5.454545 | 3.868182 / 1.495455 / 1.363636 |
| 10th pct (linear / weibull / nearest-rank) | 6.500000 / 6.000000 / 5.500000 | 5.800000 / 4.200000 / 4.000000 |
| range | [5.454545, 12.272727] | [1.363636, 15.000000] |

### 4b. Mann-Whitney U (independent, CNN n=24 vs ResNet1D n=20)

- scipy `mannwhitneyu`: U = 123.5, p = 0.00565569
- Cross-check: Monte Carlo permutation test (100,000 permutations, ties via
  midranks): p = 0.00447000
- **MATCH** (|Δp| = 0.0012, both well below any conventional α)

### 4c. Hodges-Lehmann shift (independent samples)

- HL(ResNet1D − CNN) = 2.727273 cm
- 10,000-resample bootstrap 95% CI: [1.000000, 4.272727] cm

### 4d. Paired Wilcoxon signed-rank (matched seeds 101–120, n=20)

- 1 zero-difference pair (seed 116, dropped under `zero_method='wilcox'`)
- Wilcoxon statistic = 44.0, p = 0.03997423
- Paired HL (median of 20 per-seed differences) = 4.181818 cm
- 10,000-resample bootstrap 95% CI: [1.454545, 4.590909] cm

Full per-seed difference table is in `lane7_output_raw.txt` §4.

### 4e. Fraction of seeds with floor < 6.50 cm (safety gate)

| | count/n | fraction | Clopper-Pearson 95% CI |
|---|---|---|---|
| CNN (n=24) | 2/24 | 0.0833 | [0.0103, 0.2700] |
| ResNet1D (n=20, PROVISIONAL) | 3/20 | 0.1500 | [0.0321, 0.3789] |

### 4f. anomalous_recovery counts

- CNN: 10/24 = 0.4167
- ResNet1D (n=20, PROVISIONAL): 7/20 = 0.3500

### 4g. ResNet1D restart data (CNN CSV has no equivalent columns — N/A for CNN)

Only seed 117 shows a collapse+restart (`collapse_count=1`,
`escaped_on_attempt=1`); all other 19 available seeds show 0/0.

- Overall restart rate (n=20, PROVISIONAL): 1/20 = 0.0500, Clopper-Pearson
  95% CI: [0.0013, 0.2487]

### 4h. ResNet1D 10th-percentile gate candidate (n=20, PROVISIONAL)

- linear: 5.800000 cm
- Weibull: 4.200000 cm
- nearest-rank: 4.000000 cm
- Bootstrap 95% CI on linear 10th percentile: [1.363636, 10.636364] cm (wide
  — driven by n=20 and the seed-110 outlier below)

This is **not** a like-for-like replacement for the existing 6.50cm gate:
that gate was derived from the CNN's 24-seed distribution, this is
ResNet1D's own (incomplete) distribution, shown for context only.

### 4i. Seeds where ResNet1D floor exactly equals CNN floor

- Seed 116 only: both floors = 10.909091 cm.

## 5. Surprises

- **Direction of effect**: across the matched 20 seeds, ResNet1D floors run
  *higher* than CNN floors on average (mean 10.39cm vs 8.48cm, HL shift
  ≈2.7–4.2cm depending on paired/unpaired), and this is nominally
  significant at n=20 (Mann-Whitney p=0.0057, Wilcoxon p=0.040). Whether "higher floor" is better or
  worse in this framework's convention was outside the scope of this pass, and no interpretation is offered
  here — stated as a raw descriptive fact for the researcher to interpret
  against the existing Log 11 finding (ResNet1D vs CNN accuracy difference
  not significant on the clean-data classification task, a different
  comparison from this degradation-floor one).
- **Seed 110 outlier**: ResNet1D floor = 1.3636 cm, far below every other
  available ResNet1D seed (next-lowest is seed 112 at 4.0cm; the rest
  cluster 6–15cm). This single point pulls the ResNet1D 5th/10th-percentile
  estimates down substantially and widens their bootstrap CIs. Worth
  re-examining once seed 110's full trajectory/log is available — not
  flagged as an error, just an outlier that a 4-seed-larger n=24 set may or
  may not confirm.
- Because the Wilcoxon test drops the one zero-difference pair (seed 116),
  effective paired n is 19, not 20 — disclosed rather than silently
  absorbed into the reported n=20.

## 6. Open questions (not resolved here)

1. Will seeds 121–124's ResNet1D floors fall inside or outside the current
   [1.36, 15.0] cm range? The current 10th-percentile estimate has a wide
   bootstrap CI and could shift meaningfully with 4 more points.
2. Is the "higher ResNet1D floor" direction consistent with how this
   framework interprets floor value (i.e., is higher floor evidence of
   *worse* degradation robustness, since higher naive_floor_cm presumably
   means significance broke down at a shallower/easier condition, or the
   reverse)? This report deliberately does not resolve that interpretive
   question — it is outside this lane's remit (numbers only, no
   interpretation of record claims).
3. No README/record number yet exists for a certified n=24 ResNet1D floor
   mean/SD/gate to check against — that check is deferred to a re-run of
   this same script once the sweep completes.

## 7. File list

- `PREREG.md` — pre-registered analysis plan (written before any result
  inspected)
- `run_lane7_analysis.py` — the single script producing every number in
  this report
- `lane7_output_raw.txt` — full unedited stdout of that script
- `REPORT.md` — this file

All three input files read (never modified):
- `Source/logs/degradation_aware_final_results.csv`
- `Source/logs/degradation_aware_resnet_final_results.csv`
- `Source/constants.py`

No file outside `audit_fixes/lane7/` was written. The running sweep (pid
7429) and its log/output files were not touched.

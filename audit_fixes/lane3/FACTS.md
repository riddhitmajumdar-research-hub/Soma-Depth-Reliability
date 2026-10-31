# Lane 3 — Facts from code/logs only

Status: COMPLETE. Questions 1-9 were researched read-only, and every
numeric claim below was re-run and verified before being written here (see
the verification commands in each section).
The citation check (Chen et al. / 26.47dB / 13.58dB) was run directly.

All commands below were run read-only from the repo root
`<REPO_ROOT>`; raw grep output for
the citation check is also saved in
`audit_fixes/lane3/citation_check_scratch.txt`. The running ResNet1D sweep
(pid 7429) and its log/CSV were only ever read, never written to or touched
as a process.

---

## Citation check: Chen et al. / "wearable" / 26.47dB / 13.58dB sources

### (a) "Chen" as author of the 27.05dB citation, and "wearable" describing that sensor

`grep -rn "Chen" README.md Source/*.py SOMA_Consolidated_Methodology.py`:

- [README.md:722](../../README.md) — "I went looking for a better match and found
  one: Chen et al. 2022, *Biosensors* (PMC9312695) -- a real, peer-reviewed
  wearable, skin-contact MEMS heart-sound sensor with a 10-630Hz working band
  (covers my 20-500Hz band almost exactly), SNR=27.05dB, validated against a
  real human subject and a commercial 3M Littmann stethoscope."
- [README.md:921](../../README.md) — "Fix #1 (Log 18) sourced SOMA's operating
  SNR (27.05dB) from Chen et al. 2022, and corroborated it against two other
  independent wearable MEMS heart-sound sensor papers in the same band —
  26.47dB and 13.58dB."
- [Source/constants.py:11](../../Source/constants.py#L11) — "# Source: Chen et al.
  2022, Biosensors (PMC9312695) — real wearable"
- [Source/floor_characterization.py:24](../../Source/floor_characterization.py#L24)
  — "# REPLACED: Chen et al. 2022, Biosensors (PMC9312695) -- wearable"
- [Source/degradation_aware_cnn.py:6](../../Source/degradation_aware_cnn.py#L6) —
  "# 67.03dB (1kHz, free-field microphone) -> 27.05dB (Chen et al. 2022,"
- [Source/noise_robustness_stress_test.py:5](../../Source/noise_robustness_stress_test.py#L5)
  — "# from Chen et al. 2022 (Biosensors, PMC9312695), and corroborated it
  against"

`grep -rn "wearable" ...` additionally hits (beyond the Chen lines above):

- [README.md:22](../../README.md), [README.md:32](../../README.md) — general
  project framing ("wearable organ monitoring", "planned wearable expansion"),
  not tied to a specific citation.
- [Source/constants.py:14](../../Source/constants.py#L14) — "# independent wearable
  MEMS heart-sound papers in-band (26.47dB, 13.58dB)."
- [Source/constants.py:21](../../Source/constants.py#L21) — "# (wearable MEMS
  heart-sound), not a hypothetical or cross-domain worst"
- [Source/degradation_aware_cnn.py:7](../../Source/degradation_aware_cnn.py#L7) —
  "# Biosensors PMC9312695 -- wearable skin-contact MEMS heart-sound sensor,"
- [Source/floor_characterization.py:28](../../Source/floor_characterization.py#L28)
  — "# wearable MEMS heart-sound papers in the same band (26.47dB, 13.58dB)."
- [Source/noise_robustness_stress_test.py:6](../../Source/noise_robustness_stress_test.py#L6)
  — "# two OTHER independent wearable MEMS heart-sound sensor papers in the
  same"

**MATCH**: every occurrence of "Chen" in the repo attributes 27.05dB
specifically to "Chen et al. 2022, Biosensors (PMC9312695)", and every one of
those same passages also calls the sensor "wearable" (specifically "wearable,
skin-contact MEMS heart-sound sensor"). The attribution is internally
consistent everywhere it appears — README.md, constants.py, and three other
Source/*.py files all agree on author/year/journal/PMCID.

### (b) Sources for 26.47dB and 13.58dB — named or SNR-only?

`grep -rn "26.47"` and `grep -rn "13.58"` (full result sets above) return
**only** these six files: README.md (lines 722, 921, 923, 925, 935),
Source/constants.py (lines 14, 26), Source/floor_characterization.py (line
28), Source/noise_robustness_stress_test.py (lines 7, 312, 318),
Source/noise_robustness_stress_test_clean_cnn.py (line 332), and
SOMA_Consolidated_Methodology.py (line 1684).

In every single occurrence, the two corroborating papers are referred to
only as "two more independent wearable MEMS heart-sound sensor papers" /
"two OTHER independent wearable MEMS heart-sound sensor papers" — by SNR
value and one-line description only. No author name, year, journal title,
DOI, or PMCID is given for either the 26.47dB or the 13.58dB source anywhere
in the repository (README.md, any Source/*.py, or
SOMA_Consolidated_Methodology.py).

**MISMATCH / gap vs. what a reader would expect**: README.md:722 explicitly
frames this as "Three independent groups, three different sensor designs,"
and constants.py:26's comment calls SENSOR_SNR_DB_WORST_CASE (13.58) "a real,
documented sensor-to-sensor SNR figure" — but neither of the two
corroborating papers is actually named anywhere in the codebase or README.
Only the Chen et al. 2022 (PMC9312695) source for 27.05dB is a checkable,
named citation. The 26.47dB and 13.58dB values are asserted to be real,
peer-reviewed, independent papers, but the repo provides no citation
(author/title/journal/year/identifier) a reader could verify them against —
this is a fully unresolved sourcing gap, not something this pass can
close by searching harder; it is not present in the code at all.

**Confidence**: confirmed by exhaustive grep — every literal occurrence of
"Chen", "wearable", "27.05", "26.47", and "13.58" in the three top-level
locations specified (README.md, Source/*.py, SOMA_Consolidated_Methodology.py)
is quoted above in full; nothing was filtered out.

---

## Q1 — Stage 4 dataset sizes

**(a) Training realizations per class.**
[Source/degradation_aware_final_sweep.py:29-30](../../Source/degradation_aware_final_sweep.py#L29):
`TRAIN_DEPTHS_CM = [0, 3, 6, 9, 12]`, `N_SAMPLES_PER_DEPTH_PER_CLASS = 25`.
That's 25 independent realizations/class at each of 5 training depths → 3
classes × 5 depths × 25 = 375 total training samples, corroborated verbatim
in [Source/degradation_aware_resnet_final_sweep.py:57](../../Source/degradation_aware_resnet_final_sweep.py#L57)
comment: "only 375 training samples and 150 full-batch epochs."
Confidence: confirmed by code.

**(b) N_SAMPLES_PER_CELL (eval, record claims 60→300).**
[Source/degradation_aware_final_sweep.py:36](../../Source/degradation_aware_final_sweep.py#L36):
`N_SAMPLES_PER_CELL = 300   # raised from 60`; same in
[Source/degradation_aware_resnet_final_sweep.py:82](../../Source/degradation_aware_resnet_final_sweep.py#L82).
**MATCH** with the record's "60 → 300." Caveat: `N_SAMPLES_PER_CELL` is not a
`constants.py` constant — it's a per-script literal, and the literal `60` is
still live (not "raised") in several *other*, non-final scripts:
Source/degradation_aware_cnn.py:46, degradation_aware_cnn_calibration_retrain.py:33,
extend_20hz.py:21, noise_robustness_stress_test.py:56,
noise_robustness_stress_test_clean_cnn.py:62, power_law_sensitivity.py:26,
power_law_sensitivity_certified.py:32, degradation_aware_calibration_corrected.py:39.
This is worth flagging (not resolving): the "60→300" fix applies only to the
two *final* sweep scripts, not to every script in the pipeline that computes
per-cell accuracy — a reader of the record could reasonably assume it was
raised everywhere. Confidence: confirmed by code (comment + literal).

**(c) Depth grid: 23 densified — confirmed; 82 "joint cells" — NOT FOUND.**
Verified myself:
```
BASE points: 12  DENSIFIED points: 11  merged unique: 23
```
([Source/constants.py:58](../../Source/constants.py#L58) `BASE_EVAL_DEPTHS_CM = np.linspace(0, 15, 12)`,
[Source/constants.py:65](../../Source/constants.py#L65) `DENSIFIED_DEPTHS_CM` = 11 values) —
**MATCH** with the record's "23 in the densified family." I additionally grepped
Source/*.py, all logs/CSVs, and README.md for `"82"`, `"joint"`, `"joint_cell"`
myself — no hits describing an "82 joint cells" quantity anywhere in the repo.
**NOT FOUND**: there is no code or log basis for "82 joint cells"; the record's
sourcing for that specific number is unresolved.

**(d) Train/test split rule: grouped by independent realization, via
GroupShuffleSplit (not GroupKFold).**
[Source/cnn_classifier.py](../../Source/cnn_classifier.py) `build_raw_dataset_with_groups()`
docstring: "a plain train_test_split on that pool lets windows from the same
realization land in both train and test. `groups` lets callers use
GroupShuffleSplit instead, so the split is by realization, not by window."
[Source/resnet_classifier.py:73,78](../../Source/resnet_classifier.py#L73) and
[Source/cnn_classifier.py:124](../../Source/cnn_classifier.py#L124):
`gss.split(X, y_encoded, groups=groups)`. For the ETS Montreal phantom the
grouping unit is the whole acquisition sequence:
[Source/ets_montreal_path2_retrain.py:15-16](../../Source/ets_montreal_path2_retrain.py#L15)
"Held out by WHOLE SEQUENCE (Acqui6), not random split -- A-lines within one
sequence are correlated (same phantom, same session)"; `TRAIN_SEQUENCES =
["Acqui1"..."Acqui5"]`, `TEST_SEQUENCES = ["Acqui6"]`. No literal
`GroupKFold` usage exists anywhere in Source/. Confidence: confirmed by code.

---

## Q2 — 24-seed sweep provenance: CNN vs ResNet1D, device, epochs, LR, timing, weights

- **Certified (original) CNN sweep**: [Source/degradation_aware_final_sweep.py](../../Source/degradation_aware_final_sweep.py)
  → produces [Source/logs/degradation_aware_final_results.csv](../../Source/logs/degradation_aware_final_results.csv),
  complete, all 24 seeds. I verified this myself: `24` rows in that CSV.
- **Newer, still-running ResNet1D sweep**: [Source/degradation_aware_resnet_final_sweep.py](../../Source/degradation_aware_resnet_final_sweep.py)
  → produces [Source/logs/degradation_aware_resnet_final_results.csv](../../Source/logs/degradation_aware_resnet_final_results.csv).
  I verified this myself: **15 of 24 seed rows** present (seeds 101–115), file
  (this is the running pid 7429; I only read it).
  [Source/gradcam_counterfactual_occlusion.py:9](../../Source/gradcam_counterfactual_occlusion.py#L9)
  confirms in its own comment: "NOT the in-progress 24-seed ResNet1D retest."
- **Device**: grep for `device`, `cuda`, `mps` in both sweep scripts returns
  no real hits (the ResNet1D script has one unrelated comment line containing
  neither). **NOT FOUND**: neither script explicitly sets a torch device.
  Since neither calls `.to()`/`.cuda()`, PyTorch's default is CPU, but that is
  an inference from absence, not an explicit citation — reported as such.
- **Epochs**: CNN sweep [Source/degradation_aware_final_sweep.py:70](../../Source/degradation_aware_final_sweep.py#L70)
  `n_epochs=60`. ResNet1D sweep [Source/degradation_aware_resnet_final_sweep.py:84](../../Source/degradation_aware_resnet_final_sweep.py#L84)
  `N_EPOCHS = 150   # REVISED from SimpleCNN's 60`.
- **Learning rate**: both `torch.optim.Adam(..., lr=0.001)` — CNN sweep line
  74, ResNet1D sweep lines 113 and 137.
- **Wall-clock per seed**: only logged for the ResNet1D sweep, in
  [Source/logs/degradation_aware_resnet_final_sweep_log.txt](../../Source/logs/degradation_aware_resnet_final_sweep_log.txt),
  e.g. "Seed 101: ... (2852s)", "Seed 103: ... (49393s)" (~13.7h, an outlier),
  "Seed 108: ... (5286s)" — wide spread, 2553s–51550s observed. The CNN
  sweep's `final_sweep_log.txt` has **no** timing strings at all — NOT FOUND
  for CNN per-seed wall-clock.
- **Saved weights beyond seed 121**: `Source/models/` holds only one-off
  named checkpoints (seed 121, plus other special-purpose models), i.e. NOT a
  full 24-seed CNN set. `Source/seed_weights_resnet/` holds per-seed
  `degradation_aware_resnet_seed101.pt` … `seed116.pt` — I verified myself:
  **16 files**, matching the ResNet1D sweep's in-progress state (through seed
  116, one ahead of the 115 completed CSV rows — seed 116 was mid-run at read
  time). So: no, trained weights are not saved for any CNN seed except 121;
  yes, weights ARE saved per-seed for the (separate, in-progress) ResNet1D
  sweep. Confidence: confirmed by code/filesystem, independently re-verified.

---

## Q3 — ETS Montreal (phantom) shuffled-label null test

File: [Source/ets_montreal_path2_retrain.py](../../Source/ets_montreal_path2_retrain.py).

- **Shuffle scope: global, not within-group.** `y_shuffled =
  np.random.permutation(y_train)` permutes all training labels together
  across the whole pooled training set — not per acquisition sequence.
- **Split is grouped by acquisition sequence**, confirmed: lines 15-16
  comment "Held out by WHOLE SEQUENCE (Acqui6), not random split"; the run
  log [Source/logs/ets_path2_run_log.txt](../../Source/logs/ets_path2_run_log.txt)
  literally logs "Building training set (Acqui1-5)..." / "Building held-out
  test set (Acqui6)...".
- **Why the null mean is 49.91%**: the same run log states "Total test
  windows: 16384 (8192 inclusion, 8192 background)" — an exactly balanced
  2-class test set, so chance-level accuracy under label-shuffled training is
  theoretically 50%. I recomputed this myself directly from
  [Source/logs/ets_path2_shuffle_results.csv](../../Source/logs/ets_path2_shuffle_results.csv)
  (200 shuffle runs):
  ```
  n_shuffles: 200   mean: 0.49905242919921877   sd (ddof=1): 0.03271303659705176
  ```
  **MATCH**: mean = 49.91% (matches the record exactly), and it lands almost
  exactly on the 50% chance level implied by the perfectly balanced test set.
  SD ≈ 3.27 percentage points (ddof=1) reflects run-to-run variance across
  200 independently-shuffled-and-retrained models, not a single-test binomial
  SD. Confidence: confirmed by code + independently recomputed from the
  raw CSV.

---

## Q4 — "three of the 24 seeds' crossover ambiguities" (Methodology E)

**Correction to the initial research pass**: the phrase does exist in this
repo, in [README.md:1101](../../README.md) (a later grep found it; the initial
search had missed README.md for this exact phrase): "Confirmation that the three seeds with
>1.0cm crossover ambiguity are assigned by first-FDR-failed depth, matching
the README's existing description." This is the record's own project log, not
an independent check — so it doesn't by itself establish a code-verified
basis for "three."

**I found the actual quantitative basis myself.** In the certified 24-seed
CNN CSV ([Source/logs/degradation_aware_final_results.csv](../../Source/logs/degradation_aware_final_results.csv)),
each row has `naive_floor_cm` (first FDR-failed depth) and
`last_significant_depth_cm` (last depth still significant before that). I
computed `naive_floor_cm - last_significant_depth_cm` for all 24 seeds:
```
seed 101: 9.545 - 8.182 = 1.364cm
seed 112: 12.273 - 10.909 = 1.364cm
seed 122: 12.273 - 10.909 = 1.364cm
count of seeds with gap > 1.0cm: 3
```
**MATCH**: exactly 3 seeds (101, 112, 122) have a >1.0cm gap between the
first-failed depth and the last-significant depth in the certified CNN
sweep — this is a real, code-derived quantity, and it equals the record's
"three seeds." This is a different (and correct) metric from the
`anomalous_recovery` boolean flag I also checked (which is a different
concept — a later depth becoming spuriously re-significant — and is True for
10/24 CNN seeds, not 3; do not conflate the two). The actual "ambiguity"
being described is the untested gap between two adjacent tested depths,
i.e., the true floor for these 3 seeds is only known to lie somewhere within
a >1cm window rather than pinned to a single depth. Confidence: confirmed —
computed directly from the raw CSV in this pass, command and output
shown above.

---

## Q5 — Fig. 20 (capacity-confound, three evaluation protocols) — NOT FOUND

I grepped Source/*.py and outputs/figures_earlier/ for `"capacity"`,
`"confound"`, `"protocol"`, `"fig.*20"`, `"fig20"` myself; no file, script,
or figure titled or corresponding to "Fig. 20" exists anywhere in this repo,
and `outputs/figures_earlier/` (28 PNGs + manifest) has no capacity/confound/
protocol-named entry. The closest candidate is
[Source/repeated_grouped_cv_log11.py](../../Source/repeated_grouped_cv_log11.py),
which does compare three evaluation protocols for CNN vs. ResNet1D — original
single grouped split, leakage-corrected single split, and 15× repeated
grouped 5-fold CV (this is the same "one tension I'm not resolving" table
quoted in README.md around line ~1090: "Original (leaked split) p=0.22" /
"Leakage-corrected single split: reversal within noise" / "15× repeated
grouped 5-fold CV: p=0.0002, d=0.59"). This is plausibly what "Fig. 20"
refers to conceptually, but there is no figure file, script output, or
manifest entry in this repo that is actually labeled "Fig. 20" or generates
it under that name. **NOT FOUND** as a literal artifact — flagging the
plausible source table above rather than guessing a match.

---

## Q6 — Tracking distance `d` usage outside TOF

I independently grepped the entire Source/ tree for `distance` myself:
`grep -rln "distance" Source/*.py | grep -v physics_model.py` → **empty**.
The only occurrences are in [Source/physics_model.py](../../Source/physics_model.py):
- `def time_of_flight_s(youngs_modulus_kpa, distance_m=0.010):`
- comment: "distance_m: representative tracking distance (10mm), based on
  real elastography devices."
- `return distance_m / c`
- call site: `tof = time_of_flight_s(youngs_modulus_kpa)`

**Confirmed: no other use.** The tracking distance is used exclusively
inside the time-of-flight calculation in `physics_model.py`; no other file
in the codebase references it. Confidence: confirmed by exhaustive grep, run
independently by me.

---

## Q7 — ResNet1D restart-count logs and current sweep progress

- Restart/resume mechanism, in [Source/degradation_aware_resnet_final_sweep.py](../../Source/degradation_aware_resnet_final_sweep.py):
  `print(f"Found existing results for seeds: {sorted(already_done)}")` /
  `print(f"Seed {seed}: already done, skipping.")`; `f.flush()  # checkpoint
  immediately -- a crash at seed 19 costs nothing from 1-18`; `MAX_RESTARTS =
  5` (comment: "sourced from Log 11 exactly"). The run log
  ([Source/logs/degradation_aware_resnet_final_sweep_log.txt](../../Source/logs/degradation_aware_resnet_final_sweep_log.txt))
  shows a real resume event: "Found existing results for seeds: [101]" /
  "Seed 101: already done, skipping."
- Related (but separate) restart-on-collapse safeguards also exist in
  [Source/repeated_grouped_cv_log11.py](../../Source/repeated_grouped_cv_log11.py)
  and [Source/significance_testing.py](../../Source/significance_testing.py)
  (`max_restarts = 5` in both) — these belong to the CV/significance scripts,
  not the 24-seed sweep itself; don't conflate them.
- **Current progress** (I verified this myself, read-only, without touching
  pid 7429):
  ```
  CNN sweep rows: 24         (complete)
  ResNet sweep rows so far: 15   seeds: 101–115
  ```
  The sweep log's last completed line is for seed 115 ("... (2649s)"),
  followed by "--- Seed 116 starting ---" with no completion line yet — and
  `Source/seed_weights_resnet/degradation_aware_resnet_seed116.pt` exists
  consistent with seed 116 being mid-run at the time of
  this read. **As of this pass, 15/24 ResNet1D seeds are complete; seed
  116 is in progress.**

---

## Q8 — Audit-asset inventory

Verified via `ls -la` and `head -1` (the `head -1` header check was re-run on
every CSV listed below; sizes come from an `ls -la` pass,
spot-checked for the two most important files).

| Path | Size | Modified | Holds (header / inferred) |
|---|---|---|---|
| [Source/logs/degradation_aware_final_results.csv](../../Source/logs/degradation_aware_final_results.csv) | 878 B |  | `seed,naive_floor_cm,last_significant_depth_cm,anomalous_recovery` — complete 24-seed certified CNN floor sweep |
| [Source/logs/degradation_aware_resnet_final_results.csv](../../Source/logs/degradation_aware_resnet_final_results.csv) | 1203 B |  | `seed,naive_floor_cm,last_significant_depth_cm,anomalous_recovery,real_model_collapse_count,real_model_escaped_on_attempt,control_test_mean_overall,control_sanity_ok` — in-progress 24-seed ResNet1D sweep, 15/24 rows (verified) |
| [Source/logs/degradation_aware_resnet_final_sweep_log.txt](../../Source/logs/degradation_aware_resnet_final_sweep_log.txt) | 2851 B |  | plaintext progress log, per-seed timing in seconds |
| [Source/logs/final_sweep_log.txt](../../Source/logs/final_sweep_log.txt) | 4735 B |  | plaintext progress log for CNN "final" sweep (no timing strings) |
| [Source/logs/degradation_aware_per_class_recall_by_depth.csv](../../Source/logs/degradation_aware_per_class_recall_by_depth.csv) | 2633 B |  | `depth_cm,overall_acc_mean,overall_acc_std,cirrhosis_recall_mean,...` — per-class recall vs. depth with CIs |
| [Source/logs/resnet_per_class_metrics_grouped_split.csv](../../Source/logs/resnet_per_class_metrics_grouped_split.csv) | 254 B |  | `class,precision,recall,f1,support,roc_auc_ovr` — per-class metrics, single grouped split, ResNet1D |
| [Source/logs/log11_repeated_grouped_cv_raw.csv](../../Source/logs/log11_repeated_grouped_cv_raw.csv) | 4403 B |  | `repetition,fold,baseline_acc,cnn_acc,resnet1d_acc` — raw per-fold accuracies, 15 reps × 5 folds |
| [Source/logs/log11_repeated_grouped_cv_summary.csv](../../Source/logs/log11_repeated_grouped_cv_summary.csv) | 569 B |  | `comparison,repeated_cv_mean_diff,repeated_cv_ci_lo,repeated_cv_ci_hi,permutation_p_value,...` — repeated-CV vs. original-split comparison |
| [Source/logs/log11_repeated_grouped_cv_run.log](../../Source/logs/log11_repeated_grouped_cv_run.log) | 1852 B |  | plaintext run log for repeated-CV script |
| [Source/logs/log11_grouped_paired_ttests.csv](../../Source/logs/log11_grouped_paired_ttests.csv) | 231 B |  | original single-split paired t-test results (Log 11 baseline) |
| [Source/logs/gate_statistic_verification.csv](../../Source/logs/gate_statistic_verification.csv) | 358 B |  | `statistic,value_cm,note` — MAX_VALID_DEPTH_CM gate-candidate verification |
| [Source/logs/ets_path2_shuffle_results.csv](../../Source/logs/ets_path2_shuffle_results.csv) | 4121 B |  | `shuffle_idx,control_acc` — 200-row shuffled-label null distribution (verified: mean 0.4991, sd 0.0327) |
| [Source/logs/ets_path2_run_log.txt](../../Source/logs/ets_path2_run_log.txt) | 3838 B |  | plaintext run log, real-RF training/eval, elapsed-time stamps |
| [Source/logs/seed121_run_log.txt](../../Source/logs/seed121_run_log.txt) | 687 B |  | run log for seed-121 verification/retrain |
| Source/models/soma_certified_model_seed121_corrected_noise.pt | 58,835 B |  | saved seed-121 CNN weights — the single "certified" checkpoint |
| Source/models/stage4_cnn_model.pt | 58,187 B |  | earlier Stage-4 CNN weights checkpoint |
| Source/models/soma_calibration_corrected_model_weights.pt | 58,779 B |  | calibration-corrected CNN weights |
| Source/models/ets_path2_real_model.pt | 58,115 B |  | real-RF-trained CNN for ETS Montreal Path 2 phantom |
| Source/seed_weights_resnet/degradation_aware_resnet_seed101.pt … `seed116.pt` | ~35.9 KB each, 16 files (verified count) |  →  | per-seed ResNet1D weights for the in-progress 24-seed sweep (seeds 101–116 present) |

---

## Q9 — Sensor modality per README/code

README.md frames the overall concept as "Passive acoustic elastography —
sensing tissue mechanical properties from sound alone, with no external
ultrasound pulse," for "continuous, low-cost, wearable organ monitoring"
(README.md:20-22). The concrete, cited sensor model backing the noise-floor
constant is consistent everywhere it's described:
- README.md:722 / Source/constants.py:11-14 / Source/floor_characterization.py:24-28 /
  Source/degradation_aware_cnn.py:6-7: "wearable, skin-contact MEMS
  heart-sound sensor" (Chen et al. 2022, Biosensors, PMC9312695), 10-630Hz
  band.
- README.md:720: "the device is skin-contact, not open-air" — explicitly
  distinguishing it from the earlier, rejected 67.03dB free-field-microphone
  assumption ("a free-field microphone isn't the right sensor category to
  begin with").

**Conclusion**: the code and README consistently assume the physical sensor
is a wearable, skin-contact MEMS piezoelectric heart-sound sensor — not an
open-air microphone, not an accelerometer. This is internally consistent
across every file that mentions it. Confidence: confirmed by code + README.

---

## Explicit NOT-FOUND / open-items summary

1. **Q1(c)**: "82 joint cells" — no basis anywhere in Source/, logs, or README.
2. **Q2**: explicit `torch.device`/cuda/mps selection — not set in either
   sweep script (CPU is inference-from-absence only, not an explicit
   citation). CNN sweep per-seed wall-clock timing — not logged anywhere.
3. **Q5**: a literal "Fig. 20" artifact (script, figure file, or manifest
   entry) — does not exist in this repo under that name; the closest
   conceptual match (three-protocol CNN-vs-ResNet1D comparison) is
   Source/repeated_grouped_cv_log11.py, but it is not labeled "Fig. 20."
4. **Citation check**: the two corroborating sensor papers behind 26.47dB
   and 13.58dB are never named (no author/title/journal/year/identifier)
   anywhere in the repository — only their SNR values and sensor-class
   description are given. This is a real, unresolved sourcing gap in the
   record's evidentiary chain, reported per the "stop and report contradictions,
   don't resolve them" protocol rule rather than fixed here.

All other numeric/code claims above were independently re-run and verified
in this pass (commands and outputs shown inline) in addition to the
initial research pass.

---

## Follow-up: tracing "jointly across all 82 frequency x depth cells" (clean-CNN 4.30cm floor)

This resolves the Q1(c) "NOT FOUND" item above — searched for and reproduced
from code rather than found as a literal string.

**Script**: [Source/floor_characterization_cnn.py](../../Source/floor_characterization_cnn.py)
— "Stage 4 (CNN version) -- Floor Characterization Sweep, fully certified"
(file header, line 2). This is the script that produces the 100Hz corrected
floor discussed in README.md's Log 18/19 sections (90%→83%→80%→71.67%→...
collapse pattern, 4.30cm floor after the 100Hz densification fix).

**The FDR family is a single joint one across ALL four frequencies' cells**,
not per-frequency. From [Source/floor_characterization_cnn.py:145-146](../../Source/floor_characterization_cnn.py#L145):
```python
all_p_values = [r["p_value"] for r in results]
significant_flags = bh_fdr_correction(all_p_values)
```
`results` is built by one nested loop over `for freq in FREQUENCIES_HZ: for
depth_cm in DEPTHS_BY_FREQ[freq]:` (lines 108-109) — i.e. every (frequency,
depth) cell across all 4 frequencies is appended to the same flat `results`
list before Benjamini-Hochberg correction is applied once, jointly, over
all of them. The script's own progress-print at line 105 confirms this
framing exactly: `print(f"\nRunning sweep: {total_cells} total (freq, depth)
cells " f"x {N_SAMPLES_PER_CELL} samples/cell ...")` where `total_cells =
sum(len(DEPTHS_BY_FREQ[f]) for f in FREQUENCIES_HZ)` (line 104).

**Grid construction** ([Source/floor_characterization_cnn.py:20-35](../../Source/floor_characterization_cnn.py#L20)):
```python
BASE_DEPTHS_CM = BASE_EVAL_DEPTHS_CM        # from constants.py: np.linspace(0, 15, 12) -> 12 points
FINE_NEAR_ORIGIN_CM = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.5, 1.8, 2.1]   # 11 points
DENSE_DEPTHS_CM = np.sort(np.unique(np.concatenate([BASE_DEPTHS_CM, FINE_NEAR_ORIGIN_CM])))

FINE_100HZ_CM = [4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9, 6.1, 6.3, 6.5]        # 12 points
DENSE_100HZ_CM = np.sort(np.unique(np.concatenate([BASE_DEPTHS_CM, FINE_100HZ_CM])))

FREQUENCIES_HZ = [20, 100, 250, 500]
DEPTHS_BY_FREQ = {
    20: BASE_DEPTHS_CM,      # 12 depths, no densification
    100: DENSE_100HZ_CM,     # densified 4-7cm (the borderline-call fix)
    250: DENSE_DEPTHS_CM,    # densified near origin (grid-artifact fix)
    500: DENSE_DEPTHS_CM,    # densified near origin (grid-artifact fix)
}
```

**I reproduced this grid construction verbatim and counted the cells myself**
(`audit_fixes/lane3/verify_82_cells.py`, output saved in
`audit_fixes/lane3/verify_82_cells_output.txt`):
```
20Hz: 12 depths
100Hz: 24 depths
250Hz: 23 depths
500Hz: 23 depths
TOTAL (freq, depth) CELLS in the single joint bh_fdr_correction(all_p_values) family: 82
```
No overlap-collapse surprises: `BASE_DEPTHS_CM` (12 points, 0-15cm in
1.3636cm steps) shares no values with `FINE_NEAR_ORIGIN_CM` (11 points, all
< 2.11cm, none landing exactly on a `linspace(0,15,12)` grid point) or
`FINE_100HZ_CM` (12 points, all in 4.3-6.5cm, likewise none coinciding), so
`np.unique` doesn't silently drop anything: 12+11=23 for 250/500Hz, 12+12=24
for 100Hz.

**MATCH**: 12 + 24 + 23 + 23 = **82**. The record's "jointly across all 82
frequency x depth cells" is fully reproducible from
`Source/floor_characterization_cnn.py`'s own grid-construction code and its
single joint `bh_fdr_correction(all_p_values)` call — this is not a
mysterious or unsourced number; it is exactly the total cell count this
script itself prints and iterates over. Confidence: confirmed, reproduced
independently in this pass from the actual grid-construction code
(not copied from any log or comment stating "82").

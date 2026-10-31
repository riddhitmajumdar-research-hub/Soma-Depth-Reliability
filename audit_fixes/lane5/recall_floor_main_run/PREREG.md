# Lane 5 Pre-Registration: Per-Class Certification Harness

Status: PRE-REGISTERED before any run. This file is written before Step 2/3
execution and is not edited after runs begin, except to append a dated
addendum if a deviation is discovered.

## 0. Provenance (discovered in this pass, not assumed)

| Role | File | Function(s) |
|---|---|---|
| Certified 24-seed sweep (seeds 101-124) | `Source/degradation_aware_final_sweep.py` | `run_one_seed`, `build_degradation_aware_dataset`, `train_cnn`, `generate_degraded_windows`, `bh_fdr_correction` |
| Seed-121 verification / weight save | `Source/retrain_seed121_verify.py` | imports the above unmodified; saves `models/soma_certified_model_seed121_corrected_noise.pt` |
| Noise-robustness / per-class recall (certified model, single depth family) | `Source/noise_robustness_stress_test.py` | inline per-class recall block inside its main sweep; own copy of `bh_fdr_correction` |
| Per-class recall vs. depth (certified seed121 model only) | `Source/degradation_aware_per_class_recall_floor.py` | `eval_one_replicate`, `main` |
| Windowing | `Source/cnn_classifier.py` | `extract_raw_windows(signal, time, heart_rate_hz, duration=3.0, window_size=100)` |
| Physics/classes | `Source/physics_model.py` | `generate_signal`, `stiffness_classes` |

**Confirmed: no existing script computes per-class recall for the 200
shuffled-label controls, at every seed, at every depth.** That is genuinely
new analysis — Lane 5's contribution — built strictly on top of the
unmodified `train_cnn` / `build_degradation_aware_dataset` /
`generate_degraded_windows` / `bh_fdr_correction` functions above. No
training, evaluation-window, or FDR logic is reimplemented; only the
per-class bookkeeping around calls to those functions is new.

`degradation_aware_resnet_final_sweep.py` (pid 7429, running) is a
**different model family (ResNet1D)** and is not imported, read for logic
reuse beyond citation, or touched.

## 1. Seeds and floor-agreement check

- Seeds: 101-124 inclusive (same 24 as the original CSV,
  `Source/logs/degradation_aware_final_results.csv`), same code path
  (`build_degradation_aware_dataset` + `train_cnn`, unchanged).
- For each seed, this harness independently recomputes `naive_floor_cm` via
  the same aggregate-accuracy + BH-FDR logic as the original, and diffs it
  against that seed's row in the original CSV.
- If any floor differs, the run is **not** presented as a reproduction of
  the original sweep — it is disclosed as an **independent 24-seed
  replicate** (new RNG draws for eval batches are unavoidable even with
  matched training seeds, since `generate_degraded_windows` draws fresh
  noise/jitter at eval time in the original code too — this is a real
  source of run-to-run variation in the *original* protocol, not a bug
  Lane 5 introduces).

## 2. Per-class recall + permutation test (new)

For every seed (101-124) and every depth in the original `EVAL_DEPTHS_CM`
grid:
- Per-class one-vs-rest recall of the real model (already computed
  per-cell predictions; sliced by true label).
- Per-class recall of each of the 200 shuffled-label controls from that
  seed's own `run_one_seed`-equivalent training (same shuffle protocol:
  `np.random.permutation(y_train_encoded)`, one shuffle per control, 200
  controls, reused across all depths for that seed — matching the
  original's "train once per seed, evaluate at every depth" design).
- Per-class p-value: `p = (1 + #{controls with recall_c >= recall_real}) / (1 + 200)`.
- **FDR family definition (stated per your requirement):** one family =
  one (seed, class) pair, spanning all depths for that class under that
  seed. This mirrors the original aggregate protocol's family definition
  (one family = one seed, spanning all depths) applied per class instead
  of once per seed. Rationale: depths within a seed are the natural
  "multiple looks" being corrected for; classes are conceptually separate
  hypotheses (different labels), so are not pooled into one family — this
  matches how `noise_robustness_stress_test.py` treats each SNR level as
  its own family rather than pooling across SNR levels.
  **This is a judgment call, not dictated by any existing script** (no
  prior script runs a per-class-per-seed BH-FDR); flagged for review.
- First-failed-depth floor convention: same as original — first depth
  (in increasing order) where the per-class comparison is *not*
  FDR-significant is that class's floor for that seed.

## 3. Per-class "40%-recall" depth

For every seed and every class: the shallowest depth where that class's
real-model recall first drops below 0.45 (per your Step 1 spec — note the
spec text says "0.45" explicitly even though the section header rounds to
"40%"; the harness uses the literal value **0.45**, flagged here in case
0.40 was intended).

## 4. Null-distribution risk (bimodality)

Per your flag: a shuffled-label control's predictions can collapse onto a
single class, driving that class's control recall to exactly 0 or 1 and
leaving the other two classes at 1 or 0 respectively — a genuinely
bimodal, not normal, null. The harness saves the raw 200-value control
per-class-recall array (float32, not weights) at 3 representative depths
per seed (shallowest, floor-adjacent, deepest tested) specifically so this
can be histogrammed and power assessed later. No p-value validity claim is
made beyond the empirical permutation p-value itself (which does not
assume normality).

## 5. Artifacts saved per seed

- Real model weights: `audit_fixes/lane5/seed_weights/degradation_aware_seed{seed}.pt`
- Per-seed, per-depth, per-class control recall arrays (200 floats,
  float32 `.npy`): `audit_fixes/lane5/controls/seed{seed}_depth{depth}.npy`
  (only at the 3 representative depths above, to bound disk use — full
  per-depth arrays would be 24 seeds x ~23 depths x 200 floats x 3
  classes, cheap actually (~1.3MB total), so **the harness saves all
  depths**, not just 3 — the 3-depth restriction above applies only to the
  histogram figure, not to raw storage).
- Aggregate CSV row per seed: `audit_fixes/lane5/logs/lane5_per_class_results.csv`.
- Checkpoint/resume: identical pattern to the original driver — CSV
  existence + already-done seed set determines what to skip; the harness
  is safe to interrupt (e.g. laptop sleep) and rerun.
- Log file: `audit_fixes/lane5/logs/lane5_run.log` (progress + ETA).

## 6. Sensitivity-sweep config flags (for later runs, not this pass)

Defined in `audit_fixes/lane5/lane5_config.py`, defaults set to reproduce
the original exactly:

- `N_TRAIN_REALIZATIONS_PER_CELL` (default 25): maps to the original's
  `N_SAMPLES_PER_DEPTH_PER_CLASS`.
- `N_EVAL_SAMPLES_PER_CELL` (default 300): maps to the original's
  `N_SAMPLES_PER_CELL`.
  **Judgment call:** your spec names a single "N_REALIZATIONS (dataset
  size)" flag; the original script has two independent size knobs (train
  vs. eval), not one. Lane 5 exposes both rather than conflating them.
  Flagged for your review.
- `WINDOW` (default 100): maps to `extract_raw_windows`'s `window_size`
  parameter, passed through unchanged — this parameter already exists in
  the original function signature.
- `STEP`: **not implemented.** `extract_raw_windows` anchors one window
  per detected heartbeat (`beat_t`) with no independent stride/step
  parameter in its signature or body — there is no "step" concept in the
  original windowing logic to sweep. Implementing one would mean writing
  new sliding-window logic, which Step 0 of the protocol prohibits
  ("import those functions unchanged, not reimplement them"). **STOP/OPEN
  QUESTION for you:** either (a) drop the STEP axis from the sensitivity
  sweep, or (b) explicitly authorize new sliding-window code as a
  documented deviation from "import unchanged."
- None of these constants exist in `Source/constants.py` (checked by
  reading the whole file) and are not added there — they live only in
  `lane5_config.py` per your Rule 2.

## 7. Threads

`torch.set_num_threads(4)` set at the top of the harness and the benchmark
script, as the protocol specifies.

## 8. Addendum (mid-run, added after initial pre-registration)

**Added to fix a gap found by Lane 2: the original sweep
(`degradation_aware_final_sweep.py`) computed a raw p-value and an
FDR-significance flag for every depth internally, but only ever persisted
the single resulting summary floor per seed -- every other depth's raw
p-value, BH-adjusted p-value, and significance flag were discarded once
`run_one_seed` returned.**

For every seed, Lane 5 now also checkpoints the full per-depth AGGREGATE
detail (not just per-class, which PREREG section 2 already covered):
`raw_p_value`, `bh_adjusted_p_value`, and `fdr_significant`, one row per
depth, to `audit_fixes/lane5/logs/lane5_aggregate_depth_detail_seed{seed}.csv`.

`bh_adjusted_p_value` is computed by a new helper,
`bh_adjusted_pvalues()`, added in `lane5_harness.py`. It is **not** a
reimplementation of the imported `bh_fdr_correction` -- that function only
ever returns a boolean pass/fail array at one fixed alpha and cannot
produce an adjusted p-value at all. `bh_adjusted_pvalues()` is the
standard, alpha-free Benjamini-Hochberg step-up adjusted-p-value formula
(`q_(i) = min_{j>=i} (n/j) * p_(j)`, monotone-enforced, capped at 1), using
the same per-seed depth ordering/family definition as the aggregate floor
computation elsewhere in this file (section 1). Cross-check: at the exact
alpha the flag was computed at, `bh_adjusted_p_value <= alpha` must equal
`fdr_significant` for every row -- this is checked in `REPORT.md` before
being reported as trustworthy.

This addendum was added and applied after Step 3 (benchmark) was
already running in the background; it changes the per-seed checkpoint
schema going forward but does not change any already-computed floor,
p-value, or significance decision.

## 9. Addendum 2 (pre-launch, before the full 24-seed run)

Per the protocol: for every seed, all 200 shuffled-label control
models' weights are now also saved (`state_dict`, float32 -- confirmed
`SimpleCNN`/`train_cnn` never casts to any other dtype, per
`Source/cnn_classifier.py`), not only the real model's weights, to
`audit_fixes/lane5/seed_weights/controls_seed{seed}/control{000-199}.pt`.
This was added to `lane5_harness.py`'s `run_one_seed` before the full
24-seed run was launched (see `REPORT.md` for the launch record).
Disk-cost note: 24 seeds x 200 controls = 4800 additional small
`state_dict` files.

## 10. Addendum 3 (post-mortem: the run died silently, and hardening added)

**Incident.** The 24-seed run (launched under `caffeinate -i`) stopped
producing any output (mid-training of seed
103's control models, 169/200 saved) and was found to no longer be a
running process (`ps` showed no `lane5_harness.py` process at all) when
next checked -- **a ~3h45m silent gap**. No crash
traceback, no error in the log file; it simply stopped. Most likely cause:
`caffeinate -i` only blocks *idle* sleep, not a lid-close or other forced
sleep, so the machine likely slept despite it. **Not resolved/diagnosed
further here** -- flagged, not silently worked around.

**Recovery actions taken (per the protocol), all disclosed:**
1. Seed 103's partial artifacts (`seed_weights/degradation_aware_seed103.pt`,
   `seed_weights/controls_seed103/` -- 169/200 control weights, no eval
   ever ran for this seed) were moved, not deleted, to
   `audit_fixes/lane5/quarantine_seed103_partial/seed_weights/`. Seeds
   101 and 102 (full CSV rows, 200/200 control weights each) were
   confirmed untouched before and after the move.
2. `main()`'s resume logic keys off rows in `lane5_per_class_results.csv`
   only, not filesystem state -- with seed 103 quarantined and having no
   CSV row, `run_one_seed(103)` will be called fresh on resume and will
   retrain the real model and all 200 controls from scratch. Confirmed by
   reading the resume logic (section `if __name__ == "__main__"` /
   `main()` in `lane5_harness.py`), not assumed.
3. **Heartbeat added:** `lane5_harness.py` now calls `touch_heartbeat()`
   (writing an epoch timestamp + note to
   `audit_fixes/lane5/logs/heartbeat.txt`) after every control model is
   trained and saved (i.e., after every control "batch" in this harness's
   sense, since controls are trained/saved one at a time, not in literal
   mini-batches) and after every depth's evaluation completes. This is a
   liveness signal only -- it changes no protocol parameter, no training
   logic, no statistic.
4. `check_alive.sh` added: reports the heartbeat's age and whether a
   given (or auto-discovered) PID is alive, so a dead run is visible in
   seconds instead of hours.
5. **`run_chain.py` rewritten to be PID-independent** (see the script's
   own header comment for the reasoning): it now polls
   `lane5_per_class_results.csv` for all 24 expected seed rows rather
   than waiting on an OS PID, since a PID going away is not a reliable
   signal of "all 24 seeds finished" (it could mean either "done" or
   "died at seed 3").
6. **This pass did not resume the run itself.** The exact resume
   command is in `RESUME_COMMAND.txt`, to be run from a normal terminal,
   per the protocol -- along with the reasoning for switching from
   `caffeinate -i` to `caffeinate -dimsu`.

No protocol parameter (seeds, depths, N_SHUFFLES, alpha, family
definition, floor convention) changed as part of this recovery.

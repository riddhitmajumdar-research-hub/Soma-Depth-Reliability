> **Superseded.** This report was last edited while the main run was in progress, and describes it that way. The final results, the balanced-accuracy retry, the multiplicity check and the clean-trained sweep are written up in [docs/methodology/09-lane5-per-class-certification.md](../../docs/methodology/09-lane5-per-class-certification.md). The text below is unchanged. Paths in it refer to the layout before the lane 5 folders were split; see [README.md](README.md).

# Lane 5 Report — Per-Class Certification Harness (Build + Benchmark)

> **UPDATE (live, main run in progress):** two findings surfaced from the
> first 2/24 completed seeds — a floor MISMATCH (seed 102) and a per-class
> null-bimodality effect that appears to make the pre-registered per-class
> floor metric land at 0.0cm for every class. See "LIVE FINDINGS" section
> near the top of the addendum below. **STOPPED and reporting per
> protocol rule 6 — not resolved, not silently patched.** The main run
> itself was NOT touched or paused.

## LIVE FINDINGS (main 24-seed run, in progress, 2/24 seeds complete)

**Finding 1 — seed 102 aggregate floor MISMATCH.**
Recomputed floor for seed 102 = **6.50cm**, vs. the certified CSV's
5.5cm for that seed = **MISMATCH**. Seed 101 (the only other completed
seed) matched exactly (9.545454545454545cm both ways). `PREREG.md`
section 1 pre-registered exactly this possibility (fresh eval-time noise
draws even with a matched training seed) and pre-committed to disclosing
rather than resolving it — this is that disclosure. Not yet known whether
this is a one-off or a pattern; will be visible in
`logs/lane5_per_class_results.csv` (`floor_match` column) as more seeds
complete.

**Finding 2 — per-class permutation test appears degenerate at every
depth, for every class, in both completed seeds.**
`per_class_floor_cirrhosis`, `per_class_floor_healthy`, and
`per_class_floor_significant_fibrosis` are all **0.0cm** (the shallowest
possible depth) for seeds 101 and 102 — i.e., the pre-registered
"first FDR-non-significant depth" convention is triggering at the very
first depth tested, for every class, both times.

**Verified this is a real effect, not an implementation bug**, by pulling
the raw saved control-recall arrays for seed 101 at depth=0.00cm (already
persisted per PREREG.md section 5) and histogramming them:

| Class | Controls with recall=0.0 | Controls with recall=1.0 | Mean |
|---|---|---|---|
| cirrhosis | 91/200 (45.5%) | 24/200 (12%) | 0.324 |
| healthy | 87/200 (43.5%) | 27/200 (13.5%) | 0.350 |
| significant_fibrosis | 75/200 (37.5%) | 28/200 (14%) | 0.364 |

This is **exactly the null-bimodality risk `PREREG.md` section 4
pre-registered before any run started** ("a shuffled-label control can
collapse to one predicted class, recall near 0 or 1"), now empirically
confirmed: a large fraction of the 200 shuffled-label control CNNs
degenerate into near-constant-output classifiers (likely because
`train_cnn`'s training loop is only 60 steps of *full-batch* gradient
descent — very few effective updates — on labels that carry no real
signal once shuffled, so many controls converge to predicting whichever
class dominates their random initialization rather than learning
anything). Because ~25-30 of the 200 controls tie the real model's
near-perfect depth-0 recall (1.0) purely by this collapse, the permutation
p-value is large (not significant) even at the easiest depth, for every
class. This is not something this pass's code introduced incorrectly
— it is the honest consequence of applying a one-vs-rest permutation test
to a null distribution that PREREG.md already flagged as likely bimodal;
it was verified against the actual saved data, not assumed.

**Consequence:** as currently defined, the per-class "floor" (first
non-significant depth) is likely to be 0.0cm for most or all seeds/classes
— a fundamentally different, much more pessimistic, and probably
uninformative statistic compared to the aggregate accuracy floor. The
**raw per-class recall values and the saved null-distribution arrays
remain valid and useful** (this is exactly what PREREG.md section 4 asked
to be preserved so power could be judged later) — it is specifically the
*derived floor convention* that appears broken by null degeneracy, not the
underlying data collection.

**Not resolved here, per protocol rule 6.** Options for you to weigh (not
decided in this pass): (a) accept per-class floor=0cm as the honest
answer and report it as such — the aggregate accuracy floor remains the
primary certified metric; (b) redefine the per-class test statistic to be
more robust to control collapse (e.g., requiring recall above chance
rather than a strict `>=` tie-inclusive permutation test) — this would be
a genuine methodology change, not something to apply retroactively without
your sign-off; (c) let the full 24-seed run complete as pre-registered and
treat this as a documented limitation of the per-class metric in the
writeup, using the null histograms already being saved as the evidence.
The main run continues unmodified while you decide — no data already
collected is lost either way.



Scope per the protocol: **build and benchmark, do not launch the full
24-seed run.** Everything below lives under `audit_fixes/lane5/`. Nothing
in `Source/`, `outputs/`, `models/`, `README.md`, or the running
`degradation_aware_resnet_final_sweep.py` (pid 7429) was modified. Threads
were set to 4 everywhere as specified in the protocol.

## Files produced

| File | Purpose |
|---|---|
| `PREREG.md` | Pre-registration, written before any run (plus one mid-run addendum, see below) |
| `lane5_config.py` | New config flags for later sweeps, defaults = original exactly |
| `lane5_harness.py` | The harness itself — imports training/eval-data/FDR functions unchanged |
| `lane5_validate_seed121.py` | Step 2 validation script |
| `lane5_benchmark.py` | Step 3 one-seed benchmark + projections |
| `logs/lane5_benchmark_raw_stdout.txt` | Raw stdout of the benchmark run |
| `logs/lane5_benchmark_result.json` | Benchmark numbers (hand-reconstructed post-crash, see Surprises) |
| `logs/lane5_run.log` | Log file from the benchmark's one seed |
| `seed_weights/degradation_aware_seed101.pt` | Real model weights from the benchmark seed |
| `controls/seed101_depth*.npy` | 69 raw control-recall arrays (23 depths × 3 classes) from the benchmark seed |

## Step 0 — discovery (all paths confirmed by reading, not assumed)

- Certified 24-seed sweep (seeds 101–124): `Source/degradation_aware_final_sweep.py`
  — training: `train_cnn`; data: `build_degradation_aware_dataset`,
  `generate_degraded_windows`; permutation controls: inline
  `np.random.permutation(y_train_encoded)` loop in `run_one_seed`; FDR:
  `bh_fdr_correction`.
- Seed-121 verification: `Source/retrain_seed121_verify.py` — imports the
  above unmodified, saves `Source/models/soma_certified_model_seed121_corrected_noise.pt`.
- Noise-robustness / per-class recall: `Source/noise_robustness_stress_test.py`
  (SNR sweep, inline per-class recall) and
  `Source/degradation_aware_per_class_recall_floor.py` (`eval_one_replicate`,
  the one actually used for the depth=8.48cm check values).
- `degradation_aware_resnet_final_sweep.py` (pid 7429, confirmed running via
  `ps aux`) is a different model family and was not touched or imported for
  logic reuse.
- **Confirmed gap:** no existing script computes per-class recall for the
  200 shuffled-label controls at every seed/depth — this is Lane 5's new
  analysis, built only on the unmodified functions above (see PREREG.md
  section 0).

## Step 1 — PREREG.md

Written before any run. Key disclosed judgment calls (full detail in
`PREREG.md`):
- Per-class BH-FDR family = one (seed, class) pair across all depths.
- "N_REALIZATIONS" split into two flags (`N_TRAIN_REALIZATIONS_PER_CELL=25`,
  `N_EVAL_SAMPLES_PER_CELL=300`) since the original has two independent
  size knobs, not one.
- **STEP (sliding-window stride) is NOT implemented** — `extract_raw_windows`
  has no stride parameter in the original code (one window per detected
  heartbeat); implementing one would mean writing new windowing logic,
  which conflicts with "import unchanged, don't reimplement." **This is an
  open question for you**, not resolved in this pass: drop STEP from
  the sensitivity sweep, or explicitly authorize new windowing code.
- Per-class "recall floor" threshold used the literal **0.45** from your
  spec text (the section header says "40%", the body says "0.45" — flagged
  in case 0.40 was intended).

## Step 2 — seed-121 validation

Ran `lane5_validate_seed121.py` (loads the existing certified weights,
20 replicates × 40/class at depth=8.48cm, identical method to
`degradation_aware_per_class_recall_floor.py`):

| Class | Reproduced | Check value | Result |
|---|---|---|---|
| cirrhosis | 0.1475 | 0.148 | **MATCH** |
| significant_fibrosis | 0.1575 | 0.158 | **MATCH** |
| healthy | 0.9688 | 0.969 | **MATCH** |

**OVERALL: MATCH.** (These numbers also independently reproduce the
existing `Source/logs/degradation_aware_per_class_recall_by_depth.csv`
row for depth=8.48, which was itself generated by
`degradation_aware_per_class_recall_floor.py` in a prior pass — cross-
checked, not assumed.)

## Step 3 — benchmark (one seed, seed 101, 200 controls, 4 threads)

Raw numbers (from `logs/lane5_benchmark_raw_stdout.txt`):

- `ResNet1D sweep (pid 7429) active before: True` / `after: True` —
  **the resnet sweep was running throughout this benchmark**, so these
  timings include whatever CPU contention that caused. Flagged, not
  corrected for.
- Wall-clock for one seed, full pipeline: **815.3s** (train=4.2s,
  200 controls=651.0s, eval sweep across 23 depths=80.1s).
- **Aggregate floor recomputed for seed 101 = 9.545454545454545cm,
  exactly matching the original certified CSV's seed-101 row
  (9.545454545454545cm) to full float precision — MATCH.** This is a
  second, independent confirmation (beyond Step 2) that the harness's
  reimplemented-nothing, imported-everything approach reproduces the
  original protocol exactly on real data, not just in theory.
- Linear-extrapolation projections (24 identical seeds assumed; **caveat:
  linear extrapolation likely UNDERSTATES the dataset-size sweep**, since
  eval cost scales with `N_EVAL_SAMPLES_PER_CELL` and training cost with
  training-set size — neither is free to change without changing
  per-seed wall-clock in a way this single data point can't measure):
  - (a) 24 seeds (the pre-registered run itself): **19,567s ≈ 5.44 hr**
  - (b) dataset-size sweep, 4 sizes × 24 seeds: **78,266s ≈ 21.74 hr**
  - (c) window sweep, 3 settings × 24 seeds: **58,700s ≈ 16.31 hr**
  - None of these were started. Only the single benchmark seed ran.

## Additional requirement (added mid-run)

Per your follow-up: for every seed, the harness now also checkpoints the
full per-depth **aggregate** raw p-value, BH-adjusted p-value, and
significance flag (not just per-class), to
`logs/lane5_aggregate_depth_detail_seed{seed}.csv` — addressing the gap
Lane 2 found (the original sweep discarded every depth's flag except the
first non-significant one). Documented in `PREREG.md` section 8.

- New helper `bh_adjusted_pvalues()` added in `lane5_harness.py` — the
  standard alpha-free BH step-up adjusted-p formula. It does **not**
  reimplement the imported `bh_fdr_correction` (which only returns a
  boolean flag at one fixed alpha and cannot produce an adjusted p-value).
- **Verified, not just asserted:** ran a standalone unit test (10-value
  textbook p-value array) confirming (a) `bh_adjusted_pvalues()` output is
  monotone non-decreasing in p-value order, and (b) `adjusted <= alpha`
  reproduces `bh_fdr_correction`'s own boolean flags exactly, for every
  entry. This is the invariant PREREG.md section 8 requires before
  trusting the new column.
- **This addendum was added after the Step 3 benchmark process had
  already started in the background**, so the benchmark's own weights/
  control arrays do not have a `lane5_aggregate_depth_detail_seed101.csv`
  companion file — the addendum's correctness was instead verified via
  the standalone unit test above rather than by re-running the (already
  completed, ~14-minute) benchmark seed a second time. Flagged as an
  open item: if you want seed 101's own depth-detail file for
  completeness, it requires one more one-seed run (~14 min at the
  benchmarked rate).

## Surprises

1. **`lane5_benchmark.py`'s JSON dump crashed** (`TypeError: Object of
   type bool is not JSON serializable`) because `floor_match` is a
   `numpy.bool_`, not a Python `bool`. All printed numbers (the ones
   quoted above) were captured correctly on stdout before the crash; only
   the JSON-file write failed. Fixed the script (explicit `bool()`/
   `float()` casts) for future runs. `logs/lane5_benchmark_result.json`
   in this report was reconstructed by hand from the raw stdout, not by
   re-running the benchmark — a second run of the same seed was judged
   wasteful (~14 min, same expected result) given the numbers were
   already captured intact. This is disclosed, not silently patched over.
2. **The aggregate floor for seed 101 matched the original CSV to full
   float precision** (not just to 2 decimal places as printed by the
   original script) — a stronger reproduction than strictly required,
   worth noting as positive evidence the harness's re-derivation logic is
   correct.
3. Controls directory has exactly 69 `.npy` files for seed 101 (23 eval
   depths × 3 classes) — matches the expected count exactly, confirming
   no depth was silently skipped during the benchmark's eval sweep.

## Open questions for you

1. **STEP (sliding-window stride)**: not implementable without new
   windowing logic. Drop it from the sensitivity sweep, or authorize a
   documented deviation from "import functions unchanged"?
2. **0.45 vs 40%** in your per-class-floor spec — used the literal 0.45;
   confirm this was intended.
3. **N_REALIZATIONS** split into two flags (train vs. eval dataset size)
   — confirm this decomposition is what you wanted, or specify which one
   (or both, coupled) the size sweep should vary.
4. Do you want seed 101's own `lane5_aggregate_depth_detail_seed101.csv`
   backfilled (requires one more ~14-minute one-seed run), or is the
   standalone unit-test verification sufficient before the full 24-seed
   run is authorized?
5. **Full 24-seed run is NOT started.** At ~815s/seed observed (with the
   resnet sweep contending for CPU throughout), 24 seeds alone project to
   ~5.4 hours; explicit go-ahead needed before launching it, and you may
   want to wait until pid 7429 finishes to get a cleaner, uncontended
   timing baseline first.

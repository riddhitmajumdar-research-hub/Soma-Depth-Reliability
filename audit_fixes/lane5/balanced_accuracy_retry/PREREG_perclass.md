# Lane 5 Pre-Registration — Per-Class Balanced-Accuracy Re-Evaluation

Status: PRE-REGISTERED before any run of `perclass_balacc_eval.py`.
Written as the protocol requires, in response to the null-bimodality
finding in `REPORT.md` ("LIVE FINDINGS", Finding 2): the per-class
one-vs-rest **recall** permutation test collapsed to floor=0.0cm for
every class in both completed seeds, because a large fraction of
shuffled-label controls degenerate into near-constant-output classifiers,
and recall for a constant predictor that happens to output class c is
1.0 for class c (and 0.0 for the other two) — an extreme, easily-tied
statistic. This addendum tries a *different* per-class statistic
specifically chosen to be less sensitive to that degeneracy, evaluated
independently, with the recall-based result **kept, not replaced**, as a
documented negative finding.

## 0. Scope: eval-only, no training

This is **eval-only** on the main run's own already-saved weights (real
model + 200 control models per seed, `audit_fixes/lane5/seed_weights/`).
It trains nothing new. It re-evaluates the exact same trained models the
main run already produced, on a **freshly and deterministically
regenerated** test set per (seed, depth) — deliberately NOT reusing the
main run's own in-memory eval batches (those were never saved to disk;
only derived recall arrays and p-values were), so this is a genuinely
independent re-evaluation, not a replay.

Runs "seeds as they complete" — polls for which seeds have a complete set
of saved weights (real model `.pt` + all 200
`controls_seed{seed}/control{000-199}.pt` files) and processes any
not-yet-processed seed it finds, so it can start now against seeds 101
and 102 and pick up 103+ as the resumed main run produces them. It does
**not** read, import, or depend on anything from `lane5_harness.py`
except passively reading its output *files* — no interaction with that
module's process, consistent with the freeze in effect while the main
run is alive.

## 1. Test-set generation: fixed, deterministic, shared across all 201 models

For each (seed, depth) cell: `np.random.seed(FIXED_EVAL_SEED)` is set to
a value deterministic in (seed, depth) — `900000 + seed*1000 +
depth_index` — immediately before calling the imported
`generate_degraded_windows` (unmodified, from
`degradation_aware_final_sweep.py`), so the exact same
`N_EVAL_SAMPLES_PER_CELL=300` (100/class) test batch is used to evaluate
the real model and all 200 controls at that cell — matching the "ONE test
set" requirement and the main run's own within-cell convention (it also
reused one eval batch across the real model and all 200 controls per
cell; this script's addition is making that test set reproducible
run-to-run via a fixed seed, since the main run's own eval batches were
never saved).

`FIXED_EVAL_SEED` values are disjoint from every training seed (101-124),
every shuffle draw, and the benchmark/validation scripts' own RNG use —
by construction (900000 + ... is far outside the 101-124 range and outside
any other seed value used elsewhere in Lane 5).

## 2. Statistic: per-class one-vs-rest balanced accuracy

For class c, from the 3x3 confusion matrix at that (seed, depth, model):

```
TP = cm[c, c]
FN = sum(cm[c, :]) - TP
FP = sum(cm[:, c]) - TP
TN = total - TP - FN - FP
recall_c (TPR)      = TP / (TP + FN)
specificity_c (TNR) = TN / (TN + FP)
balanced_acc_c      = (recall_c + specificity_c) / 2
precision_c         = TP / (TP + FP)   (nan if TP+FP == 0)
F1_c                = 2 * precision_c * recall_c / (precision_c + recall_c)  (nan if denom == 0)
```

**Stated explicitly, as required: a constant predictor (one that outputs
class c for every input, regardless of the true label — exactly the
degenerate control failure mode found in the recall analysis) scores
balanced accuracy = 0.5 for EVERY class by construction**: for the
predicted class, recall=1.0 but specificity=0.0 (it never predicts
anything else, so every true-negative case is a false positive) →
balanced_acc = 0.5; for each of the other two classes, recall=0.0 but
specificity=1.0 (it never predicts them, so every true negative is
correctly not-predicted) → balanced_acc = 0.5. This is the entire point
of trying this statistic: it caps the damage a degenerate constant-output
control can do to the null distribution at exactly 0.5, rather than
letting it hit the ceiling (1.0) or floor (0.0) that made the recall
statistic's null so often tie a near-perfect real model. **This will be
verified empirically (not just asserted) with a unit test before
`perclass_balacc_eval.py` is trusted, per `REPORT.md`'s standing
convention of verifying claimed invariants.**

## 3. Permutation test, per class, per depth, per seed

`p = (1 + #{controls: balanced_acc_c >= balanced_acc_c(real)}) / 201`,
identical convention to every other permutation test in Lane 5 (main run,
clean sweep). BH-FDR correction **within seed, per class** — same family
definition as the main run's per-class analysis (PREREG.md section 2):
one family = one (seed, class) pair, spanning all depths. First-failed-
depth (first FDR-non-significant depth, ascending) = that class's
balanced-accuracy floor for that seed.

## 4. Also recorded (per the protocol)

- Per-class **recall** and **F1** at every (seed, depth), for the real
  model (controls' recall/F1 are also computed as an input to the
  confusion-matrix-derived stats, and saved, since they're needed anyway
  to build balanced accuracy — no extra cost to keep them).
- Per seed and class: the depth where real-model recall first falls below
  0.45 (same threshold and convention as `PER_CLASS_RECALL_FLOOR_THRESHOLD`
  in `lane5_config.py`, section-3-style computation from `PREREG.md`).
- Per-class recall curves: the full real-model recall-by-depth series is
  saved (this IS the recall curve; no separate plotting is done in this
  eval-only script — plotting, if wanted, is a follow-up over the saved
  CSV).
- **The recall-statistic per-class floor=0.0cm result from the main run
  is NOT recomputed, overwritten, or superseded here.** It remains
  reported in `REPORT.md` as a documented negative result for the recall
  statistic specifically. This script produces a *second, independent*
  per-class floor under a different statistic (balanced accuracy) for
  comparison — both are kept.

## 5. Threads

`torch.set_num_threads(2)`, as the protocol specifies (lower than the main
run's 4, since this runs concurrently with the clean sweep and the
resumed main run on the same machine).

## 6. Output location

- `audit_fixes/lane5/perclass_balacc/logs/perclass_balacc_results.csv`
  (per seed: per-class balanced-accuracy floor, per-class recall-below-0.45
  depth)
- `audit_fixes/lane5/perclass_balacc/logs/perclass_balacc_depth_detail_seed{seed}.csv`
  (per seed, per depth, per class: recall_real, specificity_real,
  balanced_acc_real, F1_real, raw_p_value, bh_adjusted_p_value,
  fdr_significant)
- `audit_fixes/lane5/perclass_balacc/confusion_matrices/seed{seed}_depth{depth}_{real|controls}_confmat.npy`
  (independently regenerated from this script's own fixed-seed eval
  batches — NOT copied from the main run, since the main run never saved
  per-depth confusion matrices; this is this script's own from-scratch
  evaluation)
- Checkpoint/resume: CSV-row presence in `perclass_balacc_results.csv`,
  same convention as every other Lane 5 harness.

## 7. Does not touch the main run

Reads only `audit_fixes/lane5/seed_weights/**/*.pt` (files, not the
process). Does not import `lane5_harness.py` or `lane5_config.py`'s
already-imported module state from that process. Consistent with the
freeze on `lane5_harness.py` / `lane5_config.py` while the main run is
alive (this script does import `lane5_config.py` for shared constants
like `N_SHUFFLES`/`N_EVAL_SAMPLES_PER_CELL`/`FDR_ALPHA`, which is reading
already-frozen, unedited file content — not a live-process interaction).

# Lane6c — FACTS (Q7, Q8, Q9)

LANE=lane6c, read-only on the repo; write only under `audit_fixes/lane6/`.
This folder already contained files from another pass (`FACTS.md`,
`q1_q2_q3_demo.py`, `q1_q2_q3_q4_stdout.log`, `q8_beat_index_accuracy.py`,
`q9_seed121_beat_index_sweep.py` + their stdout logs) — none of those were
modified or deleted. All new output from this pass uses the `_6c` suffix.

No script in this pass retrains, modifies, or touches any certified
model, running process, or repo file. `Source/physics_model.generate_signal`,
`Source/cnn_classifier.{extract_raw_windows,SimpleCNN}`,
`Source/attenuation_model.attenuate_signal` are imported unchanged; every
place this pass needed information those functions don't expose (the
per-window beat index `k`, the raw arrival offset), a parallel wrapper
mirrors the original function's code line-for-line and its own output is
checked (`assert`) against calling the real imported function, so nothing
about the imported functions' behavior is assumed rather than verified.
Per the lane6c amendment, all scripts here run with `torch.set_num_threads(2)`
/ `OMP_NUM_THREADS=2` (the machine is running Lane 5's live 24-seed sweep
plus the long-running ResNet sweep concurrently).

---

## Q7 — offset alone, 20,000 realizations/class

Script: [`q7_offset_bayes.py`](q7_offset_bayes.py), log:
[`q7_offset_bayes_stdout.log`](q7_offset_bayes_stdout.log) (fill in once the
run finishes — see note at the end of this section).

tof (exact, no jitter — jitter only perturbs `heart_rate_hz`, not `kpa`):
- healthy (4.5 kPa): tof = 8.406300 ms
- significant_fibrosis (7.5 kPa): tof = 6.511524 ms (~6.5115ms)
- cirrhosis (12.0 kPa): tof = 5.147779 ms (~5.1478ms)

dt = 1/500 = 2.0000 ms (the sample-rate grid `extract_raw_windows` uses via
`np.searchsorted`).

**k=0 offset == tof exactly**: confirmed by construction — `beat_times[0] ==
0.0` for every realization (`np.arange(0, duration, beat_period)` always
starts at 0), and `time[0] == 0.0` (`np.linspace(0, duration, n_samples)`),
so `start_idx = np.searchsorted(time, 0.0) == 0` and `offset = (0 + tof) -
time[0] = tof - 0 = tof` bit-for-bit, with zero floating-point slack (no
subtraction of two independently-rounded quantities — `time[0]` is exactly
`0.0`, a linspace endpoint, not an approximation). The empirical run (below)
checks `max|offset - tof| == 0.0` across all 20,000×3 realizations rather
than just asserting this from the algebra.

**k>=1 offset == tof - U[0,dt)**: also confirmed by construction —
`beat_times[k]` for k>=1 is (for jittered, i.e. not-exactly-500Hz-aligned,
`heart_rate_hz`) essentially never exactly on the sample grid, so
`start_idx = searchsorted(time, beat_t)` lands at the first grid point `>=
beat_t`, meaning `time[start_idx] - beat_t` is uniform on `[0, dt)`
(standard argument for a fixed-step grid against a continuously-varying
offset), and `offset = (beat_t+tof) - time[start_idx] = tof - (time[start_idx]
- beat_t)`, i.e. `tof - U[0,dt)`, landing in `(tof-dt, tof]`. The empirical
run checks every observed offset falls in that half-open interval and that
the empirical mean matches the U[0,dt) prediction (`tof - dt/2`).

**MATCH.** Empirically, over 20,000 realizations/class (239,734 total
windows, `SEED=42`): k=0 offset equals tof exactly for all 60,000 k=0
windows (max abs error 0.0, float64-exact — not merely "close"); k>=1 offset
lands in the predicted `(tof-dt, tof]` range for all 179,734 k>=1 windows,
with empirical mean matching the U[0,dt) prediction (`tof-dt/2`) to within
~0.002ms.

**Bayes accuracy from offset alone.** Two estimators are in the log, and
they disagree with each other — worth reporting both rather than picking
whichever matches:

1. **Exact analytic decision rule** (uses the generator's *known* exact
   distribution directly — point mass at tof for k=0, uniform on
   `(tof-dt,tof]` for k>=1 — no density estimation/smoothing; ties between
   overlapping-interval classes broken uniformly at random, `random_state=SEED`,
   70/30 stratified split): **overall 90.60%**, k=0 **100%**, k>=1 **87.4-87.6%**
   across beat indices 1-3, per-class recall **healthy 97.91% / significant_fibrosis
   86.06% / cirrhosis 87.83%**.
2. **1D Gaussian-KDE density classifier** (same split, Silverman bandwidth,
   fit-set capped at 4000/class for tractability under host CPU contention):
   overall 86.30%, k=0 100%, k>=1 ~81.7%, per-class healthy 88.83% /
   significant_fibrosis 70.07% / cirrhosis 100%. This estimator is
   noticeably worse and its per-class numbers are skewed toward whichever
   class's k=0 point-mass "spike" the Gaussian kernel smooths most
   favorably — a known failure mode of applying a smooth-kernel density
   estimator to a distribution that is actually a point-mass + sharp-edged
   uniform mixture (the KDE blurs the sharp interval edges the true
   Bayes boundary depends on). **The exact analytic rule (1) is the
   correct one for a "noise-free Bayes accuracy" question** — the
   underlying distribution is exactly known here, so there's no reason to
   estimate it with a general-purpose density estimator, and (2) is kept in
   the log only for transparency about the difference approach makes.

**Comparison to the stated replication (90.5% overall, 100% at k=0, ~87% at
k>=1, recalls 97.9/86.8/86.7 H/F/C): MATCH**, against estimator (1) — all
four headline numbers agree to within ~1 point, consistent with the small
residual randomness from the tie-break RNG in the ~5-32% of each class's
k>=1 mass that falls in an overlap region with a neighboring class (H∩F
overlap is ~5% of H's k>=1 support; F∩C overlap is ~32% of both F's and C's
k>=1 support; H∩C do not overlap at all, `tof_H-dt=6.406ms > tof_C=5.148ms`).
Per-class recall is *not* invariant to the tie-break convention (a
fixed/order-based tie-break instead of a random one pushes all of a tied
region's correct-or-not outcome onto one neighboring class, e.g. an earlier
run of this same exact rule with lowest-class-index tie-breaking gave H
100% / F 96.17% / C 75.94% — same overall 90.71% and same k=0/k>=1
breakdown, but a very different per-class split) — so the replication's
particular H/F/C split is itself evidence of *which* tie-break convention
was used, not an independent fact about the generator.

---

## Q8 — accuracy by beat index k, clean noise-free windows, saved checkpoints ONLY

Per the lane6c amendment: use saved checkpoints only; if none exist for a
model, skip and say so — no retraining this pass (machine busy).

Checked every `.pt`/`.pkl` file in the repo (excluding `.venv` and other
audit_fixes lanes):

| model | checkpoint search result |
|---|---|
| Stage-3 baseline (`source_classifier.py` hand-picked-feature LogisticRegression, clean/non-degradation) | **No saved checkpoint exists.** sklearn `LogisticRegression` objects are never pickled/joblib-dumped anywhere in the repo (`find . -iname "*.pkl"` outside audit_fixes/.venv turns up only Stage-4 sweep *results* pickles — `stage4_*_results.pkl`, `stage4_encoder.pkl` — not a fitted model). `cv_classifier.py`/`source_classifier.py`'s own `__main__` blocks train and print metrics in-process without ever calling `joblib.dump`/`pickle.dump` on the model. **SKIPPED.** |
| Stage-3c `SimpleCNN` on **clean** signals (`cnn_classifier.py`'s own `__main__`, or `floor_characterization_cnn.py`'s `model_real` which is also clean-trained) | **No saved checkpoint exists.** `cnn_classifier.py`'s `__main__` never calls `torch.save`. `floor_characterization_cnn.py` trains a clean CNN as `model_real` but only pickles the depth-sweep *results* (`results/stage4_cnn_results.pkl`), never the model weights. The four `.pt` files under `Source/models/`/`outputs/stage4/models/` (`stage4_cnn_model.pt`, `soma_calibration_corrected_model_weights.pt`, `soma_certified_model_seed121_corrected_noise.pt`, `UNVERIFIED_PRE_FIX1_...pt`, `ets_path2_real_model.pt`) are all `SimpleCNN`-shaped by state-dict key names, but every one of them is a **degradation-aware** (trained across the 5-depth training grid, `TRAIN_DEPTHS_CM=[0,3,6,9,12]`) or calibration/ETS-specific model, not the plain clean-signal Stage-3c model this question asks about — confirmed by reading each producing script (`degradation_aware_cnn.py`, `degradation_aware_calibration_corrected.py`, `retrain_seed121_verify.py`, `ets_montreal_path2_retrain.py`); none of them trains only on clean, non-degraded signals. **SKIPPED** (no matching checkpoint). |
| Stage-3e `ResNet1D` on **clean** signals (`resnet_classifier.py`'s own `__main__`) | **No saved checkpoint exists.** `resnet_classifier.py`'s `__main__` never calls `torch.save` (it only keeps `best_state` in memory for early-stopping restoration, then reports metrics). The `Source/seed_weights_resnet/degradation_aware_resnet_seed*.pt` files are all from `degradation_aware_resnet_final_sweep.py` (degradation-aware, 5-depth training grid) — a different, degradation-aware model, not the clean Stage-3e one. **SKIPPED.** |

**Note on the pre-existing `q8_beat_index_accuracy.py`/`.log` in this
folder** (from the other pass, before the amendment restricting this pass
to checkpoints-only): that script retrained all three models fresh
(`significance_testing.py`'s protocol, grouped 5-fold CV) rather than
loading a saved checkpoint, because none existed — its own header says so.
That is a legitimate, real result (it is real training output, not a
fabricated number), but it was produced under a different mandate (retraining
allowed) than this pass operates under; see
[`q8_beat_index_accuracy_stdout.log`](q8_beat_index_accuracy_stdout.log) for
those numbers if a retrained-model answer is wanted instead of "no checkpoint
exists."

---

## Q9 — seed-121 certified CNN + Lane 5 per-seed CNNs, standard grid, k=0 vs k>=1

Script: [`q9_seed121_beat_index_sweep_6c.py`](q9_seed121_beat_index_sweep_6c.py),
log: [`q9_seed121_beat_index_sweep_6c_stdout.log`](q9_seed121_beat_index_sweep_6c_stdout.log).

**Important correction to the pre-existing `q9_seed121_beat_index_sweep.py`
in this folder** (written by another pass before this pass): that script
answers Q9 using `Source/seed_weights_resnet/degradation_aware_resnet_seed121.pt`
— the **ResNet1D** weights from `degradation_aware_resnet_final_sweep.py`.
The protocol explicitly says *"The ResNet1D sweep
(degradation_aware_resnet_final_sweep.py) is a DIFFERENT job; ignore it
here"* — Q9 asks about the certified **SimpleCNN**
(`outputs/stage4/models/soma_certified_model_seed121_corrected_noise.pt`) and
Lane 5's own **SimpleCNN** per-seed weights
(`audit_fixes/lane5/seed_weights/degradation_aware_seed{seed}.pt`). This
pass (`q9_seed121_beat_index_sweep_6c.py`) uses the correct checkpoints; the
older file's numbers should not be used for Q9. Both files are kept (nothing
was deleted), and this discrepancy is called out explicitly rather than
silently overwritten.

Grid: `EVAL_DEPTHS_CM` (23 depths, 0–15cm, densified 4–7cm) imported
unchanged from `degradation_aware_final_sweep.py`; `FREQ=100Hz`,
`SENSOR_SNR_DB=27.05dB` (`constants.py`), `N_SAMPLES_PER_CELL=300` (matches
the sweep's own per-cell sample count). `generate_degraded_windows` is
mirrored line-for-line with k-tagging added, and its window content is
checked against the real `extract_raw_windows()` on every call.

Lane 5 seeds available at run time (checked live against
`audit_fixes/lane5/logs/lane5_run.log` + file existence, not assumed): only
**seed101 and seed102** have both a "Seed N done" log line and a
`degradation_aware_seed{N}.pt` file — seed103 was still training (no `.pt`
yet) at run time, consistent with `lane5_run.log`'s last line ("--- Seed 103
starting ---"). Per the amendment, this does NOT wait for more seeds; it
reports only what's complete now.

**Pooled accuracy, k=0 vs k>=1, full 23-depth grid (n≈1730 / ≈5170 per
model):**

| checkpoint | k=0 n | k=0 acc | k>=1 n | k>=1 acc |
|---|---|---|---|---|
| seed121 (certified, `outputs/stage4/models/soma_certified_model_seed121_corrected_noise.pt`) | 1731 | 71.52% | 5169 | 67.50% |
| lane5 seed101 | 1732 | 71.13% | 5168 | 67.84% |
| lane5 seed102 | 1733 | 56.84% | 5167 | 60.03% |

Full per-depth breakdown for all three checkpoints is in the stdout log
linked above. Directionally consistent across seed121 and seed101: k=0
(the beat whose window always starts exactly at the true arrival point, no
sub-sample-grid slack) is *more* accurate than k>=1 by ~3-4 points pooled
across the whole depth grid — but seed102 inverts this (k>=1 slightly
*higher* than k=0), so this is not a uniform effect across seeds/checkpoints
within just these three data points, and a firmer statement would need the
remaining seeds once Lane 5 finishes.

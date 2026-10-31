# 08. Lane 6: how the window is anchored to the beat

The pre-existing Q8/Q9 scripts came from a separate pass; the "6c" set (`FACTS_6c.md`) redid them under a checkpoints-only rule.

## Question

Which feature of the synthetic windows can a classifier use without any signal shape at all?

## Method and findings

`Source/physics_model.generate_signal` evaluates the pulse on a continuous time axis; `cnn_classifier.extract_raw_windows` starts each window at the first sample on or after the beat time. Q7 (`q7_offset_bayes.py`, 20000 realizations per class, 239,734 windows):

- For beat index k=0 the offset between window start and pulse arrival equals the class time of flight exactly (max error 0.0). TOF: healthy 8.406 ms, fibrosis 6.512, cirrhosis 5.148.
- For k>=1 the offset is TOF minus a uniform draw on [0, dt), dt about 2 ms (mean matched to 0.002 ms).
- An exact analytic decision rule on the offset alone reaches 90.60% overall (k=0: 100%; k>=1: 87.4 to 87.6%). A Gaussian-KDE estimate of the same gets 86.3% and is worse because it smooths a point mass and a sharp-edged uniform. Per-class recall depends on the tie-break rule for the overlapping intervals (random tie-break: H 97.9 / F 86.1 / C 87.8; lowest-index: H 100 / F 96.2 / C 75.9).

Q8 (retrained Stage 3 models, clean windows, by beat index): baseline 71.59% overall with 100% at k=0 and about 61% at k>=1; CNN 88.33%; ResNet1D 91.11% with 100% at k=0. The 6c pass found no saved clean Stage 3 checkpoints (none of the four `.pt` files is a clean-trained Stage 3 model), so its own Q8 was skipped and the retrained numbers stay as the earlier pass's.

Q9 (degraded grid, 100 Hz, 27.05 dB, 23 depths): the k=0 advantage nearly disappears. The 6c run on the correct CNN checkpoints gives k=0 vs k>=1 pooled accuracy of 71.52 vs 67.50 (seed 121), 71.13 vs 67.84 (Lane 5 seed 101), 56.84 vs 60.03 (seed 102). The older Q9 script used the ResNet1D seed-121 weights (51.33 vs 48.88), which was the wrong model for the question. Both files were kept and the mix-up is recorded in `FACTS_6c.md`.

Q1 to Q4 and Q5 are code readings: `kpa` is fixed per class in every generating loop, and within-class variation is heart-rate jitter and noise only; the power-law exponent 0.51 is applied to alpha, anchored at 20 Hz.

## Limitations

Q9 covers seeds 121, 101, 102 only. The 24-seed version was deferred. Within-class stiffness variation was tested later in the battery (12, conditions 4a/4b); floors there were unreliable.

## Reproducibility

Scripts and stdout logs under `audit_fixes/lane6/`. Q7 took about 25 minutes under CPU contention.

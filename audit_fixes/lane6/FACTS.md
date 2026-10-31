# Lane6b — FACTS

Read-only lane. Files read (none modified): `Source/physics_model.py`,
`Source/cnn_classifier.py`, `Source/repeated_grouped_cv_log11.py`,
`Source/attenuation_model_powerlaw.py`, `Source/floor_characterization_cnn.py`,
`Source/noise_robustness_stress_test_clean_cnn.py`, `Source/constants.py`.
Demonstration script: [`q1_q2_q3_demo.py`](q1_q2_q3_demo.py); raw stdout saved
at [`q1_q2_q3_q4_stdout.log`](q1_q2_q3_q4_stdout.log).

---

## Q1 — sample-index rounding vs. continuous time axis

**`Source/physics_model.py:33`**
```python
time = np.linspace(0, duration, n_samples)
```
**`Source/physics_model.py:37, 40-42`**
```python
arrival_t = beat_t + tof     # pulse arrives tof seconds after the heartbeat
...
mask = (time >= arrival_t) & (time <= arrival_t + pulse_window)
local_t = time[mask] - arrival_t
pulse_shape = np.exp(-15 * local_t) * np.sin(2 * np.pi * 100 * local_t)
```

**Answer: the carrier is evaluated on a continuous time axis at `t-(beat_t+tof)`, not rounded to a sample index.** `time` is a `np.linspace` continuous axis; `local_t` is computed by subtracting the exact (non-quantized) `arrival_t` from that axis, and the sinusoid/envelope (`pulse_shape`) is evaluated directly on `local_t`. The only place a "grid" appears is the boolean `mask` selecting which of the pre-existing 500Hz-spaced samples fall inside the pulse window — that's an existing-sample-rate selection, not a rounding of the arrival time itself.

**Demonstration** (all three classes, default `heart_rate_hz=1.15`, so `beat_times` — and hence `beat_t`, confirmed identical across classes since `beat_times = np.arange(0, duration, 1/heart_rate_hz)` depends only on `heart_rate_hz`, not on `kpa`):

| class | kPa | tof (ms) | first \|x\|>5%peak: sample_idx | first \|x\|>5%peak: t (ms) |
|---|---|---|---|---|
| healthy | 4.5 | 8.4063 | 5 | 10.0067 |
| significant_fibrosis | 7.5 | 6.5115 | 4 | 8.0053 |
| cirrhosis | 12.0 | 5.1478 | 3 | 6.0040 |

Sub-sample lag (parabolic-interpolated cross-correlation, 3-point quadratic fit around the discrete `np.correlate(..., mode="full")` peak):

- healthy vs significant_fibrosis: discrete peak lag = -1 sample, sub-sample correction = +0.0456 samples → total lag = **-0.9544 samples = -1.9088 ms** (raw tof difference = -1.8948 ms)
- healthy vs cirrhosis: discrete peak lag = -2 samples, sub-sample correction = +0.3467 samples → total lag = **-1.6533 samples = -3.3065 ms** (raw tof difference = -3.2585 ms)

The interpolated xcorr lag tracks the raw tof difference to within ~0.01–0.05 ms, consistent with the carrier being on a continuous (non-quantized) time axis — a purely sample-rounded model would produce lags quantized to multiples of 1/500s = 2ms, which is not what's observed (sub-sample corrections of 0.0456 and 0.3467 samples are non-trivial, non-integer).

---

## Q2 — `cnn_classifier.extract_raw_windows`

**`Source/cnn_classifier.py:35-44`**
```python
beat_period = 1.0 / heart_rate_hz
beat_times = np.arange(0, duration, beat_period)
windows = []

for beat_t in beat_times:
    start_idx = np.searchsorted(time, beat_t)
    end_idx = start_idx + window_size
    if end_idx > len(signal):
        continue
    windows.append(signal[start_idx:end_idx])
```

- **Window start offset from `beat_t`: 0 samples.** `start_idx = np.searchsorted(time, beat_t)` starts exactly at (the sample nearest to) `beat_t` itself — no additional offset is added.
- **Window length:** `window_size` default = **100 samples = 200.0 ms** at the 500Hz sample rate used by `generate_signal`.
- **Windows per 3s at default heart rate (1.15Hz) with jitter (`jitter_std=0.03`):** measured over 2000 sampled realizations — **min=3, max=4, mean=3.9965** (i.e. almost always 4, occasionally 3 when jitter pushes the last beat's window past the 3s buffer).
- **Overlap: NO.** beat_period at 1.15Hz = 0.8696s ≫ window duration 0.2000s. Checked at a +4σ jittered heart rate (1.27Hz, jitter_std=0.03): beat_period = 0.7874s, still > 0.2000s → no overlap across the sampled jitter range.

---

## Q3 — Stage 3 dataset totals & per-fold sizes at `rep_seed=1000`

Using `cnn_classifier.build_raw_dataset_with_groups()` (as called by `repeated_grouped_cv_log11.main()`), with the repo's own `SEED=42` (`Source/constants.py`), defaults `n_runs_per_class=30`, `jitter_std=0.03`:

- **Total windows: 360**
- **Total realizations (groups): 90** (30 runs/class × 3 classes)
- **Per-realization windows: min=4, max=4, mean=4.0000** — with this exact seed, every one of the 90 realizations produced exactly 4 windows (matches the "mean≈4" behavior measured in the unseeded jitter sweep above, but is not stochastic here — SEED=42 with this exact call happens to land all 90 draws in the 4-window regime).

**Fold sizes for `rep_seed=1000`** (`repeated_grouped_cv_log11.py:139`, `rep_seed = 1000 + rep` at `rep=0`), via `grouped_kfold_indices(groups_raw, N_FOLDS=5, rep_seed=1000)`:

| fold | train windows | test windows | test groups |
|---|---|---|---|
| 1 | 288 | 72 | 18 |
| 2 | 288 | 72 | 18 |
| 3 | 288 | 72 | 18 |
| 4 | 288 | 72 | 18 |
| 5 | 288 | 72 | 18 |

(90 groups / 5 folds = 18 groups/fold exactly, each realization contributing exactly 4 windows at this seed → 18×4=72 test windows/fold, matching the table.)

---

## Q4 — what varies across realizations of the same class?

**Stage 3, `Source/cnn_classifier.py:81-90`:**
```python
for name, kpa in stiffness_classes.items():
    for _ in range(n_runs_per_class):
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time, signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        windows = extract_raw_windows(signal, time, jittered_hr)
```

**Stage 4 (`floor_characterization_cnn.py:48-52`):**
```python
def generate_degraded_windows(kpa, depth_m, frequency_hz, n_needed, jitter_std=0.03):
    while len(windows) < n_needed:
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
```

**Stage 4 (`noise_robustness_stress_test_clean_cnn.py:81-89`):**
```python
def generate_windows(kpa, depth_m, frequency_hz, n_needed, noise_std, jitter_std=0.03):
    while len(windows) < n_needed:
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)
        noisy = attenuated + np.random.normal(0, noise_std, size=attenuated.shape)
```

**Answer: `kpa` is never sampled or jittered within a class in any of the three generating loops inspected.** In every case `kpa` is the fixed class value from `stiffness_classes` (Stage 3) or a fixed function argument passed through unchanged (Stage 4), used identically across all realizations of that class. The only sources of per-realization variation are: (1) heart-rate jitter, `jittered_hr = 1.15 + np.random.normal(0, jitter_std)`, in all three loops; and (2) additive Gaussian sensor noise (`np.random.normal(0, noise_std/SENSOR_NOISE_STD, ...)`) in the Stage 4 degradation loops only.

---

## Q5 — power-law attenuation exponent

**`Source/attenuation_model_powerlaw.py:24-25, 36-38`**
```python
ANCHOR_FREQ_HZ = 20
POWER_LAW_EXPONENT = 0.51
...
def attenuation_coefficient_powerlaw(youngs_modulus_kpa, frequency_hz,
                                      eta=None,
                                      n=POWER_LAW_EXPONENT,
                                      anchor_freq=ANCHOR_FREQ_HZ):
    """Power-law attenuation, anchored to match Kelvin-Voigt exactly at anchor_freq."""
    alpha_anchor = attenuation_coefficient(youngs_modulus_kpa, anchor_freq, eta=eta)
    alpha = alpha_anchor * (frequency_hz / anchor_freq) ** n
    return alpha
```

**Answer: `n=0.51` is used as `alpha ~ f^n`** — the attenuation coefficient (not phase velocity `c`) scales with frequency raised to the 0.51 power, anchored so it equals the Kelvin-Voigt-derived `alpha_anchor` exactly at the anchor frequency. **Anchor frequency: 20 Hz** (`ANCHOR_FREQ_HZ = 20`, module docstring lines 15-19 explains this is deliberately SOMA's non-operating frequency, chosen because the Kelvin-Voigt weak-damping assumption holds best there).

No `c ~ f^n` (phase-velocity power law) form was found anywhere in this file or `attenuation_model.py`.

---

# Lane6c — Q7, Q8, Q9

Read-only lane. Additional files read (none modified): `Source/significance_testing.py`,
`Source/source_classifier.py`, `Source/resnet_classifier.py`, `Source/degradation_aware_final_sweep.py`,
`Source/degradation_aware_resnet_final_sweep.py`, `Source/attenuation_model.py`,
`Source/logs/degradation_aware_resnet_final_sweep_log.txt`, `Source/logs/degradation_aware_resnet_final_results.csv`,
`Source/seed_weights_resnet/` (listing only), `Source/models/` (listing only). No repo file modified; the running
sweep (pid 7429) and its outputs were not touched (only read).

Scripts: [`q7_offset_bayes.py`](q7_offset_bayes.py), [`q8_beat_index_accuracy.py`](q8_beat_index_accuracy.py),
[`q9_seed121_beat_index_sweep.py`](q9_seed121_beat_index_sweep.py). Raw stdout: `q7_offset_bayes_stdout.log`,
[`q8_beat_index_accuracy_stdout.log`](q8_beat_index_accuracy_stdout.log), [`q9_seed121_beat_index_sweep_stdout.log`](q9_seed121_beat_index_sweep_stdout.log).

## Pre-flight facts (read-only recon, before writing any code)

- **Lane 5's main 24-seed ResNet1D sweep (pid 7429, `Source/degradation_aware_resnet_final_sweep.py`) is still running** — confirmed via `ps -p 7429`, elapsed >2 days. `Source/logs/degradation_aware_resnet_final_results.csv` has completed rows only for seeds 101–120; `Source/logs/degradation_aware_resnet_final_sweep_log.txt` ends mid-seed-121 (`--- Seed 121 starting ---`, no completion line after it).
- **Seed-121 weights already exist**, though: `Source/seed_weights_resnet/degradation_aware_resnet_seed121.pt` (`torch.save` happens in `run_one_seed()` right after training, at `Source/degradation_aware_resnet_final_sweep.py:171-172`, *before* the much slower per-depth evaluation loop that produces the CSV row — so the weights file is already final and was not being written to when read here).
- **Seeds 122–124 do not exist yet.** Q9's "repeat for all seeds once Lane 5 finishes" is therefore only done for seed 121 below; the rest is explicitly pending.
- **No saved Stage-3 (non-degradation) checkpoints exist.** `Source/models/` has exactly 4 files, all Stage-4/calibration (`ets_path2_real_model.pt`, `soma_calibration_corrected_model_weights.pt`, `soma_certified_model_seed121_corrected_noise.pt`, `stage4_cnn_model.pt`) — none is a plain Stage-3 CNN/ResNet1D/baseline checkpoint. Q8 therefore trains fresh, using the identical protocol/hyperparameters in `Source/significance_testing.py` (copied verbatim into the wrapper script rather than importing that module directly, since it executes top-level training + `plt.show()`/`plt.savefig()` on import with no `if __name__=="__main__"` guard — importing it would have had side effects outside this lane).

## Q7 — offset demonstration & noise-free Bayes accuracy from offset alone

20,000 realizations/class (60,000 total) via the actual `physics_model.generate_signal`, wrapped only to additionally expose `k` and `offset = arrival_t - time[start_idx]` (mirrors `cnn_classifier.extract_raw_windows`, `Source/cnn_classifier.py:35-44`, line-for-line). Ran under heavy system contention (load average 9.5 on a 10-core machine, shared with pid 7429 and an unrelated `audit_fixes/lane5/lane5_harness.py` process) — took ~25 min wall time; results below are unaffected by that, only the wait was.

**Total windows collected: 239,734** (target 240,000 = 3×20,000×4 beats; shortfall from occasional truncated last-beat windows near t=3s under jitter, same truncation `extract_raw_windows` itself applies).

`tof` per class (deterministic): healthy=8.4063ms, significant_fibrosis=6.5115ms, cirrhosis=5.1478ms. Sample spacing `dt = time[1]-time[0] = 2.001334ms` (the `np.linspace(0,3,1500)` grid, not exactly the nominal 1/500Hz=2.000ms).

**k=0 check — offset should equal tof exactly:**

| class | n | max\|offset − tof\| | result |
|---|---|---|---|
| healthy | 20,000 | 0.000e+00s | **MATCH** |
| significant_fibrosis | 20,000 | 0.000e+00s | **MATCH** |
| cirrhosis | 20,000 | 0.000e+00s | **MATCH** |

Confirmed exactly (to float64 precision) — k=0's offset is not merely close to tof, it is bit-identical to it, consistent with the Q1 finding (`beat_t=0` lands exactly on the `time` grid's first sample for every realization, so `time[start_idx]=0` exactly and `offset = beat_t+tof-time[start_idx] = tof`).

**k≥1 check — offset should be `tof - U[0, dt)`:**

| class | n | frac=tof−offset: min | max | mean | expected mean (dt/2) | result |
|---|---|---|---|---|---|---|
| healthy | 59,904 | 0.00011ms | 2.00130ms | 1.00066ms | 1.00067ms | **MATCH** |
| significant_fibrosis | 59,919 | 0.00010ms | 2.00120ms | 1.00189ms | 1.00067ms | **MATCH** |
| cirrhosis | 59,911 | 0.00006ms | 2.00130ms | 0.99690ms | 1.00067ms | **MATCH** |

Confirmed: for k≥1, `frac = tof - offset` spans very close to the full `[0, dt)` range with mean within ~0.1-0.2% of the theoretical `dt/2` — consistent with `frac` being (near-)uniform on `[0, dt)`, exactly as hypothesized.

**Noise-free Bayes accuracy from offset alone** (1D KDE density classifier, Gaussian kernel, Silverman's-rule bandwidth=1.4257e-04s fit on pooled training offsets, equal class priors; stratified 70/30 train/test split, seed=42 from `constants.py`, n_train=167,813 / n_test=71,921):

- **Overall: 86.36%**
- Per class: healthy 88.99% (n=23,971), significant_fibrosis 70.11% (n=23,976), **cirrhosis 100.00%** (n=23,974)
- Per k: **k=0: 100.00%** (n=18,057), k=1: 81.82% (n=18,041), k=2: 81.82% (n=18,030), k=3: 81.73% (n=17,793)

**Interpretation, stated plainly:** the offset feature alone — a single scalar, no signal shape at all — already gives a Bayes-optimal classifier 86.36% accuracy on clean data, with **perfect (100%) accuracy on k=0 windows and on the cirrhosis class**. This is the mechanism behind Q8's finding: k=0 is deterministically separable (offset≡tof, and the three tof values are distinct constants), while k≥1's uniform sub-sample jitter (spanning ~2ms, comparable to the ~1.9-3.3ms gaps between the three classes' tof values) is what actually causes the residual ~18% error at k≥1 — some fraction of the `[0,dt)` jitter range pushes a window's offset into another class's zone. Q8's clean-signal CNN/ResNet1D/baseline results (k=0 dramatically higher accuracy than k≥1, see below) are consistent with the models having partly learned to exploit exactly this offset cue, though this script does not prove that causal claim — it only establishes that the *offset alone* carries this much class information.

## Q8 — accuracy by beat index k, clean noise-free windows, Stage-3 models

Training protocol copied verbatim from `Source/significance_testing.py` (5-fold `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, `SimpleCNN` 60 epochs, `ResNet1D` with the same restart-on-collapse logic, `LogisticRegression(max_iter=1000)` baseline on `source_classifier.build_dataset()` features) — datasets rebuilt with a `k`-tagging wrapper (`build_baseline_dataset_with_k`, `build_raw_dataset_with_k`) that mirrors the original generation loops exactly (`Source/source_classifier.py:57-72`, `Source/cnn_classifier.py:49-62`) but also records the beat index of each row/window. One collapse-and-restart occurred at fold 2 for ResNet1D (train_acc=33.48% on attempt 1, escaped on attempt 2) — same safeguard as the certified pipeline, not something this script introduced.

**Baseline (LogisticRegression), 5-fold pooled test predictions, n=359:**
- Overall accuracy: **71.59%**
- By class: cirrhosis 78.15% (n=119), healthy 81.67% (n=120), significant_fibrosis 55.00% (n=120)
- By beat index k: **k=0: 100.00%** (n=90), k=1: 61.11% (n=90), k=2: 64.44% (n=90), k=3: 60.67% (n=89)

**SimpleCNN, 5-fold pooled test predictions, n=360:**
- Overall accuracy: **88.33%**
- By class: cirrhosis 95.00%, healthy 91.67%, significant_fibrosis 78.33% (n=120 each)
- By beat index k: k=0: 93.33%, k=1: 83.33%, k=2: 88.89%, k=3: 87.78% (n=90 each)

**ResNet1D, 5-fold pooled test predictions, n=360:**
- Overall accuracy: **91.11%**
- By class: cirrhosis 100.00%, healthy 96.67%, significant_fibrosis 76.67% (n=120 each)
- By beat index k: k=0: 100.00%, k=1: 85.56%, k=2: 90.00%, k=3: 88.89% (n=90 each)

**Notable pattern, consistent across all three models: k=0 is the easiest beat index by a wide margin (100% for baseline and ResNet1D), and k≥1 is harder and roughly flat across k=1/2/3.** This is directly explained by Q7's mechanism once it lands: k=0's offset equals `tof` exactly (perfectly class-separating in principle), while k≥1's offset is `tof` minus a near-uniform sub-sample jitter, which is a strictly noisier version of the same signal.

## Q9 — seed-121 ResNet1D, standard grid (100Hz, 27.05dB), accuracy by k=0 vs k≥1

Checkpoint: `Source/seed_weights_resnet/degradation_aware_resnet_seed121.pt` (read-only). Grid: `EVAL_DEPTHS_CM` (23 depths) and `FREQ=100` imported unchanged from `Source/degradation_aware_final_sweep.py`; `SENSOR_SNR_DB=27.05` from `Source/constants.py`; `N_SAMPLES_PER_CELL=300` matching `Source/degradation_aware_resnet_final_sweep.py:98` exactly. `generate_degraded_windows` doesn't expose k, so a parallel wrapper (`generate_degraded_windows_with_k`) mirrors its exact code (`Source/degradation_aware_final_sweep.py:47-56`) plus tags k.

Per-depth results (n=75-77 for k=0, n=223-225 for k≥1 per depth — see full table in `q9_seed121_beat_index_sweep_stdout.log`):

| depth (cm) | k=0 acc | k≥1 acc | | depth (cm) | k=0 acc | k≥1 acc |
|---|---|---|---|---|---|---|
| 0.00 | 93.33% | 73.78% | | 6.50 | 49.33% | 54.67% |
| 1.36 | 73.33% | 62.22% | | 6.82 | 45.33% | 48.89% |
| 2.73 | 64.47% | 57.59% | | 7.00 | 50.67% | 51.11% |
| 3.50 | 52.63% | 54.46% | | 7.50 | 38.67% | 44.00% |
| 4.00 | 52.00% | 52.89% | | 8.00 | 46.05% | 45.98% |
| 4.09 | 58.67% | 51.11% | | 8.18 | 52.00% | 40.00% |
| 4.30 | 54.55% | 53.36% | | 9.55 | 30.67% | 40.44% |
| 4.50 | 58.67% | 54.67% | | 10.91 | 42.67% | 39.11% |
| 5.00 | 61.33% | 50.67% | | 12.27 | 34.67% | 33.33% |
| 5.45 | 46.67% | 55.11% | | 13.64 | 34.67% | 32.44% |
| 5.50 | 46.67% | 49.33% | | 15.00 | 29.33% | 32.00% |
| 6.00 | 64.00% | 47.11% | | | | |

**Pooled across all 23 depths:**
- **k=0: n=1730, accuracy=51.33%**
- **k≥1: n=5170, accuracy=48.88%**
- total n=6900

**Observation, stated plainly and not resolved further here:** unlike Q8's clean-signal result (where k=0 was dramatically easier, 100% vs ~85-90%), the k=0 advantage **shrinks to near parity once degradation (depth attenuation + sensor noise) is applied** — pooled 51.33% vs 48.88%, and the per-depth advantage is inconsistent (k=0 is sometimes higher, sometimes lower than k≥1, e.g. at 9.55cm k=0 is *worse*: 30.67% vs 40.44%). This is consistent with degradation (attenuation + additive sensor noise) swamping the k=0 offset-exactness advantage that dominates in the clean-signal case, but this script does not attempt to prove that causally — it only reports what was measured.

**Pending:** seeds 122–124 do not exist yet (Lane 5's sweep, pid 7429, has not finished — only seeds 101–120 are certified-complete, seed 121 just finished its weights but not necessarily its full per-seed CSV row at the time of this reading). Repeating this analysis across all 24 seeds is explicitly deferred until that run completes.

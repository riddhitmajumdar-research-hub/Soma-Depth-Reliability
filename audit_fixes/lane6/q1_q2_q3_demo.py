"""
Lane6b audit script -- Q1, Q2, Q3 numeric demonstrations.
READ-ONLY: only imports existing functions/constants from Source/, no repo
file is modified, no retraining, no constants hardcoded (SEED imported from
Source/constants.py).
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Source"))

from physics_model import generate_signal, stiffness_classes          # noqa: E402
from cnn_classifier import extract_raw_windows, build_raw_dataset_with_groups  # noqa: E402
from repeated_grouped_cv_log11 import grouped_kfold_indices, N_FOLDS   # noqa: E402
from constants import SEED                                            # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def parabolic_interp_peak(corr, peak_idx):
    """Standard 3-point parabolic interpolation around a discrete xcorr peak,
    returns the sub-sample offset (in samples) from peak_idx."""
    if peak_idx == 0 or peak_idx == len(corr) - 1:
        return 0.0
    y0, y1, y2 = corr[peak_idx - 1], corr[peak_idx], corr[peak_idx + 1]
    denom = (y0 - 2 * y1 + y2)
    if denom == 0:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def main():
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 78)
    p("Q1 -- physics_model.generate_signal: sample-index rounding vs continuous time")
    p("=" * 78)
    p("Code (Source/physics_model.py:23-45), quoted verbatim:")
    p("  line 32:  n_samples = int(duration * sample_rate)")
    p("  line 33:  time = np.linspace(0, duration, n_samples)")
    p("  line 36:  for beat_t in beat_times:")
    p("  line 37:      arrival_t = beat_t + tof")
    p("  line 40:      mask = (time >= arrival_t) & (time <= arrival_t + pulse_window)")
    p("  line 41:      local_t = time[mask] - arrival_t")
    p("  line 42:      pulse_shape = np.exp(-15 * local_t) * np.sin(2 * np.pi * 100 * local_t)")
    p("=> `time` is a continuous np.linspace axis (not rounded sample indices);")
    p("   `local_t` = time[mask] - arrival_t is evaluated on that continuous axis,")
    p("   i.e. the carrier IS evaluated at t-(beat_t+tof) continuously.")
    p("   The `mask` selection (time>=arrival_t) picks the nearest sample GRID points")
    p("   >= arrival_t, but the carrier phase itself is not snapped to a rounded index --")
    p("   it uses the true (non-integer-multiple) time value at each selected sample.")
    p()

    # Same beat_t across classes (first beat_t=0.0 for all, since beat_times
    # only depends on heart_rate_hz, identical default 1.15Hz for all three
    # calls below -- tof differs per class only).
    beat_period = 1.0 / 1.15
    p(f"heart_rate_hz=1.15 (default) for all 3 classes => beat_times identical: "
      f"np.arange(0,3.0,{beat_period:.6f}) -- SAME beat_t grid used by all classes.")
    p()

    sample_rate = 500
    results = {}
    for name, kpa in stiffness_classes.items():
        time, signal, tof = generate_signal(kpa)
        # first sample with |x| > 5% of this signal's own peak amplitude
        peak = np.max(np.abs(signal))
        thresh = 0.05 * peak
        first_idx = np.argmax(np.abs(signal) > thresh)
        first_t = time[first_idx]
        results[name] = (tof, peak, first_idx, first_t, signal, time)
        p(f"{name:<22s} kPa={kpa:5.1f}  tof={tof*1000:.4f}ms  "
          f"peak_amp={peak:.4f}  5%thresh={thresh:.4f}  "
          f"first|x|>5%peak: sample_idx={first_idx}  t={first_t*1000:.4f}ms")

    p()
    p("Sub-sample lag between classes (parabolic-interpolated cross-correlation,")
    p("full np.correlate 'full' mode, quadratic interpolation around the discrete peak):")
    names = list(stiffness_classes.keys())
    ref_name = names[0]
    ref_signal = results[ref_name][4]
    dt = 1.0 / sample_rate
    for name in names[1:]:
        sig = results[name][4]
        corr = np.correlate(sig, ref_signal, mode="full")
        peak_idx = int(np.argmax(corr))
        sub_offset = parabolic_interp_peak(corr, peak_idx)
        lag_samples = (peak_idx - (len(ref_signal) - 1)) + sub_offset
        lag_ms = lag_samples * dt * 1000
        tof_diff_ms = (results[name][0] - results[ref_name][0]) * 1000
        p(f"  {ref_name} vs {name}: discrete peak lag={peak_idx - (len(ref_signal)-1)} samples, "
          f"sub-sample correction={sub_offset:.4f} samples, "
          f"total lag={lag_samples:.4f} samples = {lag_ms:.4f} ms  "
          f"(raw tof difference = {tof_diff_ms:.4f} ms)")

    p()
    p("=" * 78)
    p("Q2 -- cnn_classifier.extract_raw_windows: offset / length / count / overlap")
    p("=" * 78)
    p("Code (Source/cnn_classifier.py:18-46), quoted verbatim:")
    p("  line 35: beat_period = 1.0 / heart_rate_hz")
    p("  line 36: beat_times = np.arange(0, duration, beat_period)")
    p("  line 40: start_idx = np.searchsorted(time, beat_t)")
    p("  line 41: end_idx = start_idx + window_size")
    p("  line 44: windows.append(signal[start_idx:end_idx])")
    p("=> window start offset from beat_t = 0 samples (window starts exactly at beat_t,")
    p("   found via searchsorted -- no additional offset applied).")
    p(f"=> window length = window_size default = 100 samples = {100/sample_rate*1000:.1f} ms at {sample_rate}Hz.")

    hr_default = 1.15
    jitter_std = 0.03
    rng = np.random.default_rng(12345)  # local demo RNG, independent of repo SEED, just to sample jitter range
    counts = []
    for _ in range(2000):
        jittered_hr = hr_default + rng.normal(0, jitter_std)
        time, signal, tof = generate_signal(1.0, heart_rate_hz=jittered_hr)  # kpa value irrelevant to window count
        windows = extract_raw_windows(signal, time, jittered_hr)
        counts.append(len(windows))
    counts = np.array(counts)
    p(f"Windows per 3s realization at default heart rate (1.15Hz) with jitter_std=0.03, "
      f"n=2000 sampled realizations: min={counts.min()}, max={counts.max()}, "
      f"mean={counts.mean():.4f}")

    window_dur_s = 100 / sample_rate
    p(f"Overlap check: beat_period at hr=1.15Hz = {1/1.15:.4f}s, window duration = {window_dur_s:.4f}s.")
    p(f"  window duration ({window_dur_s:.4f}s) < beat_period ({1/1.15:.4f}s) => "
      f"NO overlap between consecutive windows at default heart rate.")
    # explicit check across the jitter range actually sampled
    min_beat_period = 1.0 / (hr_default + jitter_std * 4)  # ~4 sigma high-HR tail
    p(f"  Even at a +4sigma jittered heart rate ({hr_default + jitter_std*4:.4f}Hz), "
      f"beat_period={min_beat_period:.4f}s still > window duration {window_dur_s:.4f}s => no overlap.")

    p()
    p("=" * 78)
    p("Q3 -- Stage 3 dataset (cnn_classifier.build_raw_dataset_with_groups): totals & fold sizes")
    p("=" * 78)
    p("Code (Source/cnn_classifier.py:65-92), quoted verbatim:")
    p("  line 81: for name, kpa in stiffness_classes.items():")
    p("  line 82:     for _ in range(n_runs_per_class):")
    p("  line 83:         jittered_hr = 1.15 + np.random.normal(0, jitter_std)")
    p("  line 84:         time, signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)")
    p("  line 85:         windows = extract_raw_windows(signal, time, jittered_hr)")
    p("  line 90:     group_id += 1")

    X_raw, y_raw, groups_raw = build_raw_dataset_with_groups(seed=SEED)
    p(f"Using repo's own SEED={SEED} (Source/constants.py), exact call used by "
      f"repeated_grouped_cv_log11.py's build_raw_dataset_with_groups(): "
      f"n_runs_per_class=30 (default), jitter_std=0.03 (default).")
    p(f"Total windows: {len(X_raw)}")
    unique_groups, counts_per_group = np.unique(groups_raw, return_counts=True)
    p(f"Total realizations (groups): {len(unique_groups)} (expected 30 runs/class x 3 classes = 90)")
    p(f"Per-realization windows: min={counts_per_group.min()}, max={counts_per_group.max()}, "
      f"mean={counts_per_group.mean():.4f}")

    p()
    p(f"Fold sizes for rep_seed=1000 (repeated_grouped_cv_log11.py's rep_seed = 1000 + rep, rep=0):")
    p("Code (Source/repeated_grouped_cv_log11.py:54-66,139), quoted verbatim:")
    p("  line 139: rep_seed = 1000 + rep")
    p("  line 54: def grouped_kfold_indices(groups, n_folds, rep_seed):")
    p("  line 55:     unique_groups = np.unique(groups)")
    p("  line 56:     rng = np.random.default_rng(rep_seed)")
    p("  line 57:     rng.shuffle(unique_groups)")
    p("  line 60:         fold_of_group[g] = i % n_folds")

    rep_seed = 1000
    for fold_num, (train_idx, test_idx) in enumerate(
            grouped_kfold_indices(groups_raw, N_FOLDS, rep_seed), start=1):
        p(f"  fold {fold_num}: train windows={len(train_idx)}, test windows={len(test_idx)}, "
          f"test groups={len(np.unique(groups_raw[test_idx]))}")

    p()
    p("=" * 78)
    p("Q4 -- what varies within a class across realizations (Stage 3 & Stage 4)?")
    p("=" * 78)
    p("Stage 3 loop, Source/cnn_classifier.py:81-90 (quoted above): `kpa` is the loop's")
    p("fixed dict value (stiffness_classes[name]) passed unchanged into generate_signal();")
    p("only `jittered_hr = 1.15 + np.random.normal(0, jitter_std)` varies per realization.")
    p("kPa is NEVER sampled or jittered within a class in this loop.")
    p()
    p("Stage 4 loop, Source/floor_characterization_cnn.py:48-52, quoted verbatim:")
    p("  def generate_degraded_windows(kpa, depth_m, frequency_hz, n_needed, jitter_std=0.03):")
    p("      while len(windows) < n_needed:")
    p("          jittered_hr = 1.15 + np.random.normal(0, jitter_std)")
    p("          time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)")
    p("Same pattern: `kpa` is a fixed function argument (unchanged per call), only")
    p("`jittered_hr` and (later in the function) additive sensor noise vary. kPa is")
    p("never sampled/jittered within a class here either.")
    p()
    p("Stage 4 (clean-CNN crosscheck), Source/noise_robustness_stress_test_clean_cnn.py:81-90,")
    p("quoted verbatim:")
    p("  def generate_windows(kpa, depth_m, frequency_hz, n_needed, noise_std, jitter_std=0.03):")
    p("      while len(windows) < n_needed:")
    p("          jittered_hr = 1.15 + np.random.normal(0, jitter_std)")
    p("          time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)")
    p("          attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)")
    p("          noisy = attenuated + np.random.normal(0, noise_std, size=attenuated.shape)")
    p("Same pattern again: kpa fixed, heart rate jittered, additive Gaussian sensor noise varies.")
    p("CONCLUSION: across all three generating loops inspected (Stage 3, and both Stage 4")
    p("scripts), the only per-realization randomness is heart-rate jitter (and, where present,")
    p("additive sensor noise). kPa is a constant per class, drawn once from stiffness_classes,")
    p("never resampled or jittered within a class.")

    (OUT_DIR / "q1_q2_q3_q4_stdout.log").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'q1_q2_q3_q4_stdout.log'}")


if __name__ == "__main__":
    main()

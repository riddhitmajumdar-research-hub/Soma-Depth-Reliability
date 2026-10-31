"""
Lane6c -- Q7. READ-ONLY on the repo. Only imports the original, unmodified
generate_signal / stiffness_classes from Source/physics_model.py and SEED
from Source/constants.py. `extract_windows_with_k` below is a wrapper that
mirrors cnn_classifier.extract_raw_windows()'s exact window-selection logic
(same beat_period/beat_times/searchsorted formula, verified line-for-line
against Source/cnn_classifier.py:35-44) but additionally returns the beat
index k and the window-relative arrival offset (arrival_t - time[start_idx])
per window, which the original function does not expose. No repo file is
modified; no retraining.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

torch.set_num_threads(2)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Source"))

from physics_model import generate_signal, stiffness_classes  # noqa: E402
from constants import SEED                                    # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
N_PER_CLASS = 20000
JITTER_STD = 0.03
HR_DEFAULT = 1.15
WINDOW_SIZE = 100  # cnn_classifier.extract_raw_windows default


def extract_windows_with_k(signal, time, heart_rate_hz, tof, duration=3.0, window_size=WINDOW_SIZE):
    """Mirrors cnn_classifier.extract_raw_windows (Source/cnn_classifier.py:35-44)
    exactly, plus exposes k and offset = arrival_t - time[start_idx]."""
    beat_period = 1.0 / heart_rate_hz
    beat_times = np.arange(0, duration, beat_period)
    rows = []
    for k, beat_t in enumerate(beat_times):
        start_idx = np.searchsorted(time, beat_t)
        end_idx = start_idx + window_size
        if end_idx > len(signal):
            continue
        arrival_t = beat_t + tof
        offset = arrival_t - time[start_idx]
        rows.append({"window": signal[start_idx:end_idx], "k": k, "offset": offset})
    return rows


def main():
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    rng_seed = SEED
    np.random.seed(rng_seed)

    p("=" * 78)
    p(f"Q7 -- offset demonstration & noise-free Bayes accuracy from offset alone")
    p(f"n_per_class={N_PER_CLASS}, jitter_std={JITTER_STD}, heart_rate_hz={HR_DEFAULT}, seed={rng_seed}")
    p("=" * 78)

    class_names = list(stiffness_classes.keys())
    all_rows = []  # (offset, class_name, k)
    tof_by_class = {}
    dt_ref = None

    for name in class_names:
        kpa = stiffness_classes[name]
        for _ in range(N_PER_CLASS):
            jittered_hr = HR_DEFAULT + np.random.normal(0, JITTER_STD)
            time_arr, signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
            if dt_ref is None:
                dt_ref = time_arr[1] - time_arr[0]
            tof_by_class[name] = tof
            for row in extract_windows_with_k(signal, time_arr, jittered_hr, tof):
                all_rows.append((row["offset"], name, row["k"]))

    p(f"\nTotal windows collected: {len(all_rows)}  (target {N_PER_CLASS*3}; "
      f"shortfall from occasional truncated last-beat windows near t=3s, same as extract_raw_windows).")

    offsets = np.array([r[0] for r in all_rows])
    classes = np.array([r[1] for r in all_rows])
    ks = np.array([r[2] for r in all_rows])

    p(f"\ntof per class (deterministic, from time_of_flight_s): " +
      ", ".join(f"{n}={tof_by_class[n]*1000:.4f}ms" for n in class_names))
    p(f"sample spacing dt (time[1]-time[0]) = {dt_ref*1000:.6f}ms  "
      f"(np.linspace(0,3,1500) grid, not exactly 1/500Hz=2.000ms)")

    p("\n--- k=0 check: offset should equal tof exactly ---")
    for name in class_names:
        mask = (classes == name) & (ks == 0)
        vals = offsets[mask]
        max_abs_err = np.max(np.abs(vals - tof_by_class[name])) if len(vals) else float("nan")
        p(f"  {name:<22s} n={len(vals):6d}  offset==tof to within max|err|={max_abs_err:.3e}s "
          f"(float64 rounding only) -> {'MATCH' if max_abs_err < 1e-12 else 'MISMATCH'}")

    p("\n--- k>=1 check: offset should be tof - U[0, dt) ---")
    for name in class_names:
        mask = (classes == name) & (ks >= 1)
        vals = offsets[mask]
        frac = tof_by_class[name] - vals  # should be in [0, dt)
        p(f"  {name:<22s} n={len(vals):6d}  frac=tof-offset: min={frac.min()*1000:.5f}ms "
          f"max={frac.max()*1000:.5f}ms mean={frac.mean()*1000:.5f}ms  "
          f"(expected range [0,{dt_ref*1000:.5f})ms, expected mean~{dt_ref*1000/2:.5f}ms)  "
          f"-> {'MATCH' if (frac.min() >= -1e-12 and frac.max() < dt_ref + 1e-9) else 'MISMATCH'}")

    # ---- Exact analytic Bayes-optimal classifier ----------------------
    # offset's class-conditional distribution is KNOWN exactly (not merely
    # estimated) from the generator itself: with probability P(k=0|class)
    # offset is a point mass exactly at tof_class (float64-exact, per the
    # k=0 check above); with probability P(k>=1|class) offset is continuous
    # uniform on (tof_class-dt, tof_class]. A Gaussian-KDE density estimate
    # (below) smooths both the point mass and the sharp interval edges,
    # which is the wrong model class for this noise-free, exactly-known
    # generator and gives a biased accuracy estimate. This block instead
    # applies the TRUE Bayes-optimal decision rule directly: if x exactly
    # equals some class's tof (float64 equality -- exact by construction for
    # k=0 samples, never true for a continuous k>=1 draw), classify to that
    # class; otherwise compare uniform likelihoods 1/dt over whichever
    # classes' (tof-dt, tof] interval contains x (equal prior 1/3 each -> a
    # tie is broken by picking the lowest-index tied class, i.e. accuracy
    # contribution 1/(#tied) for that region, exactly the same expectation
    # any tie-break rule gives).
    p("\n" + "=" * 78)
    p("Noise-free Bayes accuracy from offset alone -- EXACT analytic decision rule")
    p("(uses the generator's known exact distribution: point mass at tof for k=0, "
      "uniform on (tof-dt,tof] for k>=1 -- no density estimation/smoothing)")
    p("=" * 78)

    dt_true = dt_ref
    intervals = {name: (tof_by_class[name] - dt_true, tof_by_class[name]) for name in class_names}

    tie_rng = np.random.default_rng(rng_seed)

    def exact_bayes_predict(x_vals):
        """Ties (a point lying in >1 class's support -- equal prior, equal
        uniform likelihood, so the Bayes-optimal decision is genuinely
        non-unique) are broken uniformly at random rather than by class
        order, so no single class absorbs the tied region's loss -- overall
        and per-k accuracy are invariant to the tie-break rule (any fixed
        rule gives the same expected accuracy in a tied region), but
        PER-CLASS recall is not, so the tie-break choice is reported
        explicitly here rather than silently picked."""
        preds = np.empty(len(x_vals), dtype=object)
        for i, x in enumerate(x_vals):
            exact_hits = [name for name in class_names if x == tof_by_class[name]]
            if exact_hits:
                preds[i] = exact_hits[0]
                continue
            candidates = [name for name in class_names if intervals[name][0] < x <= intervals[name][1]]
            if not candidates:
                # shouldn't happen given the k>=1 range check above, but
                # fall back to nearest tof rather than silently mis-scoring
                candidates = [min(class_names, key=lambda n: abs(x - tof_by_class[n]))]
            preds[i] = candidates[tie_rng.integers(len(candidates))] if len(candidates) > 1 else candidates[0]
        return preds

    exact_idx = np.arange(len(offsets))
    _, exact_test_idx = train_test_split(exact_idx, test_size=0.3, random_state=rng_seed, stratify=classes)
    exact_preds = exact_bayes_predict(offsets[exact_test_idx])
    exact_truth = classes[exact_test_idx]
    exact_correct = exact_preds == exact_truth
    exact_ks = ks[exact_test_idx]

    p(f"\nOverall exact Bayes accuracy (offset alone): {exact_correct.mean():.4%}  (n_test={len(exact_test_idx)})")
    p("\nPer-class accuracy (recall):")
    for name in class_names:
        m = exact_truth == name
        p(f"  {name:<22s} n={m.sum():6d}  acc={exact_correct[m].mean():.4%}")
    p("\nPer-k accuracy:")
    for k_val in sorted(np.unique(exact_ks)):
        m = exact_ks == k_val
        p(f"  k={k_val}  n={m.sum():6d}  acc={exact_correct[m].mean():.4%}")

    # ---- Bayes-optimal accuracy from offset alone (KDE density classifier, equal priors) ----
    p("\n" + "=" * 78)
    p("(comparison) same question via a 1D Gaussian-KDE density classifier, equal priors "
      "-- kept for transparency, but see note above on why the exact rule above is preferred")
    p("=" * 78)

    idx = np.arange(len(offsets))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=rng_seed, stratify=classes)

    # Bandwidth via Silverman's rule on the pooled training offsets (single global bandwidth,
    # shared across classes -- standard KDE practice, avoids per-class bandwidth cherry-picking).
    train_offsets_all = offsets[train_idx].reshape(-1, 1)
    n_tr = len(train_offsets_all)
    sigma = np.std(offsets[train_idx])
    bandwidth = 1.06 * sigma * n_tr ** (-1 / 5)  # Silverman's rule of thumb
    p(f"KDE bandwidth (Silverman's rule, pooled train offsets): {bandwidth:.6e}")

    # KDE fit/score cost is O(n_fit * n_query); with ~14k train points/class
    # against ~18k test rows this is too slow under the current host
    # contention (Lane 5's live sweep + the long ResNet sweep both running).
    # Subsampling the FIT set to 4000/class (bandwidth still computed from
    # the full training set above) keeps the 1D density estimate smooth
    # while cutting cost ~3.5x; a prior run without this cap did not finish.
    MAX_FIT_PER_CLASS = 4000
    fit_rng = np.random.default_rng(rng_seed)
    kdes = {}
    for name in class_names:
        mask = classes[train_idx] == name
        class_offsets = offsets[train_idx][mask]
        if len(class_offsets) > MAX_FIT_PER_CLASS:
            sel = fit_rng.choice(len(class_offsets), size=MAX_FIT_PER_CLASS, replace=False)
            class_offsets = class_offsets[sel]
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth, algorithm="kd_tree")
        kde.fit(class_offsets.reshape(-1, 1))
        kdes[name] = kde

    log_dens = np.column_stack([kdes[name].score_samples(offsets[test_idx].reshape(-1, 1)) for name in class_names])
    preds = np.array(class_names)[np.argmax(log_dens, axis=1)]
    truth = classes[test_idx]
    correct = preds == truth

    p(f"\nTrain/test split: {len(train_idx)}/{len(test_idx)} windows (stratified by class, test_size=0.3, seed={rng_seed})")
    p(f"\nOverall Bayes accuracy (offset alone): {correct.mean():.4%}  (n_test={len(test_idx)})")

    p("\nPer-class accuracy (recall):")
    for name in class_names:
        m = truth == name
        p(f"  {name:<22s} n={m.sum():6d}  acc={correct[m].mean():.4%}")

    p("\nPer-k accuracy:")
    test_ks = ks[test_idx]
    for k_val in sorted(np.unique(test_ks)):
        m = test_ks == k_val
        p(f"  k={k_val}  n={m.sum():6d}  acc={correct[m].mean():.4%}")

    (OUT_DIR / "q7_offset_bayes_stdout.log").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'q7_offset_bayes_stdout.log'}")


if __name__ == "__main__":
    main()

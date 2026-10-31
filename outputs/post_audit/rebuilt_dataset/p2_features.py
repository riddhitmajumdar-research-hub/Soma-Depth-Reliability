"""Timing-agnostic features for the Phase 2 logistic regression.

Every feature is computed from the 100-sample beat-aligned window alone. None uses the
true time of flight, the known arrival time or the class kPa. The old pipeline
(Source/source_classifier.py) cut each window at beat_t + tof, the oracle arrival time.

Features (in order):
  mf_lag_samples  lag (samples, sub-sample by parabolic interpolation) of the largest |cross-correlation|
                  between the window and the known pulse shape exp(-15 t) sin(2 pi 100 t), 76 samples at 500 Hz
  mf_peak         that largest |cross-correlation| divided by the template norm
  energy          sum of squares of the window
  peak_abs        max |x|
  argmax_abs_idx  index of max |x|
  first_cross_idx first index with |x| above 4 x the SD of samples 85..99 (pulse-free tail), 100 if none
"""
import numpy as np

FEATURE_NAMES = ["mf_lag_samples", "mf_peak", "energy", "peak_abs", "argmax_abs_idx", "first_cross_idx"]
_t = np.arange(76) / 500.0
TEMPLATE = np.exp(-15 * _t) * np.sin(2 * np.pi * 100 * _t)
_TN = np.linalg.norm(TEMPLATE)


def lr_features(W):
    W = np.asarray(W, dtype=np.float64)
    out = np.zeros((W.shape[0], len(FEATURE_NAMES)))
    for i, w in enumerate(W):
        c = np.correlate(w, TEMPLATE, mode="valid")  # 25 lags
        a = np.abs(c)
        k = int(np.argmax(a))
        frac = 0.0
        if 0 < k < len(a) - 1:
            y0, y1, y2 = a[k - 1], a[k], a[k + 1]
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        sd = max(float(np.std(w[85:])), 1e-9)
        idx = np.where(np.abs(w) > 4 * sd)[0]
        out[i] = [k + frac, a[k] / _TN, float(np.sum(w ** 2)), float(np.abs(w).max()),
                  float(np.argmax(np.abs(w))), float(idx[0]) if idx.size else 100.0]
    return out


GATED_NAMES = FEATURE_NAMES + ["detected", "lag_if_detected", "log1p_mf_snr"]


def lr_features_gated(W):
    """Secondary arm only. Adds a pulse-detected flag (matched-filter peak above 4 x tail SD),
    the lag set to 0 when nothing is detected, and log1p(matched-filter peak / tail SD).
    Still computed from the window alone."""
    W = np.asarray(W, dtype=np.float64)
    F = lr_features(W)
    sd = np.maximum(W[:, 85:].std(axis=1), 1e-9)
    snr = F[:, 1] / sd
    det = (snr > 4.0).astype(float)
    return np.column_stack([F, det, F[:, 0] * det, np.log1p(snr)])

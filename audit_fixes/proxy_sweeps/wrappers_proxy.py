# Proxy-sweep wrappers: one wrapper per condition of the Table 1 sensitivity
# runs and of the eight-network rerun (decisions D1 to D9 in PREREG.md).
# The window generator is a copy of
# audit_fixes/cloud/battery/seeds_101-104/wrappers.py
# generate_degraded_windows_shifted (and, through it, of
# Source/degradation_aware_final_sweep.generate_degraded_windows) with the
# parameters of these conditions added; nothing in Source/ or in the battery
# folder is edited. With no shift the draws are identical to the original
# generator, so the baseline training set equals
# Source/degradation_aware_final_sweep.build_degradation_aware_dataset.
#
# Every condition changes the data that the real model and its controls are
# trained on AND the data they are evaluated on.

import os
import sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "Source"))

from physics_model import time_of_flight_s, stiffness_classes  # noqa: E402
from attenuation_model import attenuation_coefficient, LIVER_VISCOSITY_BY_CLASS  # noqa: E402
from cnn_classifier import extract_raw_windows, SimpleCNN  # noqa: E402
from constants import SENSOR_SNR_DB  # noqa: E402

SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)
SAMPLE_RATE_HZ = 500
SAMPLE_INTERVAL_S = 1.0 / SAMPLE_RATE_HZ      # 2 ms

# D6: alternative stiffness set (kPa)
ALT_STIFFNESS = {"healthy": 5.4, "significant_fibrosis": 7.3, "cirrhosis": 11.4}


def generate_signal_proxy(youngs_modulus_kpa, duration=3.0, heart_rate_hz=1.15, sample_rate=SAMPLE_RATE_HZ,
                          carrier_hz=100, distance_m=0.010, beat_jitter_s=0.0, rng=None):
    """Copy of physics_model.generate_signal with the carrier frequency, the
    tracking distance and a Gaussian shift of each beat's pulse arrival time
    (SD beat_jitter_s seconds, drawn independently per beat) as arguments."""
    rng = rng if rng is not None else np.random
    tof = time_of_flight_s(youngs_modulus_kpa, distance_m=distance_m)
    beat_times = np.arange(0, duration, 1.0 / heart_rate_hz)
    n_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)
    for beat_t in beat_times:
        arrival_t = beat_t + tof
        if beat_jitter_s:
            arrival_t += rng.normal(0, beat_jitter_s)
        pulse_window = 0.15
        mask = (time >= arrival_t) & (time <= arrival_t + pulse_window)
        local_t = time[mask] - arrival_t
        pulse_shape = np.exp(-15 * local_t) * np.sin(2 * np.pi * carrier_hz * local_t)
        signal[mask] += pulse_shape
    return time, signal, tof


def classes_for(shift):
    return dict(shift.get("stiffness_set") or stiffness_classes)


def generate_windows_proxy(class_name, depth_m, frequency_hz, n_needed, shift=None, jitter_std=0.03, rng=None):
    """shift keys (all optional; default = original behaviour):
      carrier_hz: pulse carrier; also the frequency of the attenuation coefficient (D3).
      distance_m: tracking distance of the time of flight (D4).
      beat_jitter_samples: SD of the Gaussian shift of each beat's arrival time, in samples of 2 ms (D2).
      window_size: samples per window (D5).
      stiffness_set: {class: kPa} replacing the class stiffness values (D6). The viscosity is taken
        from the class name, not looked up from the stiffness, so a changed stiffness does not
        change the viscosity of the class.
    """
    shift = shift or {}
    rng = rng if rng is not None else np.random
    kpa = classes_for(shift)[class_name]
    eta = LIVER_VISCOSITY_BY_CLASS[class_name]
    carrier = shift.get("carrier_hz", 100)
    dist = shift.get("distance_m", 0.010)
    jit_s = shift.get("beat_jitter_samples", 0.0) * SAMPLE_INTERVAL_S
    wsize = shift.get("window_size", 100)
    f_att = shift.get("carrier_hz", frequency_hz)
    windows = []
    while len(windows) < n_needed:
        jittered_hr = 1.15 + rng.normal(0, jitter_std)
        time, clean_signal, tof = generate_signal_proxy(
            kpa, heart_rate_hz=jittered_hr, carrier_hz=carrier, distance_m=dist, beat_jitter_s=jit_s, rng=rng)
        alpha = attenuation_coefficient(kpa, f_att, eta=eta)
        attenuated = clean_signal * np.exp(-alpha * depth_m)
        noisy = attenuated + rng.normal(0, SENSOR_NOISE_STD, size=attenuated.shape)
        windows.extend(extract_raw_windows(noisy, time, jittered_hr, window_size=wsize))
    return windows[:n_needed]


class SimpleCNNWindow(nn.Module):
    """SimpleCNN with the first dense layer sized to 16 x floor(floor(L/2)/2) (D5).
    For L = 100 this is the original 16 x 25 and the original class is used instead."""
    def __init__(self, window_size, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16 * ((window_size // 2) // 2), 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


def train_cnn_proxy(X, y_encoded, n_epochs=60, window_size=100, n_classes=3):
    """Copy of Source/degradation_aware_final_sweep.train_cnn (Adam, lr 0.001, full batch)
    with the model class chosen by window size."""
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_encoded, dtype=torch.long)
    model = SimpleCNN(n_classes=n_classes) if window_size == 100 else SimpleCNNWindow(window_size, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward()
        opt.step()
    model.eval()
    return model


# Condition registry, in the order of PREREG.md.
# id -> dict(label, kind, shift, epochs, default_run, note)
#   kind "mixed": trained on the mixed-depth degraded data generated WITH the shift
#   kind "clean": clean-only training set (lane5_harness_clean.py), evaluated under the baseline degradation
# Every condition retrains its own real model and its own 200 controls (shuffled labels of the same
# training data, same number of epochs as the real model).
_C = [
    ("0_baseline", "baseline: mixed-depth, 60 epochs, 100 samples, 10 mm, 100 Hz", "mixed", {}, 60, True),
    ("1_epochs_600", "training budget 600 epochs", "mixed", {}, 600, True),
    ("2_jitter_0.5", "beat-time jitter, SD 0.5 sample (1 ms)", "mixed", {"beat_jitter_samples": 0.5}, 60, True),
    ("2_jitter_1.0", "beat-time jitter, SD 1.0 sample (2 ms)", "mixed", {"beat_jitter_samples": 1.0}, 60, True),
    ("3_carrier_50", "pulse carrier 50 Hz", "mixed", {"carrier_hz": 50}, 60, True),
    ("3_carrier_150", "pulse carrier 150 Hz", "mixed", {"carrier_hz": 150}, 60, True),
    ("4_track_5", "tracking distance 5 mm", "mixed", {"distance_m": 0.005}, 60, True),
    ("4_track_10", "tracking distance 10 mm (same data as the baseline)", "mixed", {"distance_m": 0.010}, 60, False),
    ("4_track_15", "tracking distance 15 mm", "mixed", {"distance_m": 0.015}, 60, True),
    ("4_track_20", "tracking distance 20 mm", "mixed", {"distance_m": 0.020}, 60, True),
    ("5_window_50", "window length 50 samples", "mixed", {"window_size": 50}, 60, True),
    ("5_window_150", "window length 150 samples", "mixed", {"window_size": 150}, 60, True),
    ("6_stiffness_alt", "stiffness set 5.4 / 7.3 / 11.4 kPa", "mixed", {"stiffness_set": ALT_STIFFNESS}, 60, True),
    ("7_clean_only", "clean-only training, baseline evaluation", "clean", {}, 60, True),
]
CONDITIONS = [
    {"id": i, "label": l, "kind": k, "shift": s, "epochs": e, "default_run": d,
     # the shift applies to the evaluation windows of every condition; to the training windows of 'mixed' ones
     "train_shift": (s if k == "mixed" else {}), "eval_shift": ({} if k == "clean" else s)}
    for i, l, k, s, e, d in _C
]
BY_ID = {c["id"]: c for c in CONDITIONS}


def _make(shift):
    def wrapper(class_name, depth_m, frequency_hz, n_needed, rng=None):
        return generate_windows_proxy(class_name, depth_m, frequency_hz, n_needed, shift=shift, rng=rng)
    return wrapper


# one training wrapper and one evaluation wrapper per condition
TRAIN_WRAPPERS = {c["id"]: _make(c["train_shift"]) for c in CONDITIONS}
EVAL_WRAPPERS = {c["id"]: _make(c["eval_shift"]) for c in CONDITIONS}

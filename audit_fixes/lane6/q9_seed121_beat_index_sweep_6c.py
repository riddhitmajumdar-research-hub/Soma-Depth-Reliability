"""
Lane6c -- Q9 (my own pass; does not touch or overwrite
q9_seed121_beat_index_sweep.py, an earlier file already in this folder from
another pass -- see FACTS_6c.md for why that file answers a different
question: it uses Source/seed_weights_resnet/degradation_aware_resnet_seed121.pt,
the RESNET sweep (degradation_aware_resnet_final_sweep.py), which the
protocol explicitly says is a different job to ignore for Q9. This script uses
the actual requested checkpoints instead: the certified SimpleCNN
(outputs/stage4/models/soma_certified_model_seed121_corrected_noise.pt) and
Lane 5's own per-seed SimpleCNN weights
(audit_fixes/lane5/recall_floor_main_run/seed_weights/degradation_aware_seed{seed}.pt), for
whichever seeds are ALREADY COMPLETE per audit_fixes/lane5/recall_floor_main_run/logs/lane5_run.log
-- no retraining, no touching the still-running Lane 5 process.

READ-ONLY on the repo. Imports SimpleCNN + generate_signal + attenuate_signal
unchanged. generate_degraded_windows() (Source/degradation_aware_final_sweep.py)
does not expose beat index k, so this script uses a parallel wrapper mirroring
its code exactly (compared line-for-line below) plus k-tagging, and its
window CONTENT is checked against the real extract_raw_windows() output.
Uses <=2 threads (amendment: machine is busy running Lane 5).
"""
import re
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Source"))

from physics_model import generate_signal, stiffness_classes            # noqa: E402
from cnn_classifier import SimpleCNN, extract_raw_windows               # noqa: E402
from attenuation_model import attenuate_signal                          # noqa: E402
from sklearn.preprocessing import LabelEncoder                          # noqa: E402
from degradation_aware_final_sweep import EVAL_DEPTHS_CM, FREQ          # noqa: E402
from constants import SENSOR_SNR_DB                                      # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CERTIFIED_CKPT = REPO_ROOT / "outputs/stage4/models/soma_certified_model_seed121_corrected_noise.pt"
LANE5_DIR = REPO_ROOT / "audit_fixes/lane5/recall_floor_main_run"
LANE5_RUN_LOG = LANE5_DIR / "logs/lane5_run.log"
LANE5_WEIGHTS_DIR = LANE5_DIR / "seed_weights"

N_SAMPLES_PER_CELL = 300  # matches degradation_aware_resnet_final_sweep.py / the standard sweep
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)
CLASS_NAMES = sorted(stiffness_classes.keys())


def completed_lane5_seeds():
    """Only seeds whose 'Seed N done' line already appears in Lane 5's own
    run log AND whose .pt file exists on disk -- both checked, since a crash
    mid-checkpoint-write could leave a log line without a usable file (none
    observed, but checked rather than assumed)."""
    if not LANE5_RUN_LOG.exists():
        return []
    text = LANE5_RUN_LOG.read_text()
    done = sorted(set(int(m) for m in re.findall(r"Seed (\d+) done in", text)))
    have_file = []
    for s in done:
        p = LANE5_WEIGHTS_DIR / f"degradation_aware_seed{s}.pt"
        if p.exists():
            have_file.append(s)
        else:
            print(f"  seed {s}: log says done but {p.name} missing -- skipping")
    return have_file


def generate_degraded_windows_with_k(kpa, depth_m, frequency_hz, n_needed, jitter_std=0.03):
    """Mirrors Source/degradation_aware_final_sweep.py:47-56 verbatim:
        def generate_degraded_windows(kpa, depth_m, frequency_hz, n_needed, jitter_std=0.03):
            windows = []
            while len(windows) < n_needed:
                jittered_hr = 1.15 + np.random.normal(0, jitter_std)
                time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
                attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)
                noisy = attenuated + np.random.normal(0, SENSOR_NOISE_STD, size=attenuated.shape)
                rows = extract_raw_windows(noisy, time, jittered_hr)
                windows.extend(rows)
            return windows[:n_needed]
    -- identical random-draw sequence and order, plus k tagging via the same
    beat_times/searchsorted logic extract_raw_windows uses internally, and a
    call to the real extract_raw_windows() to confirm window content matches."""
    windows, ks = [], []
    while len(windows) < n_needed:
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time_arr, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)
        noisy = attenuated + np.random.normal(0, SENSOR_NOISE_STD, size=attenuated.shape)

        beat_period = 1.0 / jittered_hr
        beat_times = np.arange(0, 3.0, beat_period)
        my_rows, my_ks = [], []
        for k, beat_t in enumerate(beat_times):
            start_idx = np.searchsorted(time_arr, beat_t)
            end_idx = start_idx + 100
            if end_idx > len(noisy):
                continue
            my_rows.append(noisy[start_idx:end_idx])
            my_ks.append(k)

        real_rows = extract_raw_windows(noisy, time_arr, jittered_hr)
        assert len(real_rows) == len(my_rows)
        for a, b in zip(real_rows, my_rows):
            assert np.array_equal(a, b)

        windows.extend(my_rows)
        ks.extend(my_ks)
    return windows[:n_needed], ks[:n_needed]


def eval_model_on_grid(model, encoder, seed_label, lines):
    p = lines.append
    k0_correct = k0_total = 0
    kge1_correct = kge1_total = 0
    per_depth = []
    for depth_cm in EVAL_DEPTHS_CM:
        depth_m = depth_cm / 100
        n_per_class = N_SAMPLES_PER_CELL // len(CLASS_NAMES)
        X_cell, y_cell_names, k_cell = [], [], []
        for name in CLASS_NAMES:
            kpa = stiffness_classes[name]
            windows, ks = generate_degraded_windows_with_k(kpa, depth_m, FREQ, n_per_class)
            X_cell.extend(windows)
            y_cell_names.extend([name] * len(windows))
            k_cell.extend(ks)
        X_cell = np.array(X_cell)
        y_cell_encoded = encoder.transform(y_cell_names)
        k_cell = np.array(k_cell)
        X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            preds = model(X_cell_t).argmax(dim=1).numpy()
        correct = preds == y_cell_encoded

        m0 = k_cell == 0
        m1 = k_cell >= 1
        k0_correct += correct[m0].sum(); k0_total += m0.sum()
        kge1_correct += correct[m1].sum(); kge1_total += m1.sum()
        per_depth.append((depth_cm, m0.sum(), correct[m0].mean() if m0.sum() else float("nan"),
                           m1.sum(), correct[m1].mean() if m1.sum() else float("nan")))

    p(f"\n--- {seed_label} ---")
    for depth_cm, n0, a0, n1, a1 in per_depth:
        p(f"  depth={depth_cm:6.2f}cm  k=0: n={n0:4d} acc={a0:.4%}   k>=1: n={n1:4d} acc={a1:.4%}")
    p(f"  POOLED: k=0 n={k0_total} acc={k0_correct/k0_total:.4%}   "
      f"k>=1 n={kge1_total} acc={kge1_correct/kge1_total:.4%}")


def main():
    lines = []
    def p(s=""):
        print(s); lines.append(s)

    p("=" * 78)
    p("Q9 (6c) -- SimpleCNN degradation-aware checkpoints, standard grid "
      f"(100Hz, {SENSOR_SNR_DB}dB), accuracy by k=0 vs k>=1")
    p("=" * 78)
    p(f"EVAL_DEPTHS_CM ({len(EVAL_DEPTHS_CM)} depths): "
      f"{[f'{d:.2f}' for d in EVAL_DEPTHS_CM]}")
    p(f"N_SAMPLES_PER_CELL={N_SAMPLES_PER_CELL}, FREQ={FREQ}Hz")

    # dummy encoder matching the sorted class-name convention used throughout
    encoder = LabelEncoder()
    encoder.fit(CLASS_NAMES)

    p(f"\nCertified checkpoint: {CERTIFIED_CKPT.relative_to(REPO_ROOT)}")
    model = SimpleCNN(n_classes=3)
    model.load_state_dict(torch.load(CERTIFIED_CKPT, map_location="cpu"))
    model.eval()
    eval_model_on_grid(model, encoder, "seed121 (certified checkpoint)", lines)

    seeds = completed_lane5_seeds()
    p(f"\nLane 5 seeds completed so far (per lane5_run.log + file check): {seeds}")
    for s in seeds:
        ckpt = LANE5_WEIGHTS_DIR / f"degradation_aware_seed{s}.pt"
        model_s = SimpleCNN(n_classes=3)
        model_s.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model_s.eval()
        eval_model_on_grid(model_s, encoder, f"lane5 seed{s}", lines)

    (OUT_DIR / "q9_seed121_beat_index_sweep_6c_stdout.log").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'q9_seed121_beat_index_sweep_6c_stdout.log'}")


if __name__ == "__main__":
    main()

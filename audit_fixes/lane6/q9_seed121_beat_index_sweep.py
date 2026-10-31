"""
Lane6c -- Q9. READ-ONLY on the repo, and does NOT touch the running sweep
(pid 7429) or its outputs -- this script only READS the already-saved
seed-121 checkpoint (Source/seed_weights_resnet/degradation_aware_resnet_seed121.pt,
confirmed to already exist on disk: degradation_aware_resnet_final_sweep.py
saves it via torch.save() immediately after training, before the slower
per-seed depth sweep evaluation completes -- so this weights file is final
and untouched by the still-running process, which is now past seed 121).

Uses the standard grid (100Hz, 27.05dB) EXACTLY as imported, unmodified,
from Source/degradation_aware_final_sweep.py: EVAL_DEPTHS_CM, FREQ=100,
SENSOR_SNR_DB=27.05 (constants.py). N_SAMPLES_PER_CELL=300 matches
Source/degradation_aware_resnet_final_sweep.py's own sweep exactly.

generate_degraded_windows() (imported unchanged) does not expose beat index
k, so this script uses a parallel wrapper that mirrors its exact code
(verified line-for-line against Source/degradation_aware_final_sweep.py:47-56)
but also tags each window with k.

Seeds 122-124 do not exist yet (Lane 5's 24-seed sweep, pid 7429, is not
finished -- confirmed via `ps -p 7429` still running, and
Source/logs/degradation_aware_resnet_final_results.csv only has seeds
101-120 complete, with the log showing seed 121 as the most recently
started). This script only covers seed 121; the "repeat for all seeds"
part of Q9 is NOT done here and is called out explicitly in FACTS.md as
pending until Lane 5 finishes.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Source"))

from physics_model import generate_signal, stiffness_classes    # noqa: E402
from cnn_classifier import extract_raw_windows                  # noqa: E402
from attenuation_model import attenuate_signal                  # noqa: E402
from resnet_classifier import ResNet1D                          # noqa: E402
from constants import SENSOR_SNR_DB                              # noqa: E402
from degradation_aware_final_sweep import EVAL_DEPTHS_CM, FREQ  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
CKPT_PATH = REPO_ROOT / "Source" / "seed_weights_resnet" / "degradation_aware_resnet_seed121.pt"
N_SAMPLES_PER_CELL = 300  # Source/degradation_aware_resnet_final_sweep.py:98
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)  # mirrors degradation_aware_final_sweep.py:39


def generate_degraded_windows_with_k(kpa, depth_m, frequency_hz, n_needed, jitter_std=0.03):
    """Mirrors degradation_aware_final_sweep.generate_degraded_windows
    (Source/degradation_aware_final_sweep.py:47-56) exactly, plus k."""
    rows = []
    while len(rows) < n_needed:
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time_arr, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)
        noisy = attenuated + np.random.normal(0, SENSOR_NOISE_STD, size=attenuated.shape)
        beat_period = 1.0 / jittered_hr
        beat_times = np.arange(0, 3.0, beat_period)
        for k, beat_t in enumerate(beat_times):
            start_idx = np.searchsorted(time_arr, beat_t)
            end_idx = start_idx + 100
            if end_idx > len(noisy):
                continue
            rows.append((noisy[start_idx:end_idx], k))
    return rows[:n_needed]


def main():
    lines = []

    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 78)
    p("Q9 -- seed-121 ResNet1D, standard grid (100Hz, 27.05dB), accuracy by k=0 vs k>=1")
    p("=" * 78)
    p(f"Checkpoint: {CKPT_PATH.relative_to(REPO_ROOT)}")
    p(f"FREQ={FREQ}Hz, SENSOR_SNR_DB={SENSOR_SNR_DB}dB, N_SAMPLES_PER_CELL={N_SAMPLES_PER_CELL}")
    p(f"EVAL_DEPTHS_CM ({len(EVAL_DEPTHS_CM)} depths): {[f'{d:.2f}' for d in EVAL_DEPTHS_CM]}")

    if not CKPT_PATH.exists():
        p(f"\nERROR: checkpoint not found at {CKPT_PATH}. STOPPING -- not fabricating a result.")
        (OUT_DIR / "q9_seed121_beat_index_sweep_stdout.log").write_text("\n".join(lines) + "\n")
        return

    model = ResNet1D(n_classes=3)
    state_dict = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    class_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_names)

    # Need the SAME label encoding the sweep used: LabelEncoder().fit_transform(y_train_names)
    # sorts classes alphabetically by default -- reproduce that ordering exactly, read-only.
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    encoder.fit(class_names)

    k0_correct, k0_total = 0, 0
    kge1_correct, kge1_total = 0, 0
    per_depth_rows = []

    np.random.seed(121)  # reproducible eval batch; does not affect the already-trained checkpoint

    for depth_cm in EVAL_DEPTHS_CM:
        depth_m = depth_cm / 100
        X_cell, y_cell_names, k_cell = [], [], []
        for name in class_names:
            kpa = stiffness_classes[name]
            for window, k in generate_degraded_windows_with_k(kpa, depth_m, FREQ, n_per_class):
                X_cell.append(window)
                y_cell_names.append(name)
                k_cell.append(k)
        X_cell = np.array(X_cell)
        y_cell_encoded = encoder.transform(y_cell_names)
        k_cell = np.array(k_cell)
        X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            preds = model(X_cell_t).argmax(dim=1).numpy()
        correct = preds == y_cell_encoded

        m0 = k_cell == 0
        m1 = k_cell >= 1
        acc0 = correct[m0].mean() if m0.sum() else float("nan")
        acc1 = correct[m1].mean() if m1.sum() else float("nan")

        k0_correct += correct[m0].sum(); k0_total += m0.sum()
        kge1_correct += correct[m1].sum(); kge1_total += m1.sum()

        per_depth_rows.append((depth_cm, m0.sum(), acc0, m1.sum(), acc1))
        p(f"  depth={depth_cm:6.2f}cm  k=0: n={m0.sum():4d} acc={acc0:.4%}   "
          f"k>=1: n={m1.sum():4d} acc={acc1:.4%}")

    p("\n" + "=" * 78)
    p("Pooled across all depths:")
    p(f"  k=0   : n={k0_total:5d}  accuracy={k0_correct/k0_total:.4%}")
    p(f"  k>=1  : n={kge1_total:5d}  accuracy={kge1_correct/kge1_total:.4%}")
    p(f"  total : n={k0_total+kge1_total:5d}")

    p("\nNOTE: only seed 121 is covered here. Seeds 122-124 do not exist yet -- "
      "Lane 5's 24-seed sweep (pid 7429) has not finished; only seeds 101-120 are in "
      "logs/degradation_aware_resnet_final_results.csv, and the log shows seed 121 as "
      "the most recently started. Repeating this analysis for all seeds is pending "
      "until that run completes and seeds 122-124's weight files exist.")

    (OUT_DIR / "q9_seed121_beat_index_sweep_stdout.log").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {OUT_DIR / 'q9_seed121_beat_index_sweep_stdout.log'}")


if __name__ == "__main__":
    main()

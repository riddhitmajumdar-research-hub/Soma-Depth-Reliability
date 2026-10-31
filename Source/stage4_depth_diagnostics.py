# SOMA Signal Fidelity Investigation
# Stage 4 -- Multi-depth confusion matrices and depth-progression t-SNE
# File: Source/stage4_depth_diagnostics.py

import numpy as np
import torch
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED, NOMINAL_FREQ_HZ

torch.manual_seed(SEED)
np.random.seed(SEED)

FREQ = NOMINAL_FREQ_HZ

# --- IMPORTANT: these four depths are STALE, not part of a specific fix ---
# Original [0, 2.73, 5.45, 8.18] were meaningful under the OLD 67.03dB noise
# model. Under the new 27.05dB model, the floor is likely much shallower --
# run floor_characterization_cnn.py / extend_20hz.py FIRST, see where the
# new floor lands, then replace DEPTHS_TO_SHOW with four points spanning
# clean -> near-floor -> at-floor -> collapsed under the NEW numbers.
DEPTHS_TO_SHOW = [0, 2.73, 4.30, 6.82]  # updated for corrected 27.05dB noise model

N_SAMPLES_PER_CLASS = 30

# --- FIX #1: sensor noise floor (see floor_characterization.py for full citation trail) ---
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)


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


if __name__ == "__main__":
    print("Loading saved model (no retraining needed)...")
    print("NOTE: re-run extend_20hz.py or floor_characterization_cnn.py FIRST")
    print("under the new SNR, or this loads a model trained under the OLD 67.03dB.")
    with open("outputs/stage4/data/stage4_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    class_names = list(encoder.classes_)

    model = SimpleCNN(n_classes=len(class_names))
    model.load_state_dict(torch.load("outputs/stage4/models/stage4_cnn_model.pt"))
    model.eval()
    os.makedirs("figures", exist_ok=True)

    embeddings = {}
    def hook(module, input, output):
        embeddings["value"] = output
    model.fc1.register_forward_hook(hook)

    class_kpa_names = sorted(stiffness_classes.keys())

    fig_cm, axes_cm = plt.subplots(1, 4, figsize=(20, 5))
    fig_tsne, axes_tsne = plt.subplots(1, 4, figsize=(20, 5))
    colors = {"healthy": "#0F6E56", "significant_fibrosis": "#BA7517", "cirrhosis": "#993C1D"}

    for col, depth_cm in enumerate(DEPTHS_TO_SHOW):
        depth_m = depth_cm / 100
        X_cell = []
        y_cell_names = []
        for name in class_kpa_names:
            kpa = stiffness_classes[name]
            windows = generate_degraded_windows(kpa, depth_m, FREQ, N_SAMPLES_PER_CLASS)
            X_cell.extend(windows)
            y_cell_names.extend([name] * len(windows))
        X_cell = np.array(X_cell)
        y_cell_encoded = encoder.transform(y_cell_names)
        X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            outputs = model(X_cell_t)
            preds = outputs.argmax(dim=1).numpy()
            embed_vals = embeddings["value"].numpy()

        cm = confusion_matrix(y_cell_encoded, preds, labels=range(len(class_names)))
        im = axes_cm[col].imshow(cm, cmap="YlOrRd")
        axes_cm[col].set_xticks(range(len(class_names)))
        axes_cm[col].set_yticks(range(len(class_names)))
        axes_cm[col].set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
        axes_cm[col].set_yticklabels(class_names, fontsize=8)
        axes_cm[col].set_title(f"Depth = {depth_cm}cm")
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                axes_cm[col].text(j, i, cm[i, j], ha="center", va="center",
                                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=9)

        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="pca")
        embed_2d = tsne.fit_transform(embed_vals)
        for name in class_kpa_names:
            mask = np.array(y_cell_names) == name
            axes_tsne[col].scatter(embed_2d[mask, 0], embed_2d[mask, 1],
                                     color=colors[name], label=name, alpha=0.7, s=25)
        axes_tsne[col].set_title(f"Depth = {depth_cm}cm")
        if col == 0:
            axes_tsne[col].legend(fontsize=7)

        print(f"Depth {depth_cm}cm: accuracy = {np.mean(preds == y_cell_encoded):.2%}")

    fig_cm.suptitle("Stage 4 -- Confusion matrices across depth (100Hz)")
    fig_cm.tight_layout()
    fig_cm.savefig("figures/stage4_multidepth_confusion.png", dpi=200, bbox_inches="tight")

    fig_tsne.suptitle("Stage 4 -- CNN's internal representation (t-SNE) collapsing with depth (100Hz)")
    fig_tsne.tight_layout()
    fig_tsne.savefig("figures/stage4_depth_tsne.png", dpi=200, bbox_inches="tight")

    plt.show()
    print("\nBoth figures saved.")

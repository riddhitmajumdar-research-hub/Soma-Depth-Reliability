# SOMA Signal Fidelity Investigation
# COPY of frozen_embedding_linear_probe.py re-run under the corrected 27.05dB sensor noise.
# The original (67.03dB, 8.18cm) is kept unchanged as the historical record.
# Only the noise level, the evaluated depth (argv[1]) and the CSV path differ.
# The CNN is trained on clean, noiseless windows (extend_20hz.py / build_raw_dataset), so its weights do not depend on the noise level.
# Part 3(e) -- frozen-embedding linear probe at the 8.18cm Log 14 anomaly.
#
# Log 14 (README): the CLEAN-trained CNN (models/stage4_cnn_model.pt,
# trained under the OLD, superseded 67.03dB noise model -- this is the
# specific model/noise pairing that produced the anomaly, per
# stage4_depth_diagnostics.py's own header note) shows total confusion-
# matrix collapse at 8.18cm, yet t-SNE on its frozen 32-dim fc1
# representation still shows three separated clusters. That's a
# visualization, not quantitative evidence. This script freezes that
# model's fc1 output, generates an INDEPENDENT held-out set at the same
# depth/noise, and trains a plain logistic regression on top of the
# frozen embeddings -- if it scores meaningfully above chance (33%) where
# the deployed argmax decision boundary scores none, that's quantitative
# evidence the representation floor sits deeper than the operational floor.

import numpy as np
import torch
import pickle
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows
import sys
from constants import SEED, SENSOR_SNR_DB

SNR_DB_OLD = SENSOR_SNR_DB  # 27.05dB, corrected sensor noise (name kept from the original script)
SENSOR_NOISE_STD_OLD = 10 ** (-SNR_DB_OLD / 20)
FREQ = 100
DEPTH_CM = float(sys.argv[1])
N_SAMPLES_PER_CLASS = 200  # independent held-out set, larger than the original diagnostic run
CSV_PATH = f"logs/frozen_embedding_linear_probe_27p05dB_{str(DEPTH_CM).replace('.','p')}cm.csv"

np.random.seed(SEED + 777)  # different draw than the original Log 14 diagnostic run -- genuinely independent


def generate_windows(kpa, depth_m, frequency_hz, n_needed, jitter_std=0.03):
    windows = []
    while len(windows) < n_needed:
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)
        noisy = attenuated + np.random.normal(0, SENSOR_NOISE_STD_OLD, size=attenuated.shape)
        rows = extract_raw_windows(noisy, time, jittered_hr)
        windows.extend(rows)
    return windows[:n_needed]


def main():
    with open("results/stage4_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    class_names_sorted = list(encoder.classes_)

    model = SimpleCNN(n_classes=len(class_names_sorted))
    model.load_state_dict(torch.load("models/stage4_cnn_model.pt", map_location="cpu"))
    model.eval()

    embeddings_out = {}
    def hook(module, inp, output):
        embeddings_out["value"] = output.detach()
    model.fc1.register_forward_hook(hook)

    depth_m = DEPTH_CM / 100
    X, y_names = [], []
    for name in class_names_sorted:
        kpa = stiffness_classes[name]
        windows = generate_windows(kpa, depth_m, FREQ, N_SAMPLES_PER_CLASS)
        X.extend(windows)
        y_names.extend([name] * len(windows))
    X = np.array(X)
    y = encoder.transform(y_names)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        logits = model(X_t)
    preds_deployed = logits.argmax(dim=1).numpy()
    deployed_acc = accuracy_score(y, preds_deployed)

    # frozen 32-dim embedding = fc1 output, PRE-relu/dropout (raw linear features
    # the deployed model's fc2 decision boundary itself operates on downstream of relu;
    # using the raw fc1 linear output is the most literal "frozen internal representation")
    embeddings = embeddings_out["value"].numpy()

    # independent held-out split of the frozen embeddings for the probe
    emb_train, emb_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.4, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    emb_train_scaled = scaler.fit_transform(emb_train)
    emb_test_scaled = scaler.transform(emb_test)
    probe = LogisticRegression(max_iter=2000, C=0.5)  # embeddings are near-perfectly linearly separable; mild L2 avoids unbounded-weight solver warnings without changing the substantive result
    probe.fit(emb_train_scaled, y_train)
    probe_preds = probe.predict(emb_test_scaled)
    probe_acc = accuracy_score(y_test, probe_preds)

    chance = 1.0 / len(class_names_sorted)

    print(f"Depth: {DEPTH_CM}cm  |  noise: {SNR_DB_OLD}dB (corrected noise)")
    print(f"Deployed model (argmax) accuracy on independent held-out set: {deployed_acc:.4f}")
    print(f"Frozen-embedding linear probe accuracy (independent 60/40 split of same embeddings): {probe_acc:.4f}")
    print(f"Chance level (3 classes): {chance:.4f}")
    print(f"Probe accuracy - chance: {probe_acc - chance:+.4f}")
    print(f"Probe accuracy - deployed accuracy: {probe_acc - deployed_acc:+.4f}")

    row = {
        "depth_cm": DEPTH_CM, "snr_db": SNR_DB_OLD,
        "n_samples_per_class": N_SAMPLES_PER_CLASS,
        "deployed_model_accuracy": deployed_acc,
        "frozen_probe_accuracy": probe_acc,
        "chance_level": chance,
        "probe_minus_chance": probe_acc - chance,
        "probe_minus_deployed": probe_acc - deployed_acc,
    }
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"\nSaved to {CSV_PATH}")


if __name__ == "__main__":
    main()

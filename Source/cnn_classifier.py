# SOMA Signal Fidelity Investigation
# Stage 3c -- Basic 1D-CNN
# File: Source/cnn_classifier.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from physics_model import generate_signal, stiffness_classes
from constants import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)


def extract_raw_windows(signal, time, heart_rate_hz, duration=3.0, window_size=100):
    """
    BUG FOUND AND FIXED: the first version aligned each window using
    arrival_t = beat_t + tof, where tof is different per class. That meant
    every window started exactly at the pulse's own true arrival point --
    so within the window, the pulse always began at essentially the same
    position regardless of class, erasing the one thing that actually
    differs between classes (timing) before the CNN ever saw it. Loss stayed
    at ln(3), the exact value for random guessing among 3 classes -- a clear
    sign nothing was being learned.

    FIX: align windows using beat_t alone (the cardiac trigger time), not
    adjusted by tof. Since tof (5-8ms) is small relative to the 200ms
    window, the pulse still lands comfortably inside -- but now at a
    slightly different position depending on class, which is exactly the
    kind of pattern a CNN's convolutions are built to detect.
    """
    beat_period = 1.0 / heart_rate_hz
    beat_times = np.arange(0, duration, beat_period)
    windows = []

    for beat_t in beat_times:
        start_idx = np.searchsorted(time, beat_t)
        end_idx = start_idx + window_size
        if end_idx > len(signal):
            continue
        windows.append(signal[start_idx:end_idx])

    return windows


def build_raw_dataset(n_runs_per_class=30, jitter_std=0.03, seed=SEED):
    np.random.seed(seed)
    X = []
    y = []

    for name, kpa in stiffness_classes.items():
        for _ in range(n_runs_per_class):
            jittered_hr = 1.15 + np.random.normal(0, jitter_std)
            time, signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
            for window in extract_raw_windows(signal, time, jittered_hr):
                X.append(window)
                y.append(name)

    return np.array(X), np.array(y)


def build_raw_dataset_with_groups(n_runs_per_class=30, jitter_std=0.03, seed=SEED):
    """
    Same generation as build_raw_dataset(), but also returns a `groups`
    array identifying which independent 3-second realization each window
    came from. Leakage audit finding: build_raw_dataset()'s flat X/y is
    pooled from ~4 windows per realization, and a plain train_test_split
    on that pool lets windows from the same realization land in both train
    and test. `groups` lets callers use GroupShuffleSplit instead, so the
    split is by realization, not by window.
    """
    np.random.seed(seed)
    X = []
    y = []
    groups = []
    group_id = 0

    for name, kpa in stiffness_classes.items():
        for _ in range(n_runs_per_class):
            jittered_hr = 1.15 + np.random.normal(0, jitter_std)
            time, signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
            windows = extract_raw_windows(signal, time, jittered_hr)
            for window in windows:
                X.append(window)
                y.append(name)
                groups.append(group_id)
            group_id += 1

    return np.array(X), np.array(y), np.array(groups)


class SimpleCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(16 * 25, 32)
        self.fc2 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


if __name__ == "__main__":
    print("Building raw-window dataset (grouped by realization)...")
    X, y, groups = build_raw_dataset_with_groups()
    print(f"  Total samples: {len(X)}   Total independent realizations: {len(np.unique(groups))}")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    print(f"  Classes: {list(encoder.classes_)}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    print(f"  Independent realizations -- train: {len(np.unique(groups[train_idx]))}   "
          f"test: {len(np.unique(groups[test_idx]))}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = SimpleCNN(n_classes=len(encoder.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 60
    print(f"\nTraining for {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_preds = model(X_train_t).argmax(dim=1)
                train_acc = (train_preds == y_train_t).float().mean().item()
                test_preds = model(X_test_t).argmax(dim=1)
                test_acc = (test_preds == y_test_t).float().mean().item()
            print(f"  Epoch {epoch+1:3d}   loss={loss.item():.4f}   "
                  f"train_acc={train_acc:.2%}   test_acc={test_acc:.2%}")

    model.eval()
    with torch.no_grad():
        final_preds = model(X_test_t).argmax(dim=1)
        final_acc = (final_preds == y_test_t).float().mean().item()
    print(f"\nFinal CNN test accuracy: {final_acc:.2%}")
    print(f"Baseline (hand-picked features) accuracy for comparison: 70.37% (single split), 71.59% (cross-validated)")

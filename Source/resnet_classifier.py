# SOMA Signal Fidelity Investigation
# Stage 3e -- Industry-validated CNN architecture and workflow
# File: Source/resnet_classifier.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from cnn_classifier import build_raw_dataset_with_groups
from constants import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.stem_conv = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.stem_bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

        self.block1 = ResidualBlock1D(16)
        self.block2 = ResidualBlock1D(16)

        self.dropout = nn.Dropout(0.3)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.pool(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)


if __name__ == "__main__":
    print("Building raw-window dataset (grouped by realization)...")
    X, y, groups = build_raw_dataset_with_groups()
    print(f"  Total samples: {len(X)}   Total independent realizations: {len(np.unique(groups))}")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = list(encoder.classes_)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    train_idx, temp_idx = next(gss1.split(X, y_encoded, groups=groups))
    X_train, y_train = X[train_idx], y_encoded[train_idx]
    X_temp, y_temp, groups_temp = X[temp_idx], y_encoded[temp_idx], groups[temp_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss2.split(X_temp, y_temp, groups=groups_temp))
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    print(f"  Train: {len(X_train)}   Validation: {len(X_val)}   Test: {len(X_test)}")
    print(f"  Independent realizations -- train: {len(np.unique(groups[train_idx]))}   "
          f"validation: {len(np.unique(groups_temp[val_idx]))}   "
          f"test: {len(np.unique(groups_temp[test_idx]))}")

    def to_tensor(X_arr, y_arr):
        return (torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1),
                torch.tensor(y_arr, dtype=torch.long))

    X_train_t, y_train_t = to_tensor(X_train, y_train)
    X_val_t, y_val_t = to_tensor(X_val, y_val)
    X_test_t, y_test_t = to_tensor(X_test, y_test)

    model = ResNet1D(n_classes=len(class_names))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    max_epochs = 150
    patience = 15
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    print(f"\nTraining (max {max_epochs} epochs, early stopping patience={patience})...")
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            val_acc = (val_outputs.argmax(dim=1) == y_val_t).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}   train_loss={loss.item():.4f}   "
                  f"val_loss={val_loss:.4f}   val_acc={val_acc:.2%}")

        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(no val improvement for {patience} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).argmax(dim=1).numpy()

    test_acc = accuracy_score(y_test, test_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, test_preds, labels=range(len(class_names))
    )

    print(f"\nFinal test accuracy: {test_acc:.2%}")
    print(f"\nPer-class metrics:")
    print(f"{'Class':22s}{'Precision':>12s}{'Recall':>12s}{'F1':>12s}{'Support':>10s}")
    for i, name in enumerate(class_names):
        print(f"{name:22s}{precision[i]:12.2%}{recall[i]:12.2%}{f1[i]:12.2%}{support[i]:10d}")

    print(f"\nComparison:")
    print(f"  Baseline (hand-picked features, cross-validated): 71.59%")
    print(f"  Basic 1D-CNN (Stage 3c):                           88.89%")
    print(f"  ResNet1D (Stage 3e):                               {test_acc:.2%}")

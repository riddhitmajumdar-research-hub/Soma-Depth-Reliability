# Stage 3e (retroactive addition) -- ROC and Precision-Recall curves
# File: Source/stage3e_roc_pr.py

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from cnn_classifier import build_raw_dataset
from resnet_classifier import ResNet1D
from constants import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

X, y = build_raw_dataset()
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = list(encoder.classes_)
n_classes = len(class_names)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

def to_tensor(X_arr, y_arr):
    return (torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1),
            torch.tensor(y_arr, dtype=torch.long))

X_train_t, y_train_t = to_tensor(X_train, y_train)
X_val_t, y_val_t = to_tensor(X_val, y_val)
X_test_t, y_test_t = to_tensor(X_test, y_test)

model = ResNet1D(n_classes=n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

best_val_loss, no_improve, best_state = float("inf"), 0, None
for epoch in range(150):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t), y_val_t).item()
    if val_loss < best_val_loss:
        best_val_loss, no_improve = val_loss, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        no_improve += 1
    if no_improve >= 15:
        break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.softmax(logits, dim=1).numpy()

y_test_onehot = np.eye(n_classes)[y_test]
colors = {"healthy": "#0F6E56", "significant_fibrosis": "#BA7517", "cirrhosis": "#993C1D"}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for i, name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_onehot[:, i], probs[:, i])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=colors[name], label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="chance")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC curves (one-vs-rest)")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

for i, name in enumerate(class_names):
    precision, recall, _ = precision_recall_curve(y_test_onehot[:, i], probs[:, i])
    axes[1].plot(recall, precision, color=colors[name], label=name, linewidth=2)
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall curves (one-vs-rest)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.suptitle("Stage 3e -- ROC and Precision-Recall curves, ResNet1D")
plt.tight_layout()
plt.savefig("stage3e_roc_pr_curves.png", dpi=200, bbox_inches="tight")
plt.show()

print("Per-class AUC:")
for i, name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_test_onehot[:, i], probs[:, i])
    print(f"  {name:22s} AUC = {auc(fpr, tpr):.4f}")

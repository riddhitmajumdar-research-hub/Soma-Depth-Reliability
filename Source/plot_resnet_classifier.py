# Stage 3e -- ResNet1D result visualization
# File: Source/plot_resnet_classifier.py

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from cnn_classifier import build_raw_dataset
from resnet_classifier import ResNet1D
from constants import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

X, y = build_raw_dataset()
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = list(encoder.classes_)

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

model = ResNet1D(n_classes=len(class_names))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

max_epochs, patience = 150, 15
best_val_loss, epochs_without_improvement, best_state, stop_epoch = float("inf"), 0, None, None
train_losses, val_losses, val_accs = [], [], []

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

    train_losses.append(loss.item())
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        stop_epoch = epoch + 1
        break

model.load_state_dict(best_state)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(train_losses, label="train loss", color="#0F6E56")
axes[0].plot(val_losses, label="val loss", color="#BA7517")
if stop_epoch:
    axes[0].axvline(x=len(val_losses) - patience, color="gray", linestyle="--",
                     linewidth=0.8, label="best checkpoint")
axes[0].set_title("Loss curves (early stopping)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(val_accs, color="#0F6E56")
axes[1].set_title("Validation accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].grid(True, alpha=0.3)

plt.suptitle("Stage 3e -- ResNet1D training curves")
plt.tight_layout()
plt.savefig("stage3e_training_curves.png", dpi=200, bbox_inches="tight")
plt.show()

model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1).numpy()

cm = confusion_matrix(y_test, preds, labels=range(len(class_names)))
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="YlOrRd")
ax.set_xticks(range(len(class_names)))
ax.set_yticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=30, ha="right")
ax.set_yticklabels(class_names)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Stage 3e -- ResNet1D confusion matrix (test set)")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im, ax=ax, label="Count")
plt.tight_layout()
plt.savefig("stage3e_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, preds, labels=range(len(class_names))
)
x = np.arange(len(class_names))
width = 0.25
fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x - width, precision, width, label="Precision", color="#0F6E56")
ax.bar(x, recall, width, label="Recall", color="#BA7517")
ax.bar(x + width, f1, width, label="F1", color="#993C1D")
ax.set_xticks(x)
ax.set_xticklabels(class_names)
ax.set_ylabel("Score")
ax.set_title("Stage 3e -- Per-class precision, recall, F1")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("stage3e_per_class_metrics.png", dpi=200, bbox_inches="tight")
plt.show()

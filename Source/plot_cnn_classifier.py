# Stage 3c -- CNN result visualization
# File: Source/plot_cnn_classifier.py

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from cnn_classifier import build_raw_dataset, SimpleCNN
from constants import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

X, y = build_raw_dataset()
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = list(encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_train, dtype=torch.long)
y_test_t = torch.tensor(y_test, dtype=torch.long)

model = SimpleCNN(n_classes=len(class_names))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

n_epochs = 60
losses, train_accs, test_accs = [], [], []

for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train_t).argmax(dim=1) == y_train_t).float().mean().item()
        test_acc = (model(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()

    losses.append(loss.item())
    train_accs.append(train_acc)
    test_accs.append(test_acc)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(losses, color="#0F6E56")
axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(train_accs, label="train", color="#0F6E56")
axes[1].plot(test_accs, label="test", color="#993C1D")
axes[1].set_title("Train vs. test accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Stage 3c -- Basic 1D-CNN training curves")
plt.tight_layout()
plt.savefig("stage3c_training_curves.png", dpi=200, bbox_inches="tight")
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
ax.set_title("Stage 3c -- CNN confusion matrix (test set)")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im, ax=ax, label="Count")
plt.tight_layout()
plt.savefig("stage3c_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

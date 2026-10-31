# Stage 3 — Classifier result visualization
# File: Source/plot_classifier.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from physics_model import stiffness_classes
from source_classifier import build_dataset

X, y = build_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

colors = {"healthy": "#0F6E56", "significant_fibrosis": "#BA7517", "cirrhosis": "#993C1D"}

plt.figure(figsize=(10, 6))
for label in sorted(stiffness_classes.keys()):
    values = X[y == label][:, 0] * 1000
    plt.hist(values, bins=20, alpha=0.6, label=label, color=colors[label])

plt.xlabel("Time in cycle (ms)")
plt.ylabel("Count")
plt.title("Distribution of arrival-time feature by stiffness class")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("stage3_feature_distribution.png", dpi=200, bbox_inches="tight")
plt.show()

labels = sorted(stiffness_classes.keys())
cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="YlOrRd")

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
ax.set_title("Confusion matrix - baseline classifier on clean signals")

for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.colorbar(im, ax=ax, label="Count")
plt.tight_layout()
plt.savefig("stage3_confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.show()

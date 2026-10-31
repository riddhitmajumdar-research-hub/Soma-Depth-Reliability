# Stage 3f (retroactive addition) -- Saliency maps and feature map visualization
# File: Source/stage3f_saliency_featuremaps.py

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    test_preds = model(X_test_t).argmax(dim=1).numpy()

colors = {"healthy": "#0F6E56", "significant_fibrosis": "#BA7517", "cirrhosis": "#993C1D"}


def compute_saliency(x_tensor, true_label):
    x_tensor = x_tensor.clone().requires_grad_(True)
    output = model(x_tensor)
    score = output[0, true_label]
    score.backward()
    saliency = x_tensor.grad.abs().squeeze().numpy()
    return saliency


fig1, axes1 = plt.subplots(3, 1, figsize=(11, 9))
for ax, class_idx in zip(axes1, range(len(class_names))):
    class_name = class_names[class_idx]
    correct_idx = np.where((y_test == class_idx) & (test_preds == class_idx))[0][0]
    x_example = X_test_t[correct_idx:correct_idx+1]
    saliency = compute_saliency(x_example, class_idx)
    signal = X_test[correct_idx]

    ax2 = ax.twinx()
    ax.plot(signal, color=colors[class_name], linewidth=1.3, label="signal")
    ax2.fill_between(range(100), saliency, alpha=0.3, color="gray", label="saliency")
    ax.set_title(f"{class_name} -- vanilla saliency (gradient w.r.t. raw input)")
    ax.set_ylabel("Amplitude")
    ax2.set_ylabel("Saliency")
    ax.legend(loc="upper right", fontsize=8)

axes1[-1].set_xlabel("Sample index (within window)")
plt.suptitle("Stage 3f (addition) -- Vanilla saliency maps, an independent check on Grad-CAM")
plt.tight_layout()
plt.savefig("stage3f_saliency_maps.png", dpi=200, bbox_inches="tight")
plt.show()

activations = {}
def hook(module, input, output):
    activations["value"] = output
model.stem_conv.register_forward_hook(hook)

example_idx = np.where((y_test == 0) & (test_preds == 0))[0][0]
x_example = X_test_t[example_idx:example_idx+1]
with torch.no_grad():
    model(x_example)
feature_maps = activations["value"].squeeze().numpy()

fig2, ax = plt.subplots(figsize=(11, 6))
im = ax.imshow(feature_maps, aspect="auto", cmap="viridis",
                extent=[0, 100, feature_maps.shape[0], 0])
ax.set_xlabel("Sample index (within window)")
ax.set_ylabel("Channel (filter)")
ax.set_title(f"Stage 3f (addition) -- First conv layer feature maps "
              f"(16 filters, example: {class_names[0]})")
plt.colorbar(im, ax=ax, label="Activation")
plt.tight_layout()
plt.savefig("stage3f_feature_maps.png", dpi=200, bbox_inches="tight")
plt.show()

print("Both figures saved.")

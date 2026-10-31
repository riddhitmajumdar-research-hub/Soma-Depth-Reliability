# SOMA Signal Fidelity Investigation
# Stage 3f -- CNN interpretability (Grad-CAM)
# File: Source/gradcam_resnet.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

best_val_loss, epochs_without_improvement, best_state = float("inf"), 0, None
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
        best_val_loss = val_loss
        epochs_without_improvement = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= 15:
        break

model.load_state_dict(best_state)
model.eval()
print("Model retrained to match committed Stage 3e result.")

activations = {}
gradients = {}

def forward_hook(module, input, output):
    activations["value"] = output

def backward_hook(module, grad_input, grad_output):
    gradients["value"] = grad_output[0]

model.block2.register_forward_hook(forward_hook)
model.block2.register_full_backward_hook(backward_hook)


def compute_gradcam(x_tensor, true_label):
    model.zero_grad()
    output = model(x_tensor)
    score = output[0, true_label]
    score.backward()

    acts = activations["value"][0]
    grads = gradients["value"][0]

    weights = grads.mean(dim=1)
    cam = F.relu((weights[:, None] * acts).sum(dim=0))
    cam_raw = cam.detach().numpy()
    cam_raw = cam_raw / (cam_raw.max() + 1e-8)

    cam_upsampled = np.interp(
        np.linspace(0, len(cam_raw) - 1, 100),
        np.arange(len(cam_raw)),
        cam_raw
    )
    return cam_raw, cam_upsampled


model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).argmax(dim=1).numpy()

fig, axes = plt.subplots(3, 1, figsize=(11, 9))
colors = {"healthy": "#0F6E56", "significant_fibrosis": "#BA7517", "cirrhosis": "#993C1D"}

print("\nComparing Grad-CAM attention peak to the raw signal's own peak position:")
for ax, class_idx in zip(axes, range(len(class_names))):
    class_name = class_names[class_idx]
    correct_idx = np.where((y_test == class_idx) & (test_preds == class_idx))[0][0]

    x_example = X_test_t[correct_idx:correct_idx+1]
    cam_raw, cam = compute_gradcam(x_example, class_idx)

    signal = X_test[correct_idx]
    signal_peak_pos = np.argmax(np.abs(signal))
    cam_peak_pos = np.argmax(cam)

    ax2 = ax.twinx()
    ax.plot(signal, color=colors[class_name], linewidth=1.3, label="signal")
    ax2.fill_between(range(100), cam, alpha=0.3, color="gray", label="Grad-CAM attention")
    ax.axvline(signal_peak_pos, color="black", linestyle="--", linewidth=0.8,
               label=f"signal peak (sample {signal_peak_pos})")

    ax.set_title(f"{class_name} -- Grad-CAM peak at sample {cam_peak_pos}, "
                 f"signal peak at sample {signal_peak_pos}")
    ax.set_ylabel("Amplitude")
    ax2.set_ylabel("Attention")
    ax.legend(loc="upper right", fontsize=8)

    print(f"  {class_name:22s} Grad-CAM peak: sample {cam_peak_pos:3d}   "
          f"Signal peak: sample {signal_peak_pos:3d}   "
          f"Difference: {abs(cam_peak_pos - signal_peak_pos)} samples")

axes[-1].set_xlabel("Sample index (within window)")
plt.suptitle("Stage 3f -- Grad-CAM: where the ResNet1D actually looks")
plt.tight_layout()
plt.savefig("figures/stage3f_gradcam.png", dpi=200, bbox_inches="tight")
plt.show()

fig2, axes2 = plt.subplots(3, 1, figsize=(11, 8))

for ax, class_idx in zip(axes2, range(len(class_names))):
    class_name = class_names[class_idx]
    correct_idx = np.where((y_test == class_idx) & (test_preds == class_idx))[0][0]
    x_example = X_test_t[correct_idx:correct_idx+1]
    cam_raw, _ = compute_gradcam(x_example, class_idx)

    block_starts = np.arange(len(cam_raw)) * 4
    ax.bar(block_starts, cam_raw, width=4, align="edge",
           color=colors[class_name], alpha=0.7, edgecolor="white")
    signal = X_test[correct_idx]
    signal_peak_pos = np.argmax(np.abs(signal))
    ax.axvline(signal_peak_pos, color="black", linestyle="--", linewidth=0.8,
               label=f"signal peak (sample {signal_peak_pos})")
    ax.set_title(f"{class_name} -- native Grad-CAM resolution (25 blocks of ~4 samples each)")
    ax.set_ylabel("Attention")
    ax.legend(loc="upper right", fontsize=8)

axes2[-1].set_xlabel("Sample index (within window)")
plt.suptitle("Stage 3f -- Grad-CAM's real resolution, before upsampling smooths it")
plt.tight_layout()
plt.savefig("figures/stage3f_gradcam_resolution.png", dpi=200, bbox_inches="tight")
plt.show()

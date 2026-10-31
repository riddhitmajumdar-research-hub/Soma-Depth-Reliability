# SOMA Signal Fidelity Investigation
# Part 3(d) -- Grad-CAM counterfactual occlusion test.
#
# DISCLOSURE: this result is run against Log 08's ORIGINAL ResNet1D model
# (models/stage3e_resnet1d_model.pt if present, else retrained fresh here
# to match the committed Stage 3e result -- IDENTICAL setup to
# gradcam_resnet.py: clean signals, no degradation-aware training, no
# corrected noise model). This is NOT the certified degradation-aware
# model, and is NOT the in-progress 24-seed ResNet1D retest. Flagged as
# pending reconfirmation against the final certified ResNet1D once that
# run completes, exactly as done for the 3.4 per-class breakdown.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from cnn_classifier import build_raw_dataset
from resnet_classifier import ResNet1D
from constants import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

CSV_PATH = "logs/gradcam_counterfactual_occlusion.csv"

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
print("Model retrained to match committed Stage 3e / Log 08 result (clean signals, no degradation-aware training).")

WINDOW_SIZE = 100
OCCLUSION_LEN = 20
NEGATIVE_CONTROL_START = 60  # region with no expected signal, per Grad-CAM feature-map evidence


def occlude_zero(x, start, length):
    x2 = x.clone()
    x2[:, :, start:start + length] = 0.0
    return x2


def occlude_randomize(x, start, length, rng_seed):
    torch.manual_seed(rng_seed)
    x2 = x.clone()
    noise = torch.randn(x2[:, :, start:start + length].shape) * x2.std()
    x2[:, :, start:start + length] = noise
    return x2


def shift_pulse(x, shift_samples):
    """Shifts the window's content along the sample axis, preserving amplitude/shape."""
    x2 = torch.zeros_like(x)
    n = x.shape[2]
    if shift_samples >= 0:
        x2[:, :, shift_samples:] = x[:, :, :n - shift_samples]
    else:
        x2[:, :, :n + shift_samples] = x[:, :, -shift_samples:]
    return x2


with torch.no_grad():
    test_preds = model(X_test_t).argmax(dim=1).numpy()

rows = []
print("\n--- (i) Zero/randomize first 20 samples vs uninformative-region negative control ---")
for class_idx in range(len(class_names)):
    class_name = class_names[class_idx]
    idxs = np.where((y_test == class_idx) & (test_preds == class_idx))[0]
    if len(idxs) == 0:
        continue
    for trial_i, correct_idx in enumerate(idxs[:5]):  # up to 5 correctly-classified examples per class
        x_example = X_test_t[correct_idx:correct_idx + 1]

        with torch.no_grad():
            baseline_probs = F.softmax(model(x_example), dim=1)[0]
        baseline_true_prob = baseline_probs[class_idx].item()

        with torch.no_grad():
            zeroed = occlude_zero(x_example, 0, OCCLUSION_LEN)
            probs_zeroed = F.softmax(model(zeroed), dim=1)[0]
        drop_zeroed = baseline_true_prob - probs_zeroed[class_idx].item()

        with torch.no_grad():
            randomized = occlude_randomize(x_example, 0, OCCLUSION_LEN, rng_seed=1000 + trial_i)
            probs_randomized = F.softmax(model(randomized), dim=1)[0]
        drop_randomized = baseline_true_prob - probs_randomized[class_idx].item()

        with torch.no_grad():
            neg_control = occlude_zero(x_example, NEGATIVE_CONTROL_START, OCCLUSION_LEN)
            probs_neg_control = F.softmax(model(neg_control), dim=1)[0]
        drop_neg_control = baseline_true_prob - probs_neg_control[class_idx].item()

        rows.append({
            "class": class_name, "trial": trial_i, "test_idx": int(correct_idx),
            "baseline_true_class_prob": baseline_true_prob,
            "prob_drop_zero_first20": drop_zeroed,
            "prob_drop_randomize_first20": drop_randomized,
            "prob_drop_negative_control": drop_neg_control,
            "zero_larger_than_control": drop_zeroed > drop_neg_control,
            "randomize_larger_than_control": drop_randomized > drop_neg_control,
        })

print(f"{'class':>22} {'trial':>5} {'baseline_p':>11} {'drop_zero':>10} {'drop_rand':>10} {'drop_ctrl':>10}")
for r in rows:
    print(f"{r['class']:>22} {r['trial']:>5} {r['baseline_true_class_prob']:>11.4f} "
          f"{r['prob_drop_zero_first20']:>10.4f} {r['prob_drop_randomize_first20']:>10.4f} "
          f"{r['prob_drop_negative_control']:>10.4f}")

mean_drop_zero = np.mean([r["prob_drop_zero_first20"] for r in rows])
mean_drop_rand = np.mean([r["prob_drop_randomize_first20"] for r in rows])
mean_drop_ctrl = np.mean([r["prob_drop_negative_control"] for r in rows])
print(f"\nMean prob drop -- zero first20: {mean_drop_zero:.4f}  randomize first20: {mean_drop_rand:.4f}  "
      f"negative control: {mean_drop_ctrl:.4f}")
print(f"Zero-occlusion drop larger than negative control: {mean_drop_zero > mean_drop_ctrl}")
print(f"Randomize-occlusion drop larger than negative control: {mean_drop_rand > mean_drop_ctrl}")

print("\n--- (ii) Timing-shift test: shift pulse position, preserve amplitude/shape ---")
shift_rows = []
SHIFT_AMOUNTS = [-15, -10, 10, 15]
for class_idx in range(len(class_names)):
    class_name = class_names[class_idx]
    idxs = np.where((y_test == class_idx) & (test_preds == class_idx))[0]
    if len(idxs) == 0:
        continue
    for trial_i, correct_idx in enumerate(idxs[:5]):
        x_example = X_test_t[correct_idx:correct_idx + 1]
        with torch.no_grad():
            baseline_pred = model(x_example).argmax(dim=1).item()
        for shift in SHIFT_AMOUNTS:
            shifted = shift_pulse(x_example, shift)
            with torch.no_grad():
                shifted_pred = model(shifted).argmax(dim=1).item()
            shift_rows.append({
                "class": class_name, "trial": trial_i, "test_idx": int(correct_idx),
                "shift_samples": shift,
                "baseline_pred": class_names[baseline_pred],
                "shifted_pred": class_names[shifted_pred],
                "prediction_changed": baseline_pred != shifted_pred,
            })

n_changed = sum(1 for r in shift_rows if r["prediction_changed"])
print(f"Timing shifts tested: {len(shift_rows)}  |  predicted class changed: {n_changed}/{len(shift_rows)} "
      f"({100*n_changed/len(shift_rows):.1f}%)")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved occlusion results to {CSV_PATH}")

with open("logs/gradcam_timing_shift_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(shift_rows[0].keys()))
    writer.writeheader()
    writer.writerows(shift_rows)
print("Saved timing-shift results to logs/gradcam_timing_shift_results.csv")

print("\n*** DISCLOSURE: run against Log 08's ORIGINAL ResNet1D model (clean-trained, ")
print("pre-degradation-training, pre-noise-correction) -- NOT the certified degradation-aware")
print("model, and NOT the in-progress 24-seed ResNet1D retest. Pending reconfirmation once ")
print("that run completes. ***")

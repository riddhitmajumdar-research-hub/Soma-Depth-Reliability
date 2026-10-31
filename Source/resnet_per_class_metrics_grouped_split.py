# SOMA Signal Fidelity Investigation
# Part F5 -- ResNet1D per-class precision/recall/F1/ROC-AUC under the
# CORRECTED grouped split (logs/grouped_split_leakage_fix_results.csv
# recorded only the aggregate 86.11% for this split; the file is
# untracked in git with no producing script or commit history -- confirmed
# via `git log --all --follow` on the CSV itself, genuinely unrecoverable).
#
# RECONCILIATION NOTE: an earlier reconstruction using nested
# GroupShuffleSplit(test_size=0.2) then GroupShuffleSplit(test_size=0.25)
# reproduced the recorded 54/18/18 REALIZATION COUNTS exactly, but scored
# 88.89% test accuracy, not the recorded 86.11% -- a 2.78pp gap. Isolated
# the cause by first confirming the CNN side of this same CSV
# (corrected_accuracy=0.8981) reconstructs to 0.89815 EXACTLY using
# GroupShuffleSplit(test_size=0.3, random_state=42) directly on
# build_raw_dataset_with_groups() -- confirming dataset generation, seeding,
# and split methodology are correct in general. The gap was therefore
# specific to ResNet1D's train/val/test group partition: different
# GroupShuffleSplit call sequences can produce the SAME final group
# counts (54/18/18) from DIFFERENT specific group-to-split assignments,
# since GroupShuffleSplit's internal permutation for a given random_state
# depends on which array (and length) it's permuting, not just the final
# proportions. Testing gradcam_resnet.py's EXACT split proportions --
# the only other 3-way train/val/test ResNet1D pipeline already in this
# repo (test_size=0.4 for train vs temp, then test_size=0.5 on temp for
# val/test), made group-aware -- reproduced the recorded 0.8611 EXACTLY.
# This is therefore almost certainly the actual split the original script
# used (very likely a grouped variant of gradcam_resnet.py itself), not
# my initially-assumed nested 0.2/0.25 split. Using that split below.

import numpy as np
import torch
import torch.nn as nn
import csv
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from cnn_classifier import build_raw_dataset_with_groups
from resnet_classifier import ResNet1D
from constants import SEED

torch.manual_seed(SEED)
np.random.seed(SEED)

X, y, groups = build_raw_dataset_with_groups()
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_names = list(encoder.classes_)

gss1 = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(gss1.split(X, y_encoded, groups=groups))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_rel_idx, test_rel_idx = next(gss2.split(
    X[temp_idx], y_encoded[temp_idx], groups=groups[temp_idx]))
val_idx = temp_idx[val_rel_idx]
test_idx = temp_idx[test_rel_idx]

n_train_groups = len(np.unique(groups[train_idx]))
n_val_groups = len(np.unique(groups[val_idx]))
n_test_groups = len(np.unique(groups[test_idx]))
print(f"Realization counts -- train:{n_train_groups} val:{n_val_groups} test:{n_test_groups} "
      f"(recorded: 54/18/18)")
assert (n_train_groups, n_val_groups, n_test_groups) == (54, 18, 18), \
    "Split does not reproduce the recorded realization counts -- stop and report, do not proceed silently."

X_train_t = torch.tensor(X[train_idx], dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_encoded[train_idx], dtype=torch.long)
X_val_t = torch.tensor(X[val_idx], dtype=torch.float32).unsqueeze(1)
y_val_t = torch.tensor(y_encoded[val_idx], dtype=torch.long)
X_test_t = torch.tensor(X[test_idx], dtype=torch.float32).unsqueeze(1)
y_test = y_encoded[test_idx]

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

with torch.no_grad():
    logits = model(X_test_t)
    probs = torch.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()

overall_acc = np.mean(preds == y_test)
print(f"\nOverall test accuracy: {overall_acc:.4f}  (recorded corrected_accuracy: 0.8611)")

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, preds, labels=range(len(class_names)), zero_division=0
)

rows = []
print(f"\n{'class':>22}{'precision':>12}{'recall':>10}{'f1':>10}{'support':>10}{'roc_auc':>10}")
for i, name in enumerate(class_names):
    y_true_binary = (y_test == i).astype(int)
    try:
        auc = roc_auc_score(y_true_binary, probs[:, i])
    except ValueError:
        auc = float("nan")  # only one class present in this tiny test fold for this label
    print(f"{name:>22}{precision[i]:>12.4f}{recall[i]:>10.4f}{f1[i]:>10.4f}{support[i]:>10d}{auc:>10.4f}")
    rows.append({
        "class": name, "precision": precision[i], "recall": recall[i],
        "f1": f1[i], "support": int(support[i]), "roc_auc_ovr": auc,
    })

with open("logs/resnet_per_class_metrics_grouped_split.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print("\nSaved to logs/resnet_per_class_metrics_grouped_split.csv")

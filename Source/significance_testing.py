# SOMA Signal Fidelity Investigation
# Stage 3h -- Statistical significance testing
# File: Source/significance_testing.py

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import stats
import matplotlib.pyplot as plt

from source_classifier import build_dataset
from cnn_classifier import build_raw_dataset, SimpleCNN
from resnet_classifier import ResNet1D
from constants import SEED

np.random.seed(SEED)
torch.manual_seed(SEED)

N_FOLDS = 5

X_feat, y_feat = build_dataset()
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

baseline_accs = []
for train_idx, test_idx in skf.split(X_feat, y_feat):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_feat[train_idx])
    X_test = scaler.transform(X_feat[test_idx])
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_feat[train_idx])
    baseline_accs.append(accuracy_score(y_feat[test_idx], model.predict(X_test)))

print("Baseline fold accuracies:", [f"{a:.2%}" for a in baseline_accs])

X_raw, y_raw = build_raw_dataset()
encoder = LabelEncoder()
y_raw_enc = encoder.fit_transform(y_raw)

skf2 = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
cnn_accs = []
resnet_accs = []

for fold_num, (train_idx, test_idx) in enumerate(skf2.split(X_raw, y_raw_enc), start=1):
    X_train_t = torch.tensor(X_raw[train_idx], dtype=torch.float32).unsqueeze(1)
    y_train_t = torch.tensor(y_raw_enc[train_idx], dtype=torch.long)
    X_test_t = torch.tensor(X_raw[test_idx], dtype=torch.float32).unsqueeze(1)
    y_test_t = torch.tensor(y_raw_enc[test_idx], dtype=torch.long)

    cnn = SimpleCNN(n_classes=3)
    opt = torch.optim.Adam(cnn.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for epoch in range(60):
        cnn.train()
        opt.zero_grad()
        loss = crit(cnn(X_train_t), y_train_t)
        loss.backward()
        opt.step()
    cnn.eval()
    with torch.no_grad():
        acc = (cnn(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
    cnn_accs.append(acc)

    from sklearn.model_selection import train_test_split as _tts
    train_sub, val_sub = _tts(
        train_idx, test_size=0.2, random_state=42,
        stratify=y_raw_enc[train_idx]
    )
    X_tr = torch.tensor(X_raw[train_sub], dtype=torch.float32).unsqueeze(1)
    y_tr = torch.tensor(y_raw_enc[train_sub], dtype=torch.long)
    X_vl = torch.tensor(X_raw[val_sub], dtype=torch.float32).unsqueeze(1)
    y_vl = torch.tensor(y_raw_enc[val_sub], dtype=torch.long)

    max_restarts = 5
    for attempt in range(max_restarts):
        resnet = ResNet1D(n_classes=3)
        opt2 = torch.optim.Adam(resnet.parameters(), lr=0.001)
        best_val_loss, no_improve, best_state = float("inf"), 0, None
        for epoch in range(150):
            resnet.train()
            opt2.zero_grad()
            loss = crit(resnet(X_tr), y_tr)
            loss.backward()
            opt2.step()
            resnet.eval()
            with torch.no_grad():
                val_loss = crit(resnet(X_vl), y_vl).item()
            if val_loss < best_val_loss:
                best_val_loss, no_improve = val_loss, 0
                best_state = {k: v.clone() for k, v in resnet.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= 15:
                break
        resnet.load_state_dict(best_state)
        resnet.eval()
        with torch.no_grad():
            train_acc_check = (resnet(X_tr).argmax(dim=1) == y_tr).float().mean().item()

        if train_acc_check > 0.5:
            break
        else:
            print(f"    [Fold {fold_num}] ResNet1D collapsed on attempt {attempt+1} "
                  f"(train_acc={train_acc_check:.2%}) -- restarting with new initialization")

    with torch.no_grad():
        acc = (resnet(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
    resnet_accs.append(acc)

    print(f"  Fold {fold_num}: CNN={cnn_accs[-1]:.2%}  ResNet1D={resnet_accs[-1]:.2%}")

print("\nFull fold-by-fold comparison:")
print(f"{'Fold':6s}{'Baseline':>12s}{'CNN':>12s}{'ResNet1D':>12s}")
for i in range(N_FOLDS):
    print(f"{i+1:<6d}{baseline_accs[i]:>12.2%}{cnn_accs[i]:>12.2%}{resnet_accs[i]:>12.2%}")

def paired_test(name_a, a, name_b, b):
    diffs = np.array(b) - np.array(a)
    t_stat, p_val = stats.ttest_rel(b, a)
    ci = stats.t.interval(0.95, len(diffs)-1, loc=diffs.mean(),
                            scale=stats.sem(diffs)) if len(diffs) > 1 else (np.nan, np.nan)
    print(f"\n{name_b} vs {name_a}:")
    print(f"  Mean difference: {diffs.mean():.2%}  (95% CI: {ci[0]:.2%} to {ci[1]:.2%})")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"  {'Statistically significant (p < 0.05)' if p_val < 0.05 else 'NOT statistically significant at p < 0.05'}")

print("\n" + "="*60)
print("PAIRED T-TESTS (n=5 folds -- interpret with appropriate caution)")
print("="*60)
paired_test("Baseline", baseline_accs, "CNN", cnn_accs)
paired_test("Baseline", baseline_accs, "ResNet1D", resnet_accs)
paired_test("CNN", cnn_accs, "ResNet1D", resnet_accs)

methods = ["Baseline", "Basic CNN", "ResNet1D"]
all_accs = [baseline_accs, cnn_accs, resnet_accs]
colors = ["#BA7517", "#0F6E56", "#993C1D"]

means = [np.mean(a) for a in all_accs]
cis = [stats.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=stats.sem(a)) for a in all_accs]
errs = [[means[i]-cis[i][0] for i in range(3)], [cis[i][1]-means[i] for i in range(3)]]

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(3)
ax.bar(x, means, yerr=errs, capsize=8, color=colors, alpha=0.75, edgecolor="black", linewidth=0.8)

for i, accs in enumerate(all_accs):
    jitter = np.random.uniform(-0.08, 0.08, len(accs))
    ax.scatter(x[i] + jitter, accs, color="black", s=25, zorder=5, label="individual folds" if i == 0 else None)

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Accuracy")
ax.set_title("Stage 3h -- Method comparison, 5-fold paired (mean \u00b1 95% CI)")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("stage3h_significance_comparison.png", dpi=200, bbox_inches="tight")
plt.show()

fig2, ax2 = plt.subplots(figsize=(9, 6))
x2 = np.arange(3)
for fold_idx in range(N_FOLDS):
    fold_scores = [baseline_accs[fold_idx], cnn_accs[fold_idx], resnet_accs[fold_idx]]
    ax2.plot(x2, fold_scores, marker="o", color="#0F6E56", alpha=0.6, linewidth=1.5)
    ax2.annotate(f"fold {fold_idx+1}", (x2[-1], fold_scores[-1]),
                  textcoords="offset points", xytext=(6, 0), fontsize=8, color="#444")

ax2.set_xticks(x2)
ax2.set_xticklabels(methods)
ax2.set_ylabel("Accuracy")
ax2.set_title("Stage 3h -- Paired trajectory: each fold's score across methods")
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("stage3h_paired_trajectory.png", dpi=200, bbox_inches="tight")
plt.show()

comparisons = [
    ("CNN - Baseline", np.array(cnn_accs) - np.array(baseline_accs)),
    ("ResNet1D - Baseline", np.array(resnet_accs) - np.array(baseline_accs)),
    ("ResNet1D - CNN", np.array(resnet_accs) - np.array(cnn_accs)),
]

fig3, ax3 = plt.subplots(figsize=(9, 6))
for i, (label, diffs) in enumerate(comparisons):
    mean_diff = diffs.mean()
    ci = stats.t.interval(0.95, len(diffs)-1, loc=mean_diff, scale=stats.sem(diffs))
    color = "#0F6E56" if ci[0] > 0 else ("#993C1D" if ci[1] < 0 else "#BA7517")
    ax3.errorbar(i, mean_diff, yerr=[[mean_diff-ci[0]], [ci[1]-mean_diff]],
                  fmt="o", color=color, markersize=10, capsize=8, linewidth=2)
    jitter = np.random.uniform(-0.06, 0.06, len(diffs))
    ax3.scatter(np.full(len(diffs), i) + jitter, diffs, color="black", s=20, alpha=0.6, zorder=5)

ax3.axhline(y=0, color="gray", linestyle="--", linewidth=1)
ax3.set_xticks(range(len(comparisons)))
ax3.set_xticklabels([c[0] for c in comparisons])
ax3.set_ylabel("Accuracy difference")
ax3.set_title("Stage 3h -- Paired differences with 95% CI\n(crosses zero = not statistically significant)")
ax3.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("stage3h_paired_differences.png", dpi=200, bbox_inches="tight")
plt.show()

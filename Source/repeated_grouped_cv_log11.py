# SOMA Signal Fidelity Investigation
# Part 3(a) -- repeated grouped resampling for the Log 11 paired comparisons.
# Addresses n=5 fold fragility: Log 11's original comparison used ONE
# grouped 5-fold split. This script repeats grouped 5-fold CV
# N_REPETITIONS times (different group-to-fold shuffles each time) for
# all three comparisons (baseline logistic regression, CNN, ResNet1D),
# using the same models/training protocol as significance_testing.py
# (including the ResNet1D restart safeguard), and reports a
# sign-flip permutation p-value plus a percentile-bootstrap CI over all
# repetition x fold paired differences, alongside the original single-split
# numbers (logs/log11_grouped_paired_ttests.csv) for comparison.

import numpy as np
import torch
import torch.nn as nn
import csv

from source_classifier import extract_pulses_with_features
from cnn_classifier import build_raw_dataset_with_groups, SimpleCNN
from resnet_classifier import ResNet1D
from physics_model import generate_signal, stiffness_classes
from constants import SEED
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(SEED)
torch.manual_seed(SEED)

N_REPETITIONS = 15
N_FOLDS = 5
N_PERMUTATIONS = 5000
N_BOOTSTRAP = 2000
CSV_PATH = "logs/log11_grouped_paired_ttests.csv"


def build_baseline_dataset_with_groups(n_runs_per_class=30, jitter_std=0.03, seed=SEED):
    np.random.seed(seed)
    X, y, groups = [], [], []
    group_id = 0
    for name, kpa in stiffness_classes.items():
        for _ in range(n_runs_per_class):
            jittered_hr = 1.15 + np.random.normal(0, jitter_std)
            time, signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
            feature_rows = extract_pulses_with_features(signal, time, jittered_hr, tof)
            for features in feature_rows:
                X.append(features)
                y.append(name)
                groups.append(group_id)
            group_id += 1
    return np.array(X), np.array(y), np.array(groups)


def grouped_kfold_indices(groups, n_folds, rep_seed):
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(rep_seed)
    rng.shuffle(unique_groups)
    fold_of_group = {}
    for i, g in enumerate(unique_groups):
        fold_of_group[g] = i % n_folds
    fold_assignment = np.array([fold_of_group[g] for g in groups])
    for fold in range(n_folds):
        test_mask = fold_assignment == fold
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield train_idx, test_idx


def train_cnn(X_train_t, y_train_t, n_epochs=60):
    cnn = SimpleCNN(n_classes=3)
    opt = torch.optim.Adam(cnn.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for _ in range(n_epochs):
        cnn.train()
        opt.zero_grad()
        loss = crit(cnn(X_train_t), y_train_t)
        loss.backward()
        opt.step()
    cnn.eval()
    return cnn


def train_resnet_with_restarts(X_raw, y_raw_enc, train_idx, rep_seed, fold_num):
    from sklearn.model_selection import train_test_split as _tts
    train_sub, val_sub = _tts(
        train_idx, test_size=0.2, random_state=rep_seed * 100 + fold_num,
        stratify=y_raw_enc[train_idx]
    )
    X_tr = torch.tensor(X_raw[train_sub], dtype=torch.float32).unsqueeze(1)
    y_tr = torch.tensor(y_raw_enc[train_sub], dtype=torch.long)
    X_vl = torch.tensor(X_raw[val_sub], dtype=torch.float32).unsqueeze(1)
    y_vl = torch.tensor(y_raw_enc[val_sub], dtype=torch.long)
    crit = nn.CrossEntropyLoss()

    max_restarts = 5
    n_restarts_used = 0
    resnet, best_state = None, None
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
        n_restarts_used += 1
    return resnet, n_restarts_used


def main():
    X_feat, y_feat, groups_feat = build_baseline_dataset_with_groups()
    X_raw, y_raw, groups_raw = build_raw_dataset_with_groups()
    encoder = LabelEncoder()
    y_raw_enc = encoder.fit_transform(y_raw)

    all_diffs = {"cnn_vs_baseline": [], "resnet1d_vs_baseline": [], "resnet1d_vs_cnn": []}
    per_rep_fold_rows = []
    total_restarts = 0

    for rep in range(N_REPETITIONS):
        rep_seed = 1000 + rep
        # baseline (feature-based) grouped folds
        baseline_accs = []
        for train_idx, test_idx in grouped_kfold_indices(groups_feat, N_FOLDS, rep_seed):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_feat[train_idx])
            X_test = scaler.transform(X_feat[test_idx])
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_feat[train_idx])
            baseline_accs.append(accuracy_score(y_feat[test_idx], model.predict(X_test)))

        # raw (CNN/ResNet1D) grouped folds -- SAME group-to-fold mapping shared
        # across CNN/ResNet1D within a repetition (groups_raw has same group
        # structure/order as groups_feat: 30 runs/class x 3 classes)
        cnn_accs, resnet_accs = [], []
        for fold_num, (train_idx, test_idx) in enumerate(
                grouped_kfold_indices(groups_raw, N_FOLDS, rep_seed), start=1):
            X_train_t = torch.tensor(X_raw[train_idx], dtype=torch.float32).unsqueeze(1)
            y_train_t = torch.tensor(y_raw_enc[train_idx], dtype=torch.long)
            X_test_t = torch.tensor(X_raw[test_idx], dtype=torch.float32).unsqueeze(1)
            y_test_t = torch.tensor(y_raw_enc[test_idx], dtype=torch.long)

            cnn = train_cnn(X_train_t, y_train_t)
            with torch.no_grad():
                cnn_acc = (cnn(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
            cnn_accs.append(cnn_acc)

            resnet, n_restarts = train_resnet_with_restarts(
                X_raw, y_raw_enc, train_idx, rep_seed, fold_num)
            total_restarts += n_restarts
            with torch.no_grad():
                resnet_acc = (resnet(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()
            resnet_accs.append(resnet_acc)

        for f in range(N_FOLDS):
            all_diffs["cnn_vs_baseline"].append(cnn_accs[f] - baseline_accs[f])
            all_diffs["resnet1d_vs_baseline"].append(resnet_accs[f] - baseline_accs[f])
            all_diffs["resnet1d_vs_cnn"].append(resnet_accs[f] - cnn_accs[f])
            per_rep_fold_rows.append({
                "repetition": rep, "fold": f + 1,
                "baseline_acc": baseline_accs[f], "cnn_acc": cnn_accs[f],
                "resnet1d_acc": resnet_accs[f]
            })
        print(f"rep {rep+1}/{N_REPETITIONS}  baseline={np.mean(baseline_accs):.4f} "
              f"cnn={np.mean(cnn_accs):.4f} resnet1d={np.mean(resnet_accs):.4f}")

    # Save per-rep-per-fold raw accuracies
    with open("logs/log11_repeated_grouped_cv_raw.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repetition", "fold", "baseline_acc", "cnn_acc", "resnet1d_acc"])
        writer.writeheader()
        writer.writerows(per_rep_fold_rows)

    # original single-split numbers, for comparison
    original = {}
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original[row["comparison"]] = row

    rng = np.random.default_rng(42)
    summary_rows = []
    print("\n" + "=" * 70)
    print(f"REPEATED GROUPED CV ({N_REPETITIONS} reps x {N_FOLDS} folds = "
          f"{N_REPETITIONS * N_FOLDS} paired diffs per comparison)")
    print(f"Total ResNet1D restarts used across all fold trainings: {total_restarts}")
    print("=" * 70)
    for comp, diffs in all_diffs.items():
        diffs = np.array(diffs)
        mean_diff = diffs.mean()

        # percentile bootstrap CI
        boot_means = np.array([
            rng.choice(diffs, size=len(diffs), replace=True).mean()
            for _ in range(N_BOOTSTRAP)
        ])
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        # sign-flip permutation test (paired, two-sided)
        obs_stat = abs(mean_diff)
        perm_stats = np.empty(N_PERMUTATIONS)
        for i in range(N_PERMUTATIONS):
            signs = rng.choice([-1, 1], size=len(diffs))
            perm_stats[i] = abs((diffs * signs).mean())
        perm_p = (np.sum(perm_stats >= obs_stat) + 1) / (N_PERMUTATIONS + 1)

        orig = original.get(comp, {})
        holds = (mean_diff > 0) == (float(orig.get("mean_diff", 0)) > 0)
        print(f"\n{comp}:")
        print(f"  repeated-CV mean_diff={mean_diff:.4f}  95% bootstrap CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  permutation p-value={perm_p:.4f}")
        print(f"  original single-split mean_diff={orig.get('mean_diff')}  "
              f"original t-test p={orig.get('p_value')}")
        print(f"  original directional conclusion holds: {holds}")

        summary_rows.append({
            "comparison": comp,
            "repeated_cv_mean_diff": mean_diff,
            "repeated_cv_ci_lo": ci_lo,
            "repeated_cv_ci_hi": ci_hi,
            "permutation_p_value": perm_p,
            "n_reps": N_REPETITIONS,
            "n_folds": N_FOLDS,
            "original_single_split_mean_diff": orig.get("mean_diff"),
            "original_single_split_p_value": orig.get("p_value"),
            "directional_conclusion_holds": holds,
        })

    with open("logs/log11_repeated_grouped_cv_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("\nSaved logs/log11_repeated_grouped_cv_raw.csv and logs/log11_repeated_grouped_cv_summary.csv")


if __name__ == "__main__":
    main()

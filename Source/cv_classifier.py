# SOMA Signal Fidelity Investigation
# Stage 3a — K-fold cross-validation
# File: Source/cv_classifier.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from physics_model import stiffness_classes
from source_classifier import build_dataset

if __name__ == "__main__":
    print("Building dataset...")
    X, y = build_dataset()
    print(f"  Total samples: {len(X)}")

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_accuracies = []

    print(f"\nRunning {n_folds}-fold cross-validation...")
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)

        print(f"  Fold {fold_num}: {acc:.2%}  (train n={len(train_idx)}, test n={len(test_idx)})")

    fold_accuracies = np.array(fold_accuracies)
    print(f"\nMean accuracy across {n_folds} folds: {fold_accuracies.mean():.2%}")
    print(f"Standard deviation:                 {fold_accuracies.std():.2%}")
    print(f"Range:                               {fold_accuracies.min():.2%} - {fold_accuracies.max():.2%}")

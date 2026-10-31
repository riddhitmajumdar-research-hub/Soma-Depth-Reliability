# Lane 8 -- local config for the Random Forest depth-floor certification.
# Per audit_fixes rules: nothing here overrides or duplicates
# Source/constants.py. These are NEW knobs the original CNN sweep never
# exposed (it had no model hyperparameters beyond n_epochs, which Lane 8
# does not touch since it doesn't apply to RF). See PREREG.md section 3.

# Random Forest hyperparameters (fixed, no tuning -- per the protocol).
RF_N_ESTIMATORS = 300
RF_N_JOBS = 2
# All other RandomForestClassifier kwargs left at sklearn defaults
# (criterion="gini", max_depth=None, min_samples_split=2,
# min_samples_leaf=1, max_features="sqrt", bootstrap=True).
# See PREREG.md section 3 for why this is flagged, not silently assumed.

# Permutation controls -- matches the CNN sweep exactly.
N_SHUFFLES = 200

# BH-FDR alpha -- matches the CNN sweep's bh_fdr_correction default.
FDR_ALPHA = 0.05

SEEDS_TO_RUN = list(range(101, 125))  # 101-124 inclusive, matches original

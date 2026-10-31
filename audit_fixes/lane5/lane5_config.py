# Lane 5 -- local config for the per-class certification harness.
# Per audit_fixes rules: nothing here overrides or duplicates
# Source/constants.py. These are NEW knobs the original scripts never
# exposed as constants; defaults reproduce the original sweep exactly.
# See audit_fixes/lane5/recall_floor_main_run/PREREG.md section 6 for full provenance/caveats.

# Dataset-size sweep (maps to two independent original knobs; see PREREG 6).
N_TRAIN_REALIZATIONS_PER_CELL = 25   # == degradation_aware_final_sweep.N_SAMPLES_PER_DEPTH_PER_CLASS
N_EVAL_SAMPLES_PER_CELL = 300        # == degradation_aware_final_sweep.N_SAMPLES_PER_CELL

# Window-size sweep. WINDOW maps to cnn_classifier.extract_raw_windows's
# existing `window_size` kwarg (pass-through, not new logic).
WINDOW = 100

# STEP: NOT IMPLEMENTED. extract_raw_windows has no stride/step parameter
# in the original code (one window per detected heartbeat). See PREREG.md
# section 6 -- open question for the researcher, not resolved here.
STEP = None

# Per-class recall floor threshold (Step 1 spec literal value; see PREREG 3).
PER_CLASS_RECALL_FLOOR_THRESHOLD = 0.45

# Permutation controls.
N_SHUFFLES = 200

# Threads (per the protocol).
N_TORCH_THREADS = 4

# BH-FDR alpha (matches original bh_fdr_correction default).
FDR_ALPHA = 0.05

# Representative depths (index into sorted EVAL_DEPTHS_CM at runtime) used
# for the null-histogram figure: shallowest, floor-adjacent, deepest.
# Resolved to actual depths inside the harness once EVAL_DEPTHS_CM is known.
HISTOGRAM_DEPTH_POSITIONS = ("shallowest", "floor_adjacent", "deepest")

SEEDS_TO_RUN = list(range(101, 125))  # 101-124 inclusive, matches original

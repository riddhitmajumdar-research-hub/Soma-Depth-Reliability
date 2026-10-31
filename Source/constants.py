"""
SOMA — single source of truth for physical/sensor constants.

Do not hardcode these values anywhere else. Import from this module.
Changing a value here changes it everywhere — that's the point.
"""

import numpy as np

# Sensor noise floor (Log 18, Fix #1)
# Source: Yang et al. 2022, Biosensors (PMC9312695) — real wearable
# skin-contact MEMS heart-sound sensor, 10-630Hz band, validated against
# a human subject + 3M Littmann stethoscope. Corroborated by two more
# independent wearable MEMS heart-sound papers in-band (26.47dB, 13.58dB).
# Superseded: 67.03dB (1kHz, free-field mic — wrong band, wrong sensor
# category). Do not reintroduce that value anywhere.
SENSOR_SNR_DB = 27.05

# Weakest of the 3 corroborating papers for Fix #1 (Log 18): a real,
# documented sensor-to-sensor SNR figure within SOMA's exact sensor class
# (wearable MEMS heart-sound), not a hypothetical or cross-domain worst
# case. Used only by Source/noise_robustness_stress_test.py to stress-test
# whether the certified floor survives real sensor-to-sensor variability.
# SENSOR_SNR_DB (27.05) remains the primary/best-case value used everywhere
# else in the pipeline -- unchanged.
SENSOR_SNR_DB_WORST_CASE = 13.58

# Default RNG seed for synthetic-data generation and model training.
# Arbitrary — no special meaning, no stated rationale in the README or any
# commit. Value 42 has simply been used consistently since Stage 3 (Log 04)
# for reproducibility, not because it's derived from anything.
#
# NOT the same concept as Source/retrain_seed121_verify.py's local
# `SEED = 121` (which identifies a specific seed from the 24-seed sweep in
# degradation_aware_final_sweep.py) or that sweep's own `SEEDS_TO_RUN`
# range (101-124) — those are per-run seed *values under test*, not this
# default. Do not repoint either of those onto this constant.
SEED = 42

# SOMA's stated nominal operating frequency (README: "SOMA's actual stated
# 100Hz operating frequency" / "at my stated 100Hz operating frequency").
# The device's full operating band is 20-500Hz; 100Hz is the single
# frequency most certified floor/sensitivity results are reported at.
NOMINAL_FREQ_HZ = 100

# --- Stage 4 depth grids ----------------------------------------------------
# Three separate grids, used together across the degradation-aware CNN /
# power-law-sensitivity family of scripts. Do not merge them into one name --
# they mean different things and some scripts use only a subset.

# Training depths for degradation-aware CNNs (Log 15). Not an eval grid.
TRAINING_DEPTHS_CM = [0, 3, 6, 9, 12]

# The original 12-point 0-15cm evaluation grid (1.36cm spacing). This coarse
# spacing is what produced the grid-artifact problem for 250Hz/500Hz
# elsewhere in the codebase (Log 18, Fix #4) -- kept for comparability, not
# because it's sufficient resolution on its own.
BASE_EVAL_DEPTHS_CM = np.linspace(0, 15, 12)

# Densified points in the 3-8cm range, where the degradation-aware floor is
# expected to land (Log 18 Fix #4 / Log 19's borderline-call resolution).
# Callers combine this with BASE_EVAL_DEPTHS_CM themselves, e.g.:
#   EVAL_DEPTHS_CM = np.sort(np.unique(np.concatenate(
#       [BASE_EVAL_DEPTHS_CM, DENSIFIED_DEPTHS_CM])))
DENSIFIED_DEPTHS_CM = [3.5, 4.0, 4.3, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]

# Domain-of-validity safety gate (Log 16): the worked-example demo rejects
# any scan with depth_cm > MAX_VALID_DEPTH_CM, using depth itself rather than
# the model's own softmax confidence, because Log 16 found confidence
# becomes falsely overconfident exactly past this point, and Log 23 confirmed
# at the logit level (top-vs-second-class gap) that no calibration method can
# make confidence trustworthy in that region. Do not confuse with 8.18cm/8.48cm
# (the accuracy floor itself) -- this is the separate safety cutoff.
#
# UPDATED (gate-candidate analysis, then corrected): previously 9.0 (chosen
# just past the certified 8.18cm single-seed accuracy floor from Log 15).
# A first pass set this to 7.89cm -- the 5th percentile of BOOTSTRAP RESAMPLE
# MEANS of the 24-seed floor distribution. An external methodology review
# caught that this is the wrong statistic for a safety-critical threshold:
# it's a 95% lower confidence bound on the *population mean* floor, not a
# tolerance bound on individual-unit risk, and the observed worst seed
# (5.45cm) sits over 3cm below it -- meaning a real unit could fail well
# inside what 7.89cm claimed was a safe margin.
#
# Corrected to 6.50cm: the empirical 10th percentile of the 24 individual
# per-seed floor values themselves (Source/logs/degradation_aware_final_results.csv),
# chosen over the 5th percentile (5.65cm) for reduced sensitivity to
# small-sample noise at n=24, while remaining a genuine per-unit tolerance
# bound rather than a bound on the average. Full derivation and all
# candidate values: Source/logs/gate_statistic_verification.csv.
MAX_VALID_DEPTH_CM = 6.50

# Seed-121's own certified accuracy floor from Log 20's 24-seed sweep
# (corrected noise), NOT the 24-seed mean (8.48cm +/- 1.81cm). This is
# specifically the floor for the one seed whose weights are the reloadable
# certified model (outputs/stage4/models/soma_certified_model_seed121_corrected_noise.pt),
# reproduced and verified independently in Source/retrain_seed121_verify.py.
# Coincidentally numerically identical to Log 15's original, PRE-correction
# single-seed (42) floor -- that is a different quantity from a different
# run; do not conflate them. If the certified model is ever retrained on a
# different seed, this must be updated to that seed's own floor.
SEED121_CERTIFIED_FLOOR_CM = 8.18

# Basic (clean-trained, non-degradation-aware) CNN's certified floor at
# 100Hz under corrected noise (Log 19), as computed by
# Source/floor_characterization_cnn.py. Used elsewhere only as a
# cross-reference comparison point, e.g. in degradation_aware_cnn.py's
# result printout -- update here if floor_characterization_cnn.py's
# certified number is ever re-derived.
BASIC_CNN_CORRECTED_FLOOR_CM = 4.30

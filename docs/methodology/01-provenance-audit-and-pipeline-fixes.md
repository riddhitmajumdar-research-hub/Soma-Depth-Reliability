# 01. Provenance audit and pipeline fixes

## Question

Where did each physical constant in the Stage 4 pipeline come from, and did the code still use the corrected values everywhere after the constants were changed?

## What was done

**Provenance re-certification.** Four sourcing gaps were fixed and every Stage 4 floor was re-run:

1. Sensor SNR changed from 67.03 dB (a 1 kHz free-field microphone) to 27.05 dB, taken from a wearable skin-contact MEMS heart-sound paper (now cited as Yang et al. 2022, earlier recorded as 'Chen et al.'; see 06 for the later attribution problem).
2. Stiffness-class citation replaced (Yoo et al. (earlier recorded as 'Lee et al.') cutoffs did not match the values in use; Zayadeen et al. (earlier recorded as 'Al-Khalili et al.') per-stage values did).
3. Viscosity made class-dependent (healthy/fibrosis 0.5 Pa·s, cirrhosis 1.77 Pa·s, after Reiter et al. 2021) instead of one constant for all classes.
4. Depth grid densified near the origin (250/500 Hz) and between 4 and 7 cm (100 Hz). The old shared 12-point grid had reported a 1.36 cm floor at both 250 Hz and 500 Hz. With the finer grid they resolve to 0.75 cm and 0.30 cm, so the earlier "same floor" was a grid artifact.

Re-certified numbers at 100 Hz: baseline 0.00 cm, clean CNN 4.30 cm, degradation-aware CNN 8.48 cm (24 seeds; the spread was later corrected, see 03).

**Two cleanup changes.** A sweep for leftovers from the first changes found:
- `attenuation_model_powerlaw.py` imported a viscosity constant that fix 3 had renamed, so the import was broken.
- `power_law_sensitivity.py` and `power_law_sensitivity_certified.py` still used 67.03 dB and the coarse grid.
- `run_worked_example.py` ran an untrained random model and printed plausible-looking output. Deleted.
- `run_calibration_plot.py` duplicated `calibration_analysis.py` with a different bin range. Deleted; `run_all_remaining.py` now calls the canonical script.
- `stage4_degradation_aware_results.pkl` was suspected stale. Checked and found current (8.18 cm, densified grid present).
- Calibration data regenerated under the corrected noise and grid. A second `.gitignore` blanket rule (`*.pt`, `*.pkl`) was also hiding tracked outputs.

**Recalibration change (Log 23).** Calibration recalibration: four methods tried, root cause traced to raw logit decay with depth, partial fix scoped to the domain-of-validity gate.

**Centralisation changes.** `SEED`, `NOMINAL_FREQ_HZ`, `SENSOR_SNR_DB`, the Stage 4 depth grids and `MAX_VALID_DEPTH_CM` moved into `Source/constants.py`. Stale model references repointed to the verified seed-121 weights. The 24-seed CNN sweep now saves per-seed weights (it had not before, which is why later audit work could only use seed 121 for the CNN). Five hardcoded floor literals replaced with named constants. Noise-robustness stress test added (Log 24).

## Limitations

The 26.47 dB and 13.58 dB corroborating sensors are still unnamed in the repo (see 06).

## Reproducibility

Scripts as of the provenance re-certification and later under `Source/`. `Source/PROVENANCE_AUDIT.md` was added in that change and deleted in a later cleanup, after its content went into README Logs 18 to 20. The file does not exist in the working tree and has not been restored. Its text is not included here, and several of its statements have since been overtaken: the floor SD is given as 1.77 cm (now 1.81, see 03), the 27.05 dB source is given as Chen et al. (the write-up now says Yang, see 06), and the note that no saved weights exist predates the per-seed weights saved from. Read it as the state of the audit, not as a current reference.

## Leads to

02 (external review, the gate value and grouped splits), 06 (citation check).

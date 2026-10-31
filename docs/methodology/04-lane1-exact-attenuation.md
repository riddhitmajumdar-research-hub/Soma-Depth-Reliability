# 04. Lane 1: exact vs weak-damping attenuation

Files taken from the backup copy in `~/soma_backup/audit_fixes_backup`.

Follows from the leading-order bound in the external review script batch (02). Feeds the weak-damping limitation in the write-up and the battery conditions 6a/6b (12).

## Question

How far does the weak-damping formula `alpha = omega^2 eta / (2 G c)` used in `Source/attenuation_model.py` sit from the exact complex-wavenumber solution?

## Method

Exact: `k = omega * sqrt(rho / (G + i omega eta))`, `alpha = |Im k|`. `numpy.sqrt` returns the growing-wave branch (Im k < 0) on this input, so the script takes the absolute value. Constants are imported from `Source/`, not retyped. Five tasks, each with its own script, stdout log and CSV in `audit_fixes/lane1/`.

## Numbers

- alpha_weak / alpha_exact at 100 Hz: healthy 1.027, fibrosis 1.010, cirrhosis 1.048. At 250 Hz up to 1.28 (cirrhosis), at 500 Hz up to 1.99. Weak damping always overestimates attenuation. Seven check values from the protocol matched.
- Depth shift at 100 Hz, amplitude-matched: +0.23 cm (healthy), +0.08 (fibrosis), +0.41 (cirrhosis) relative to 8.48 cm; +0.31 cm worst case against the 6.50 cm gate. This bounds the physics-model error only. It is not a re-certified floor.
- Arrival-time separability at fs = 500 Hz (2 ms/sample): 14 of 30 distance x class-pair x velocity-model cells have a gap under one sample. At 5 mm no pair clears one sample; from 15 mm all do. Using `c_exact` does not change which cells pass.
- 20 Hz heuristic depth equivalent: 155 to 210 cm. Self-consistent, physically meaningless.
- Monte Carlo (5000 draws, viscosity and density ranges imported from `joint_uncertainty.py`, healthy class): amplitude-equivalent depth mean 4.627 cm, SD 1.417, 5th/95th percentile 3.138/7.613 cm. The upper viscosity bound is cirrhosis's, so the shift is mechanical.

## Limitations

Task 2 and Task 5 are heuristics. Turning them into a certified number needs an eval re-run with the exact attenuation law, which battery conditions 6a/6b later did (12): floors moved deeper by a median of 1.46 cm (seeds 101 to 104) and 0.34 cm (105 to 108).

## Reproducibility

`python audit_fixes/lane1/task1_exact_vs_weak.py`, then `task2` to `task5` in order. Report: `audit_fixes/lane1/REPORT.md`.

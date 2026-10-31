# 12. Cloud robustness battery (conditions 1 to 8)

Both batteries start from the same repository snapshot (`run base`).

## Question

How does the floor move when the simulator or noise assumptions change, and is the direction the same across seeds?

## Method

19 conditions per seed on the certified pipeline (`audit_fixes/cloud/battery/`): amplitude randomization (2), geometric spreading p=0.5 and 1.0 (3), within-class stiffness CV (4), SNR 20/13.58/10/5/0 dB (5), exact attenuation with and without exact TOF (6), viscosity variants (7), and a pure-noise probe (8). Real model and 200 controls retrained per seed, controls trained once on the original distribution and reused. PREREG.md was committed before each battery ran. A regression test reproduced the baseline data generator bit-for-bit (27/27 in battery-B).

## Findings

Direction of median floor shift against each seed's baseline, seeds 101 to 104 / 105 to 108:
- SNR: shallower as SNR falls, 0 dB about -4.3 / -5.0 cm.
- Spreading: p=0.5 -2.1 / -2.5; p=1.0 -4.0 / -4.6. Seed 108 at p=0.5 goes the other way (+4.09 cm, floor at the 15 cm maximum) and is the only sign flip in 8 seeds. `audit_fixes/cloud/spreadcheck/FINDINGS.md` found that the premise (control collapse near the floor in seeds 106 and 107) did not match the data, and that seed 108 has the deepest baseline (10.91 cm) of the eight, which leaves more room for shifts in both directions. Its baseline also shows anomalous recovery. Cause not settled.
- Exact attenuation (6a): +1.46 / +0.34 cm, 6b about 0.
- Viscosity: healthy/fibrosis x1.4 to 2.6 deeper (median +1.4 to +2.4); cirrhosis 1.2 Pa·s much deeper (+5.45 / +4.77, several seeds censored); cirrhosis 2.18 and 2.5 shallower (about -1 to -3). At x2.6 seeds 103 and 104 shallowed by 4.7 and 5.3 cm while the other six did not. `audit_fixes/cloud/vischeck/FINDINGS.md` pulled the per-depth data: accuracy declines smoothly in all five seeds it examined, and the two shallow floors come from a few mid-depth cells (accuracy 0.45 to 0.55, p just above the BH threshold) failing significance. Same artifact family as item 2 below. No reruns were done.
- Stiffness CV (4a/4b): bimodal, some seeds at the 15 cm maximum, some at 0.00; no usable floor shift.
- Noise probe (8): across all 8 seeds the real model never predicts fibrosis on pulse-free noise; healthy takes 48 to 100%. The write-up, Section F cites this.

## Bugs and corrections

1. **Amplitude gain (condition 2).** The draw was log-uniform, `exp(U(ln 0.5, ln 2))`, with mean about 1.08, not 1.0. Battery-A added condition 2b, `g ~ U(0.5, 1.5)` linear, mean 1.0, rerun. Seeds 101, 103, 104 kept the same floor (+1.36, +1.36, +1.55); seed 102 went from +0.50 to 0.00. The "all four seeds agree in sign" statement in the first report no longer holds. Battery-B used the corrected draw only, so 105 to 108 pool with 2b rows, not condition 2. With both, 7 of 8 seeds are non-negative, one is zero. A read-only check of condition 2 (not in this tree) confirmed 2b was absent from the original run and appended later; the 2b rows are in the detail CSVs and floor CSV but not in the main REPORT table.
2. **Floor of 0.00 cm with real accuracy 67 to 79%.** Same resolution limit as Q6 (05): minimum p 0.00498 against a rank-1 BH threshold of 0.00217. Seen in 3 cells of battery-A and 2 of battery-B. Not fixed; raising the shuffle count would break comparability with the certified sweep.
3. **Suspected untrained real model.** Real model trained in under a second while 200 controls took about 60 s. Checked with a logged loss curve (1.102 to 0.855, train accuracy 67.5%, 0.31 s): full-batch, 60 epochs, 375 windows. Not a bug.
4. **Cross-platform.** Linux baseline floors against the Mac replicate: 101 exact, 102 -0.5, 103 -1.37, 104 +0.5 cm. Not diagnosed.

## Limitations

n=4 per battery and no significance tests, by design. Floors pinned at 15.0 cm are censored. A read-only check of the protocol (not in this tree) also found that it named the second battery (seeds 105 to 108), which was not yet available; it appeared afterward.

## Reproducibility

Everything is in `audit_fixes/cloud/battery/seeds_101-104/` and `audit_fixes/cloud/battery/seeds_105-108/` (protocols, reports, scripts and logs) and in the check folders `audit_fixes/cloud/extract/`, `audit_fixes/cloud/vischeck/` and `audit_fixes/cloud/spreadcheck/` (`FINDINGS.md` each); a read-only check of the protocol and a read-only check of condition 2 are not in the tree. Raw per-depth CSVs (`floors_*.csv`, `depth_detail_*.csv`, `noise_probe_*.csv`) are in `audit_fixes/cloud/battery/results/`, pulled from both runs and checked against them. Confusion matrices (`confmat_*.npz`) are not in this tree.

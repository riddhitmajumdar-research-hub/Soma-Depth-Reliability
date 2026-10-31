# 10. Lane 7: ResNet1D vs CNN floor distributions

The ResNet1D sweep itself started earlier and the status log shows it running and then stopped, so seed 124 finished between those two checks (log line: 3972 s for that seed). The CSV and log were saved at that point.

Follows from the 24-seed CNN certification and the paired-CV work (02, 05). Write-up Appendix B has the ResNet1D row.

## Question

Is the ResNet1D floor distribution different from the CNN's on the same 24 seeds?

## Method

PREREG.md fixed the tests before any result was read. Inputs: the two 24-row CSVs in `Source/logs/`. Mann-Whitney with a 100,000-permutation cross-check, Hodges-Lehmann shift with bootstrap CI, paired Wilcoxon on matched seeds, sign-flip test, fraction below 6.50 cm with Clopper-Pearson. v2 added right-censoring, a seed-pairing check and a per-seed look at seed 110.

## Numbers (n=24, `final_v1_output.txt`, `final_v2_output.txt`)

| | CNN | ResNet1D |
|---|---|---|
| mean / SD | 8.475 / 1.812 | 10.360 / 3.163 |
| median | 8.00 | 10.91 |
| 5th pct (linear) | 5.65 | 4.30 |
| 10th pct (linear) | 6.50 | 6.245 |
| below 6.50 cm | 2/24 | 3/24 (seeds 101, 110, 112) |
| anomalous recovery | 10/24 | 8/24 |

Mann-Whitney p = 0.00235 (permutation 0.00177). HL shift 2.73 cm, CI [1.36, 4.09]. Paired Wilcoxon p = 0.0284 (one zero difference dropped). Sign-flip on mean difference 1.88 cm, p = 0.0300. Gehan-Wilcoxon with seed 103 censored: p = 0.00227; log-rank p = 0.00052. ResNet1D restarts: 1 of 24 (seed 117).

Censoring: both sweeps set the floor to 15.0 cm when every depth stays significant. ResNet1D seed 103 hits that; no CNN seed does. The n=20 report treated it as an ordinary observation; v2 flags this as a disagreement with v1 and reports both.

Seed pairing: for seed 101 the two scripts produce identical training sets, identical 200 control permutations and identical eval windows at depth 0. The pairing is real, not just matching seed numbers.

Seed 110 (ResNet1D floor 1.36 cm, last significant 13.64 cm) is a real observation, not censored. Re-evaluated weights from a fresh eval draw gave accuracy 0.58 at 0 cm and 0.36 to 0.56 elsewhere.

## Limitation

The ResNet1D has a higher median floor but a wider spread and a shallower 5th percentile than the CNN, so the two are not simply ordered. `lane7/REPORT.md` and `REPORT_v2.md` in the repo are still the provisional n=20 texts (their "blocked" status is out of date); the n=24 numbers exist only in the two `final_*_output.txt` files and the write-up. A `lane7_provisional_backup/` directory held a copy of the n=20 set; all 12 of its files were byte-identical to those in `lane7/` and it was removed. The Gehan and log-rank tests are hand-written because `lifelines` is not installed; a cross-check is open. No CNN weights exist for seed 110.

## Reproducibility

`python audit_fixes/lane7/run_lane7_analysis.py`, `run_lane7_v2_analysis.py`, `verify_seed_pairing.py`. Needs numpy and scipy only.

# Methodology record

One file per piece of work, in work order. Timing is not recorded in these notes.

| # | File | Summary |
|---|---|---|
| 01 | [provenance audit](01-provenance-audit-and-pipeline-fixes.md) | Sensor SNR, stiffness and viscosity sources corrected; depth grid densified; stale scripts and constants cleaned up |
| 02 | [external review](02-external-review-fixes.md) | Gate 9.0 to 6.50 cm, grouped splits, weak-damping bound, seed-121 per-class recall |
| 03 | [figures and SD](03-figure-and-sd-corrections.md) | Two figure bugs; floor SD 1.77 to 1.81 cm |
| 04 | [Lane 1](04-lane1-exact-attenuation.md) | Exact vs weak-damping attenuation |
| 05 | [Lane 2, Q6](05-lane2-and-q6-statistical-rechecks.md) | Nadeau-Bengio, Cohen's d, gate percentile, BH family size |
| 06 | [Lane 3](06-lane3-fact-checks.md) | Fact-checks; sensor citation gaps |
| 07 | [Lane 4](07-lane4-baseline-architectures.md) | SVM/RF/GRU baselines, v1 failure and v2 fix |
| 08 | [Lane 6](08-lane6-beat-offset-mechanism.md) | Window-offset cue: 90.6% accuracy from one scalar |
| 10 | [Lane 7](10-lane7-resnet1d-vs-cnn.md) | ResNet1D vs CNN floors, 24 seeds, censoring |
| 11 | [Lane 8, RF](11-lane8-and-cloud-random-forest.md) | Random forest floor, 24 seeds in `audit_fixes/cloud/rf/` |
| 12 | [battery](12-cloud-robustness-battery.md) | 19-condition robustness runs, seeds 101 to 108 |
| 13 | [corrections](13-collapse-narrative-and-probe-corrections.md) | Fig. 10 pooled confusion, Fig. 11 rerun at 27.05 dB |
| 09 | [Lane 5](09-lane5-per-class-certification.md) | Per-class floors, 24 seeds; diagnostics from outside the repo |

File 09 is listed last because its follow-up work (clean-trained sweep, balanced accuracy, bug fix) ends after the runs in 10 to 12; it started before them.

## Lane and note numbers

The notes are numbered in work order and the lanes were numbered when they were assigned, so the two do not line up. The files were not renumbered. Each lane directory has its own README with the question, times and contents.

| Lane directory | Note | Notes on the mismatch |
|---|---|---|
| none (main line of work) | 01, 02, 03 | Provenance audit, external review fixes, figures and SD |
| [lane1](../../audit_fixes/lane1/README.md) | 04 | |
| [lane2](../../audit_fixes/lane2/README.md) | 05 | Shares note 05 with Q6 |
| [q6_bh_fdr_families](../../audit_fixes/q6_bh_fdr_families/README.md) | 05 | Named by question, not lane |
| [lane3](../../audit_fixes/lane3/README.md) | 06 | |
| [lane4](../../audit_fixes/lane4/README.md) | 07 | |
| [lane6](../../audit_fixes/lane6/README.md) | 08 | Note 08 is lane 6 |
| [lane5](../../audit_fixes/lane5/README.md) | 09 | Note 09 is lane 5, listed last in the table above |
| [lane7](../../audit_fixes/lane7/README.md) | 10 | |
| [lane8](../../audit_fixes/lane8/README.md) | 11 | Results are in `audit_fixes/cloud/rf/`, see below |
| [cloud](../../audit_fixes/cloud/) | 12 | The robustness battery results are in `audit_fixes/cloud/battery/results/` |

## Where to look

One row per question people ask. Each row names the note that gives the answer and lists every other place that holds part of it, so the note is the place to start.

| Question | Start here | Also holds part of it |
|---|---|---|
| What is the random forest floor? | [11](11-lane8-and-cloud-random-forest.md): median 12.27 cm over 24 seeds, none under 6.50 cm | [lane8](../../audit_fixes/lane8/README.md): harness, PREREG and the seed-101 benchmark only (13.64 cm). `audit_fixes/cloud/rf/`: per-seed tables (`results/rf_floors_101-112.csv`, `results/rf_floors_113-124.csv`), reports, the wrapper script and the logs. Write-up Appendix B: the median and the 0/24 count. Missing everywhere: a script or report that combines the 24 seeds. |
| Can a per-class floor be certified? | [09](09-lane5-per-class-certification.md) | [lane5](../../audit_fixes/lane5/README.md) for scripts and outputs; `lane5/REPORT.md` is superseded |
| ResNet1D against the CNN | [10](10-lane7-resnet1d-vs-cnn.md) | [lane7](../../audit_fixes/lane7/README.md): n=24 numbers are in `final_v1_output.txt` and `final_v2_output.txt`; the two REPORT files are superseded |
| Robustness to simulator and noise assumptions | [12](12-cloud-robustness-battery.md) | `audit_fixes/cloud/battery/results/` |
| Collapse direction (Fig. 10), frozen probe noise level (Fig. 11) | [13](13-collapse-narrative-and-probe-corrections.md) | `Source/make_fig10_pooled_confusion.py`, `Source/make_fig11_corrected_probe.py` |
| Exact against weak-damping attenuation | [04](04-lane1-exact-attenuation.md) | [lane1](../../audit_fixes/lane1/README.md) |
| Why a 200-shuffle BH result reads 0.00 cm | [05](05-lane2-and-q6-statistical-rechecks.md) | [q6_bh_fdr_families](../../audit_fixes/q6_bh_fdr_families/README.md), then notes 09 and 12 for where it recurs |
| Where the 6.50 cm gate and the 1.81 SD come from | [02](02-external-review-fixes.md), [03](03-figure-and-sd-corrections.md) | [lane2](../../audit_fixes/lane2/README.md) for the recheck |
| Sensor citations | [01](01-provenance-audit-and-pipeline-fixes.md), [06](06-lane3-fact-checks.md) | [lane3](../../audit_fixes/lane3/README.md) |
| Source script index | [Source/README.md](../../Source/README.md) | |

## Threads

**Gate and floor statistics.** The re-certification of 8.48 cm (01) → the gate change replaces the 9.0 cm gate with the 10th percentile, 6.50 cm (02) → the SD correction to 1.81 (03) → Lane 2 recomputes both and finds the interpolation dependence (05) → Lane 7 repeats them for ResNet1D (10) → write-up Appendix B, "Safety gate" and "24-seed" rows.

**Train/test leakage and significance.** The grouped splits (02) → Lane 2 Nadeau-Bengio, p=0.25 for ResNet1D vs CNN, and the unreproducible d values (05) → Lane 4 uses the same folds for other model families (07) → Appendix B rows for accuracy, Cohen's d and significance.

**Weak damping.** The leading-order bound (02) → Lane 1 exact solution (04) → battery conditions 6a/6b move floors by a median of +1.46 and +0.34 cm (12) → write-up "250/500Hz downgraded".

**BH resolution limit.** Q6: 200 shuffles cannot pass a 23-test BH family (05) → same zero floors in Lane 5's per-class run (09) → same in three battery cells (12).

**Per-class floor.** Seed-121 recall in the external review script batch (02) → Lane 5 harness, degenerate recall null, balanced-accuracy retry, clean-trained sweep and the clean-harness loop bug (09) → Section F and the Appendix B per-class row, which say certification is not attainable under this protocol. The pure-noise probe in the battery (12) supports the same collapse toward "healthy" described there.

**Model-family dependence.** ResNet1D 24-seed sweep (10) and random forest 24 seeds (11) → Appendix B rows. Lane 6 (08) is the mechanism side: the window offset carries most of the class information on clean data, and it fades once degradation is applied.

**Citations.** Provenance audit (01) → Lane 3 grep (06) → write-up changes Chen to Yang. The repo files have not been changed to match.

## Open gaps

- The repository and the write-up now cite the same first authors; the 26.47 and 13.58 dB sources are unnamed (06).
- `audit_fixes/lane5/REPORT.md` and `lane7/REPORT*.md` are stale relative to their final results (09, 10). No report exists for the Lane 5 balanced-accuracy, multiplicity and clean-sweep results beyond file 09.
- No combined 24-seed random forest report or comparison script (11); the RF results CSVs and the battery CSVs are in `audit_fixes/cloud/rf/results/` and `audit_fixes/cloud/battery/results/`, and the battery confusion matrices are not in the tree (11, 12).
- The RF median and 0/24 count in Appendix B were recounted here from the per-seed tables in the two reports in `audit_fixes/cloud/rf/`. No script that combines the 24 RF seeds was found.
- Exact time of the clean-harness bug fix is unknown (09).
- Original Log 11 restart counts were never saved (02).
- Lane 4 GRU failure unexplained (07). The v2 SVM/RF results were corrected with Nadeau-Bengio (07).
- Lane 6 Q9 covers 3 seeds only (08). The seed-108 spreading flip and the x2.6 viscosity shallowing are not resolved beyond what file 12 states (12).
- `soma_nullcheck_output.txt` ends in a KeyError and was not rerun (09).
- Not saved: `/tmp` write-up edit scripts; an earlier copy adds nothing.
- The saved CSV and log of the ResNet1D sweep were left as they are.

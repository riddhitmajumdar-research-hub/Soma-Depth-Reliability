# Lane 4: baseline architectures

**Question.** Do SVM, random forest and GRU models match the CNN and ResNet1D on the clean 3-class task, using the same grouped 5-fold indices as `Source/repeated_grouped_cv_log11.py`?

**Methodology note.** [07](../../docs/methodology/07-lane4-baseline-architectures.md).

**Contents.**
- `PREREG.md` and `REPORT.md` for v1, `PREREG_v2.md` and `REPORT_v2.md` for the v2 amendment. Each PREREG was written before its run.
- `run_lane4_comparison.py`, `run_lane4_v2_comparison.py`, `diagnose_v1.py`, with their logs and CSVs.

The v1 run failed, `diagnose_v1.py` found why, and v2 is the fix. The GRU failure is unexplained. The v2 CSVs were put through the Nadeau-Bengio correction (`nb_v2.py`, `nb_v2_output.txt`).

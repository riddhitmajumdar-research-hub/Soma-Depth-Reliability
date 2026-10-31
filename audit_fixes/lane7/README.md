# Lane 7: ResNet1D against the CNN

**Question.** Is the ResNet1D floor distribution different from the CNN's on the same 24 seeds?

**Methodology note.** [10](../../docs/methodology/10-lane7-resnet1d-vs-cnn.md).

**Which files are current.**
- `REPORT.md` and `REPORT_v2.md` are the n=20 texts and are out of date. Each has a notice at the top.
- `final_v1_output.txt` and `final_v2_output.txt` hold the n=24 numbers. There is no committed n=24 report.
- `lane7_output_raw.txt`, `lane7_v2_output_raw.txt`, `seed110_eval_raw.txt`, `seed_pairing_raw.txt` are the n=20 raw outputs.

**Other contents.** `PREREG.md` and `PREREG_v2.md`; `run_lane7_analysis.py`, `run_lane7_v2_analysis.py`, `eval_seed110.py`, `verify_seed_pairing.py`.

`lane7_check.py` compares the Lane 5 rerun floors with the ResNet1D floors. It reads [../lane5](../lane5/README.md) outputs and was recovered from outside the repo.

A directory `lane7_provisional_backup/` held a copy of the n=20 files. All 12 of its files were byte-identical to those in this directory and it was removed.

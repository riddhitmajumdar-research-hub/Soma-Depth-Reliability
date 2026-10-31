# Lane 8: random forest floor

**Question.** Is the depth floor a property of the physics and noise, or of the CNN? The same pipeline was certified with a random forest.

The 24-seed runs are in `audit_fixes/cloud/rf/` (run overnight).

**Methodology note.** [11](../../docs/methodology/11-lane8-and-cloud-random-forest.md).

**Contents.** `PREREG.md` (written before any run), `lane8_config.py`, `lane8_harness.py`, `lane8_benchmark.py`, and `logs/` with the seed-101 benchmark. `controls/` is an empty local directory.

**Where the results are.** Only seed 101 is in this directory. The 24-seed results are in `audit_fixes/cloud/rf/`, and there is no combined report. The [topic finder in the methodology index](../../docs/methodology/README.md#where-to-look) lists all four places and what each holds.

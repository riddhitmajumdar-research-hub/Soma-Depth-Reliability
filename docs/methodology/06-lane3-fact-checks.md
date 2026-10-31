# 06. Lane 3: code and log fact-checks

Read-only lane; `audit_fixes/lane3/FACTS.md` holds the answers.

## What was checked

Nine factual questions about dataset sizes, sweep provenance, the ETS Montreal null test, tracking-distance usage and sensor modality, each answered from code and logs with the command shown. Checked numbers include: 23 unique eval depths (12 base + 11 densified); ETS Montreal shuffled-label null mean 0.499, SD 0.0327 over 200 shuffles; 10/24 anomalous-recovery seeds; three seeds (101, 112, 122) whose floor and last-significant depth differ by 1.36 cm; the 4.30 cm clean-CNN floor traced to a single BH family of 82 (frequency, depth) cells (20 Hz: 12 depths, 100 Hz: 24, 250 Hz: 23, 500 Hz: 23; `verify_82_cells.py`).

## Findings that required action

- **Citation.** Every "Chen et al. 2022" attribution for 27.05 dB was consistent inside the repo. The write-up ledger later records the correct first author as Yang et al. 2022 (same paper). The repo text still says Chen; `grep Yang` in `Source/constants.py` and `README.md` returns nothing. Open.
  - Update: the repository now cites Yang et al. 2022 (earlier recorded as 'Chen et al.'), aligned with the write-up. The finding above is kept as originally written.
- **Corroborating sensors.** The 26.47 dB and 13.58 dB papers are never named in the repo. Only SNR and a one-line description exist. The ledger says two of the three papers share a research group.
- **Not found:** an artifact labeled "Fig. 20"; explicit device selection in the sweeps; CNN per-seed wall-clock timing.

## Reproducibility

`python audit_fixes/lane3/verify_numbers.py`, `verify_82_cells.py`. The `citation_check_scratch.txt` file lists the grep commands only; its output is not on disk.

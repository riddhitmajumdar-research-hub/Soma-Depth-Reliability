# How to reproduce the headline

This bundle regenerates the 200-epoch headline (Stage 2 plus Stage 2C, 24 seeds, 3 draws each) and the Stage 2D jitter result (8 seeds) from saved permutation-count files. Nothing is trained and no random permutations are drawn; the scripts only read the saved `.npz` files.

## Requirements
Python 3 with `numpy` and `scipy` (tested with Python 3.10, numpy 2.2.6, scipy 1.15.3).

## Commands
Run everything from the folder that contains this README.

```bash
python3 verify.py
```

`verify.py` does three things: it checks every file against `SHA256SUMS`, runs the two analysis scripts, and compares the printed numbers with `expected_results.txt`. It prints PASS or FAIL for each line.

To run the analysis scripts by hand instead:

```bash
mkdir -p outputs/stage2c outputs/stage2d
python3 outputs/phase3/code/stage2c_analyze.py
python3 outputs/phase3/code/stage2d_analyze.py
```

They write `outputs/stage2c/STAGE2C_RESULTS.md` and `outputs/stage2d/STAGE2D_RESULTS.md`.

## Expected results
- Stage 2 plus 2C, N = 24: mean 13.864, SD 1.113, median 13.636, min 12.273, max 16.364, P10 12.273, bootstrap CI for P10 [12.273, 13.636]. Against the certified 60-epoch per-seed medians: 20 up / 4 equal / 0 down, sign test with zero differences dropped, n = 20, p = 1.907e-06.
- Stage 2D, J = 2.0 ms vs J = 1.0 ms, N = 8: mean 12.273 vs 14.148; 0 higher / 2 equal / 6 lower; sign test with zero differences dropped, n = 6, p = 0.03125.

## What is in the bundle
- `outputs/phase3/raw_stage2/e200/` seed 101-124, draw 0 (Stage 2, J = 1.0 ms), 24 files
- `outputs/phase3/raw_stage2c/e200/` seed 101-124, draws 1 and 2 (Stage 2C), 48 files
- `outputs/phase3/raw_stage2d/j2p0_e200/` seed 101-108, J = 2.0 ms, 8 files
- `outputs/phase3/analysis_cnn/cnn_floor_by_seed.csv` certified 60-epoch per-seed floors, used as the comparison
- `outputs/phase3/code/` the four scripts the analysis needs, copied unchanged
- `SHA256SUMS` hashes of every file listed above; `SCRUB_MAP.tsv` see below

## One change to the data files
Each `.npz` originally stored the absolute path of its network checkpoint in a text field named `checkpoint`. In this bundle that field was shortened to a path relative to the repository root. Every numeric array is bit-identical to the original (checked when the bundle was built). Because the file bytes changed, the file hashes differ from the originals; `SCRUB_MAP.tsv` lists the original and the bundle hash for each file. The analysis scripts do not read the `checkpoint` field.

## Not included
The 60-epoch raw files (`raw/cnn`, about 13.5 MB) that the certified per-seed floors were computed from, the trained checkpoints, and the training code. Without them the certified 60-epoch floors are taken from the shipped CSV rather than recomputed.

# RUNLOG — battery-A (seeds 101-104)

Environment recorded before any run.

## Hardware
- nproc: 4
- RAM: 15Gi total (free -h), 0B swap

## Software
- Python: 3.11.15 (/usr/local/bin/python3)
- pip: 24.0
- numpy: 2.2.6
- scipy: 1.17.1
- scikit-learn: 1.9.1
- matplotlib: 3.11.2
- torch: 2.14.0+cu130 (installed the CUDA build as specified; the
  CPU-only index was blocked from this network, so this build runs on CPU:
  `torch.cuda.is_available()` -> False, confirmed).
- Installed via `pip install --index-url https://pypi.org/simple numpy==2.2.6
  scipy scikit-learn matplotlib torch==2.14.0` (plain PyPI, no CUDA/CPU
  wheel index override).

## Path symlink
`sudo ln -sfn <REPO_ROOT>
<REPO_ROOT>` — succeeded (sudo
available, no password prompt in this container). Verified target exists.

## Repo state
- Branch: run battery-101-104, created from <base>
  (commit <commit> "Snapshot for cloud runs (clean)").
- Working directory for this lane: audit_fixes/cloud/battery/

# RUNLOG -- battery-B (seeds 105-108)

## Setup
- `git fetch origin && git checkout <base>` (via `<base>`,
  since a sibling worktree already holds the local `<base>` branch
  pointer), then `git checkout -b run battery-105-108 <base>`.
  HEAD at branch creation: `<commit> Snapshot for cloud runs (clean)`.
- Optional sudo symlink for the Mac path (`sudo ln -s
  <REPO_ROOT> /Users`): succeeded, exit 0. Not required
  for this lane (confirmed unnecessary in the smoke test per the task); kept
  since it succeeded and is harmless.
- `pip install --index-url https://pypi.org/simple numpy==2.2.6 scipy
  scikit-learn matplotlib torch==2.14.0`: all requirements already
  satisfied in this container's Python 3.11 environment. Installed
  versions verified by import:
  - numpy 2.2.6
  - scipy 1.17.1
  - scikit-learn 1.9.1
  - matplotlib 3.11.2
  - torch 2.14.0+cu130 (CUDA build; runs CPU-only here, as in 101-104's
    REPORT.md which notes "torch 2.14.0+cu130" too)

## Code
Copied from `run battery-101-104` (read via `git show`, branch
never merged/checked out) into this branch's
`audit_fixes/cloud/battery/`: `wrappers.py`, `harness.py`,
`test_regression.py` unchanged; `config.py` edited (seeds 105-108,
condition 2 -> corrected linear gain only, comment block updated).
`PREREG.md` written and committed before any run.

## Regression test
`python3 test_regression.py` run before seed 105 is launched: PASSED --
"REGRESSION TEST PASSED: baseline wrapper reproduces original bit-for-bit"
for all 3 seeds (101, 42, 7) x 3 classes x 3 depths (27/27 OK).

## Per-seed log
Appended after each seed completes (elapsed time, condition-8 confirmation,
git commit SHA of that seed's checkpoint push).

- seed 105: launched via `nohup python3 harness.py > logs/harness_stdout.log 2>&1 &`
  (all 4 seeds run in one process, sequential). TOTAL 208s (train=17s,
  controls=98s, 18 depth-sweep conditions + noise probe ~93s). This is the
  run's first-seed baseline for the 3x stop threshold (harness.py's own
  built-in logic: threshold = 624s). Checkpointed + pushed at commit
  <commit>.
- seed 106: TOTAL 176s (train=1s, controls=80s). Within threshold.
  Checkpointed + pushed at commit <commit>.
- seed 107: TOTAL 172s (train=1s, controls=77s). Within threshold.
  Checkpointed + pushed at commit <commit>.
- seed 108: TOTAL 190s (train=1s, controls=91s). Within threshold.
  Checkpointed + pushed at commit <commit>. Harness printed
  "ALL SEEDS COMPLETE"; no STATUS.md was written (no 3x-slowdown, no
  3-failure condition was hit for any seed).

Total wall-clock, all 4 seeds: 208+176+172+190 = 746s (~12.4 min).
See REPORT.md for the full results, deviations and headline findings.

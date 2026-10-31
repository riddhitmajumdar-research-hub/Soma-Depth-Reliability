# RUNLOG — smoke, run smoke branch

Branch: created from `<base>` (<commit> "Snapshot for cloud runs (clean)"), pushed as `run smoke`.
Scope: read-only w.r.t. `Source/` (not modified). No other job started.

## 1. Git setup

```
git fetch origin
git checkout <base>      # <commit>
git checkout -b run smoke
```

## 2. Symlink to Mac-hardcoded path

Target: `<REPO_ROOT>`

```
sudo mkdir -p <REPO_PARENT>
sudo ln -sfn "$PWD" <REPO_ROOT>
```

**Result: succeeded** (sudo was available in this container). Symlink verified:
```
<REPO_ROOT> -> <REPO_ROOT>
```
Not needed for Lane 8 (its imports use relative paths, `sys.path.insert` off `os.path.dirname(__file__)`), but created in case any other script on this box depends on it.

## 3. Environment report

| Item | Value |
|---|---|
| `nproc` | 4 |
| RAM (total) | 15 Gi (`free -h`: 15Gi total, 15Gi available at time of check) |
| Python | 3.11.15 (`/usr/local/bin/python3`) |
| GPU | none — `nvidia-smi` not found; torch ran CPU-only by default |

Pre-install package check: **none of torch/numpy/scikit-learn/scipy were present** (all raised `ModuleNotFoundError`).

### Installs performed

| Package | Requested | Method that worked | Installed version |
|---|---|---|---|
| numpy | ==2.2.6 | plain `pip install numpy==2.2.6` | 2.2.6 |
| scipy | latest | plain `pip install scipy` | 1.17.1 |
| scikit-learn | latest | plain `pip install scikit-learn` | 1.9.1 |
| torch | ==2.14.0 CPU | **CPU-only index FAILED** (see below); fell back to plain PyPI | 2.14.0+cu130 |
| matplotlib | — (not in task's install list) | plain `pip install matplotlib` | 3.11.2 |

**Flaw / deviation #1 — torch CPU-only index unreachable.**
`pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu` failed: the outbound proxy in this container returned `403 Forbidden` on every retry against `download.pytorch.org`. Fell back to plain PyPI per the fallback rule, which succeeded but installed the **CUDA build** (`torch==2.14.0+cu130`), not a CPU-only wheel, because that's what plain PyPI ships by default. There is no GPU in this container (`nvidia-smi` not found), so torch runs on CPU regardless and results should not be affected, but this is a real deviation from "CPU" as specified in the protocol and is recorded (see requirement 5).

**Flaw / deviation #2 — undeclared dependency: matplotlib.**
The task's install list did not mention `matplotlib`, but the Lane 8 import chain (`lane8_harness.py` → `Source/degradation_aware_final_sweep.py` → `Source/physics_model.py`) does `import matplotlib.pyplot as plt` at module load time, purely for an unused plotting import in a file that otherwise only supplies `stiffness_classes`/`generate_signal`. The benchmark cannot run without it. Installed `matplotlib` (latest, 3.11.2) to unblock; this is a missing/undeclared dependency in the repo's own requirements chain for Lane 8, worth fixing (either add matplotlib to lane8's own requirements, or make the plotting import lazy in `physics_model.py` since Lane 8 is documented as headless).

All other named packages installed cleanly on the first attempt.

## 4. Lane 8 single-seed benchmark (seed 101)

Ran exactly as `audit_fixes/lane8/lane8_benchmark.py` does (unmodified script, unmodified `lane8_harness.py`/`lane8_config.py`):

```
python3 audit_fixes/lane8/lane8_benchmark.py
```

Output:
```
Seed 101 end-to-end wall-clock: 550.2s (train=0.6s, controls=122.0s, eval=427.4s)
Projected 24-seed run (linear extrapolation from this one seed): 13204s = 220.1min = 3.67hr
```

Raw result written by the script to `audit_fixes/lane8/logs/lane8_benchmark_result.json` (overwritten from the prior Mac run — the Mac numbers below were captured from git history/this report before the overwrite):

```json
{
  "bench_seed": 101,
  "wall_clock_seconds": 550.1807386875153,
  "train_seconds": 0.6081204414367676,
  "controls_seconds": 121.98216104507446,
  "eval_seconds": 427.43135166168213,
  "naive_floor_cm_rf": 13.636363636363635,
  "anomalous_recovery_rf": false,
  "projected_24_seeds_seconds": 13204.337728500366
}
```

### Comparison vs Mac (`audit_fixes/lane8/logs/lane8_benchmark_result.json`, pre-run baseline, 410 s total)

| Phase | Mac (s) | Cloud (s) | Ratio (run Mac) |
|---|---|---|---|
| train | 0.65 | 0.61 | 0.93x |
| controls | 142.19 | 121.98 | 0.86x |
| eval | 267.25 | 427.43 | **1.60x** |
| **total wall-clock** | **410.43** | **550.18** | **1.34x** |

**Verdict: MISMATCH on wall-clock (within the stop threshold).** Total is 1.34x the Mac time — well under the "stop if >3x" threshold, so the run was allowed to complete and is reported here rather than in STATUS.md. The mismatch is concentrated entirely in the `eval` phase (1.60x slower), while `train` and `controls` were actually *faster* than the Mac. Plausible cause: `eval` does per-depth signal generation + prediction over `RandomForestClassifier.predict` calls across 201 models (1 real + 200 shuffle controls) × 23 depths — likely sensitive to per-core clock speed / memory bandwidth differences between this container's 4 vCPUs and the Mac's CPU, rather than to core count (RF training/controls, which parallelize via `n_jobs=2`, were if anything faster here). Not investigated further per scope (single-seed smoke only).

### RF floor (this cloud run) vs certified CNN floor (Mac, `Source/logs/degradation_aware_final_results.csv`, seed 101)

| Quantity | Value | Source |
|---|---|---|
| Lane 8 RF naive floor, seed 101 (cloud, this run) | 13.6364 cm | `audit_fixes/lane8/logs/lane8_benchmark_result.json` (this run) |
| Certified CNN floor, seed 101 | 9.5455 cm | `Source/logs/degradation_aware_final_results.csv` (`101,9.545454545454545,8.181818181818182,False`) |

**Verdict: MISMATCH (13.636 cm RF vs 9.545 cm CNN) — but this is comparing two different models, not a reproducibility check.**

**Flaw / deviation #3 — the requested CNN-vs-RF comparison compares apples to oranges.** Lane 8 (`lane8_harness.py`) explicitly replaces `train_cnn` with `train_rf` (`RandomForestClassifier`, see `lane8_config.py` line 7-9: "New (not in original)... Lane 8 does not touch since it doesn't apply to RF"). The script only ever reports `naive_floor_cm_rf` — there is no CNN floor computed by `lane8_benchmark.py` at all. So comparing Lane 8's RF-based floor (13.64 cm) against `Source/logs/degradation_aware_final_results.csv`'s CNN-based floor (9.55 cm) is not a reproducibility check of the same computation on two machines — it's a comparison between two different classifiers evaluated under the same depth-sweep/FDR protocol. The ~4 cm gap says RF's naive degradation floor is coarser than the certified CNN's, which is expected (different model family, different capacity), not indicative of an environment or numerical problem. I'm reporting the numbers as instructed, but flagging that the *comparison itself*, as specified in the task, does not test what it appears to test (no MATCH is possible here even in principle, since no cloud CNN run was performed nor requested). If the intent was to verify RF-floor reproducibility across machines, the correct baseline would be a *previous RF Lane 8 run's* `naive_floor_cm_rf`, not the CNN CSV — none was available in this repo prior to this smoke run for direct RF-vs-RF comparison.

Note also: this run's RF floor (13.636363636363635) matches the *previous Mac* Lane 8 JSON's RF floor (`naive_floor_cm_rf: 13.636363636363635`, `anomalous_recovery_rf: false`) to full floating-point precision — i.e., the RF-vs-RF result (Mac vs cloud, same seed, same protocol) is an exact **MATCH**. That is the actually meaningful reproducibility signal in this run; the task's requested comparison target (the CNN CSV) is a different model and was never going to match.

## 5. Other flaws / notes surfaced in these requirements

- Requirement (4) says "for the certified CNN floor of seed 101... Report MATCH/MISMATCH" against a Random-Forest-only script's output — see deviation #3 above; the requirements conflate two different classifiers under one seed-101 floor comparison.
- Requirement (3)'s install list omits `matplotlib`, which is a hard runtime dependency of the Lane 8 import chain via `Source/physics_model.py` (deviation #2).
- Requirement (2)'s symlink path assumes `sudo` may or may not be present; it was present here, so the fallback path was not exercised, but is worth having tested in a future run.
- The pinned `torch==2.14.0` CPU-only index (`download.pytorch.org/whl/cpu`) is not reachable through this container's proxy (403 on all 5 retries) — if CPU-only wheels are a hard requirement for future runs, this environment's proxy allowlist needs `download.pytorch.org` added, or the fallback-to-PyPI CUDA build needs to be treated as acceptable when no GPU is present.
- `lane8_benchmark.py` overwrites `audit_fixes/lane8/logs/lane8_benchmark_result.json` in place with no backup of the prior (Mac) run — the original Mac numbers used for comparison above were captured from the task description / read before this run overwrote the file. Future benchmark scripts should write per-host output files rather than a single shared path, if cross-machine comparison is a recurring need.

## 6. Status

Completed within limits (1.34x Mac time, under the 3x/3-failure stop thresholds). No other job started. `Source/` untouched. Committing this RUNLOG.md and pushing to `run smoke` only, then stopping.

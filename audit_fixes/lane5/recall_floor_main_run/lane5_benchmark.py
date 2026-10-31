# Lane 5 -- Step 3: benchmark ONE seed end-to-end (train real model + 200
# shuffled-label controls + full per-class eval sweep) at 4 threads, then
# project totals for the three later sweeps. Does NOT start anything
# longer than one seed. Read-only w.r.t. Source/ and outputs/.

import os
import subprocess
import time
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Source"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # lane5_config.py lives one level up

import torch  # noqa: E402
torch.set_num_threads(4)

import lane5_harness as harness  # noqa: E402
import lane5_config as cfg       # noqa: E402

N_SEEDS = 24
N_DATASET_SIZES = 4
N_WINDOW_SETTINGS = 3

BENCH_SEED = 101  # first seed of the pre-registered set; not seed 121 (already special-cased in Step 2)


def resnet_sweep_active():
    try:
        out = subprocess.run(
            ["ps", "-p", "7429", "-o", "comm="], capture_output=True, text=True, timeout=5
        )
        return bool(out.stdout.strip())
    except Exception as e:
        return f"could not check (error: {e})"


def main():
    active_before = resnet_sweep_active()
    print(f"ResNet1D sweep (pid 7429) active before benchmark: {active_before}")

    t0 = time.time()
    row = harness.run_one_seed(BENCH_SEED)
    wall_clock = time.time() - t0

    active_after = resnet_sweep_active()
    print(f"ResNet1D sweep (pid 7429) active after benchmark:  {active_after}")

    print(f"\nSeed {BENCH_SEED} end-to-end wall-clock: {wall_clock:.1f}s "
          f"(train={row['train_seconds']:.1f}s, controls={row['controls_seconds']:.1f}s, "
          f"eval={row['eval_seconds']:.1f}s)")

    proj_24_seeds = wall_clock * N_SEEDS
    proj_dataset_sweep = wall_clock * N_DATASET_SIZES * N_SEEDS
    proj_window_sweep = wall_clock * N_WINDOW_SETTINGS * N_SEEDS

    def fmt(seconds):
        return f"{seconds:.0f}s = {seconds/60:.1f}min = {seconds/3600:.2f}hr"

    print("\nProjections (linear extrapolation from this one seed; see REPORT.md "
          "for caveats -- e.g. dataset-size sweep changes per-seed cost non-uniformly):")
    print(f"  (a) 24 seeds (this pre-registered run):                 {fmt(proj_24_seeds)}")
    print(f"  (b) dataset-size sweep, 4 sizes x 24 seeds:             {fmt(proj_dataset_sweep)}")
    print(f"  (c) window sweep, 3 settings x 24 seeds:                {fmt(proj_window_sweep)}")

    result = {
        "bench_seed": BENCH_SEED,
        "resnet_sweep_active_before": bool(active_before) if isinstance(active_before, (bool,)) else active_before,
        "resnet_sweep_active_after": bool(active_after) if isinstance(active_after, (bool,)) else active_after,
        "wall_clock_seconds": float(wall_clock),
        "train_seconds": float(row["train_seconds"]),
        "controls_seconds": float(row["controls_seconds"]),
        "eval_seconds": float(row["eval_seconds"]),
        "agg_floor_cm": float(row["agg_floor_cm"]),
        "original_agg_floor_cm": float(row["original_agg_floor_cm"]) if row["original_agg_floor_cm"] is not None else None,
        "floor_match": bool(row["floor_match"]),
        "projected_24_seeds_seconds": float(proj_24_seeds),
        "projected_dataset_sweep_seconds": float(proj_dataset_sweep),
        "projected_window_sweep_seconds": float(proj_window_sweep),
    }
    out_path = os.path.join(os.path.dirname(__file__), "logs", "lane5_benchmark_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved raw benchmark result to {out_path}")


if __name__ == "__main__":
    main()

# Lane 8 -- runtime estimate: benchmark ONE seed end-to-end (train real RF +
# 200 shuffled-label control RFs + full 23-depth eval sweep), then project
# the 24-seed total. Does NOT start anything longer than one seed -- the
# full run is launched separately, only after explicit go-ahead
# (see PREREG.md section 9). Read-only w.r.t. Source/.

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(__file__))

import lane8_harness as harness  # noqa: E402
import lane8_config as cfg       # noqa: E402

N_SEEDS = 24
BENCH_SEED = 101  # first seed of the pre-registered set


def main():
    t0 = time.time()
    row = harness.run_one_seed(BENCH_SEED)
    wall_clock = time.time() - t0

    print(f"Seed {BENCH_SEED} end-to-end wall-clock: {wall_clock:.1f}s "
          f"(train={row['train_seconds']:.1f}s, controls={row['controls_seconds']:.1f}s, "
          f"eval={row['eval_seconds']:.1f}s)")

    proj_24_seeds = wall_clock * N_SEEDS

    def fmt(seconds):
        return f"{seconds:.0f}s = {seconds/60:.1f}min = {seconds/3600:.2f}hr"

    print(f"\nProjected 24-seed run (linear extrapolation from this one seed): {fmt(proj_24_seeds)}")

    result = {
        "bench_seed": BENCH_SEED,
        "wall_clock_seconds": float(wall_clock),
        "train_seconds": float(row["train_seconds"]),
        "controls_seconds": float(row["controls_seconds"]),
        "eval_seconds": float(row["eval_seconds"]),
        "naive_floor_cm_rf": float(row["naive_floor_cm_rf"]),
        "anomalous_recovery_rf": bool(row["anomalous_recovery_rf"]),
        "projected_24_seeds_seconds": float(proj_24_seeds),
    }
    out_path = os.path.join(os.path.dirname(__file__), "logs", "lane8_benchmark_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved raw benchmark result to {out_path}")


if __name__ == "__main__":
    main()

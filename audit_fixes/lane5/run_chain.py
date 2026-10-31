# Lane 5 -- chain runner (v2, PID-independent).
#
# v1 blocked on the main run's OS PID -- but that run died silently
# with no crash trace, and nothing was watching its PID,
# so the death went unnoticed for ~3.75h. A PID can vanish for reasons
# that have nothing to do with "the 24 seeds are done" (a crash, a kill,
# a resume under a NEW pid) and nothing to do with "the 24 seeds are NOT
# done" (the process legitimately exiting after seed 24). Polling the
# actual completion signal -- 24 rows in the results CSV -- is correct
# either way and doesn't care which PID (if any) is currently running it.
#
# Blocks until audit_fixes/lane5/recall_floor_main_run/logs/lane5_per_class_results.csv has all
# 24 pre-registered seed rows, then launches, in order:
#   1) lane5_harness_clean.py (24-seed clean comparison sweep)
#   2) multiplicity_robustness.py --run main
#   3) multiplicity_robustness.py --run clean
# Does NOT launch or resume the main run itself -- see recall_floor_main_run/RESUME_COMMAND.txt
# for that (run manually from a normal terminal, per the protocol).

import os
import sys
import csv
import time
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_CSV_PATH = os.path.join(HERE, "recall_floor_main_run", "logs", "lane5_per_class_results.csv")
LOG_PATH = os.path.join(HERE, "recall_floor_main_run", "logs", "lane5_chain_run.log")

sys.path.insert(0, HERE)
import lane5_config as cfg  # noqa: E402

EXPECTED_SEEDS = set(cfg.SEEDS_TO_RUN)  # 101-124, 24 seeds
POLL_INTERVAL_SECONDS = 60


def log(msg):
    line = f"{datetime.datetime.now().isoformat()} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def completed_seeds():
    if not os.path.exists(MAIN_CSV_PATH):
        return set()
    done = set()
    with open(MAIN_CSV_PATH, "r") as f:
        for row in csv.DictReader(f):
            try:
                done.add(int(row["seed"]))
            except (KeyError, ValueError):
                pass
    return done


def main():
    log(f"Chain started (PID-independent). Polling {MAIN_CSV_PATH} every {POLL_INTERVAL_SECONDS}s "
        f"for all {len(EXPECTED_SEEDS)} seeds: {sorted(EXPECTED_SEEDS)}")

    while True:
        done = completed_seeds()
        missing = EXPECTED_SEEDS - done
        if not missing:
            log(f"All {len(EXPECTED_SEEDS)} seeds present in {MAIN_CSV_PATH}. Proceeding with chain.")
            break
        log(f"Waiting: {len(done)}/{len(EXPECTED_SEEDS)} seeds done. Missing: {sorted(missing)}")
        time.sleep(POLL_INTERVAL_SECONDS)

    log("Step 1/3: launching clean sweep (lane5_harness_clean.py)...")
    r1 = subprocess.run(["python3", "lane5_harness_clean.py"], cwd=os.path.join(HERE, "clean_trained_sweep"))
    log(f"Clean sweep exited with code {r1.returncode}")

    log("Step 2/3: multiplicity_robustness.py --run main")
    r2 = subprocess.run(["python3", "multiplicity_robustness.py", "--run", "main"], cwd=HERE)
    log(f"multiplicity_robustness (main) exited with code {r2.returncode}")

    log("Step 3/3: multiplicity_robustness.py --run clean")
    r3 = subprocess.run(["python3", "multiplicity_robustness.py", "--run", "clean"], cwd=HERE)
    log(f"multiplicity_robustness (clean) exited with code {r3.returncode}")

    log("Chain complete.")


if __name__ == "__main__":
    main()

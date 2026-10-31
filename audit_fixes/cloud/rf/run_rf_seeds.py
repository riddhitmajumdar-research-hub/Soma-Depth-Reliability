# rf cloud wrapper -- restricts the pre-registered 24-seed Lane 8 RF
# certification run (audit_fixes/lane8/) to a given seed sub-range, without
# editing any file under audit_fixes/lane8/ or Source/. Calls
# lane8_harness.run_one_seed() directly (bypassing lane8_harness.main(),
# which is hardcoded to cfg.SEEDS_TO_RUN = range(101, 125) and would run
# all 24 seeds). No protocol parameter (model config, eval grid, FDR alpha,
# shuffle count, random_state scheme) is changed -- this file only adds a
# seed-range filter and its own checkpoint/resume + CSV writer.
#
# Known side effect (not avoidable without editing lane8 files, so flagged
# here and in REPORT.md rather than silently worked around): the unmodified
# lane8_harness.run_one_seed() hardcodes its own per-seed detail CSV to
# audit_fixes/lane8/logs/lane8_depth_detail_seed{seed}.csv. That file is a
# side effect of the pre-registered harness itself, written by this run;
# it is left in the working tree but is NOT committed by this wrapper
# (only files under results/ and audit_fixes/cloud/rf/ are), so as not to
# treat "unavoidable output of the untouched harness" as "editing
# audit_fixes/lane8" -- but it does mean the local disk under
# audit_fixes/lane8/logs/ is written to. Documented, not hidden.

import csv
import os
import sys
import time
import argparse
import subprocess

LANE8_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "lane8")
sys.path.insert(0, LANE8_DIR)

import lane8_harness as harness  # noqa: E402 -- unmodified import
import lane8_config as cfg       # noqa: E402 -- unmodified import

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..", "..", "..")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
CLOUD_LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CLOUD_LOG_DIR, exist_ok=True)


def bh_adjusted_pvalues(p_values):
    """Standard Benjamini-Hochberg adjusted p-values (q-values), computed
    here (not present in lane8_harness / degradation_aware_final_sweep,
    which only stores a pass/fail significance flag from the same BH
    step-up procedure, not a numeric adjusted p). This is a post-hoc
    derived quantity from the harness's own raw p-values -- it does not
    change any protocol parameter or re-run anything."""
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    q = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = order[n - rank]
        raw_rank = n - rank + 1
        val = min(prev, p_values[i] * n / raw_rank)
        q[i] = val
        prev = val
    return q


def load_done_seeds(results_csv):
    done = set()
    if os.path.exists(results_csv):
        with open(results_csv, "r", newline="") as f:
            for row in csv.DictReader(f):
                done.add(int(row["seed"]))
    return done


RESULT_FIELDS = [
    "seed", "depth_cm", "real_acc", "p_value_raw", "p_value_bh_adjusted",
    "fdr_significant", "floor_cm", "last_significant_cm", "anomalous_recovery",
    "train_seconds", "controls_seconds", "eval_seconds", "total_seconds",
]


def append_seed_rows(results_csv, seed, summary_row, detail_rows):
    file_exists = os.path.exists(results_csv)
    p_raw = [float(r["p_value"]) for r in detail_rows]
    p_adj = bh_adjusted_pvalues(p_raw)
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not file_exists:
            writer.writeheader()
        for r, adj in zip(detail_rows, p_adj):
            writer.writerow({
                "seed": seed,
                "depth_cm": r["depth_cm"],
                "real_acc": r["real_acc"],
                "p_value_raw": r["p_value"],
                "p_value_bh_adjusted": adj,
                "fdr_significant": r["fdr_significant"],
                "floor_cm": summary_row["naive_floor_cm_rf"],
                "last_significant_cm": summary_row["last_significant_depth_cm_rf"],
                "anomalous_recovery": summary_row["anomalous_recovery_rf"],
                "train_seconds": summary_row["train_seconds"],
                "controls_seconds": summary_row["controls_seconds"],
                "eval_seconds": summary_row["eval_seconds"],
                "total_seconds": summary_row["total_seconds"],
            })


def git_commit_and_push(repo_root, results_csv, checkpoint_log, seed, branch):
    """Per-seed checkpoint commit: only the results CSV and this wrapper's
    own log file are staged (git add -f, since results/ and
    audit_fixes/cloud/ are not otherwise gitignored, -f is defensive).
    Nothing under audit_fixes/lane8/ or Source/ is staged here."""
    try:
        subprocess.run(["git", "add", "-f", results_csv, checkpoint_log], cwd=repo_root, check=True)
        commit = subprocess.run(
            ["git", "commit", "-m", f"rf seed {seed}: RF floor checkpoint"],
            cwd=repo_root, capture_output=True, text=True,
        )
        if commit.returncode != 0 and "nothing to commit" not in commit.stdout:
            print(f"WARN: git commit for seed {seed} returned {commit.returncode}: "
                  f"{commit.stdout} {commit.stderr}", flush=True)
            return
        push = subprocess.run(["git", "push", "origin", branch], cwd=repo_root, capture_output=True, text=True)
        if push.returncode != 0:
            print(f"WARN: git push for seed {seed} failed: {push.stderr}", flush=True)
    except Exception as e:
        print(f"WARN: git checkpoint for seed {seed} raised {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True, help="inclusive")
    ap.add_argument("--branch", type=str, required=True)
    args = ap.parse_args()

    all_seeds = [s for s in cfg.SEEDS_TO_RUN if args.start <= s <= args.end]
    results_csv = os.path.join(RESULTS_DIR, f"rf_floors_{args.start}-{args.end}.csv")
    checkpoint_log = os.path.join(CLOUD_LOG_DIR, f"rf_run_{args.start}-{args.end}.log")

    done_seeds = load_done_seeds(results_csv)
    remaining = [s for s in all_seeds if s not in done_seeds]

    with open(checkpoint_log, "a") as logf:
        logf.write(f"=== wrapper start: seeds {args.start}-{args.end}, "
                    f"already done {sorted(done_seeds)}, remaining {remaining} ===\n")
        logf.flush()

        for seed in remaining:
            t0 = time.time()
            logf.write(f"--- seed {seed} starting ---\n")
            logf.flush()

            row = harness.run_one_seed(seed)
            row["total_seconds"] = row["train_seconds"] + row["controls_seconds"] + row["eval_seconds"]

            detail_path = os.path.join(LANE8_DIR, "logs", f"lane8_depth_detail_seed{seed}.csv")
            with open(detail_path, "r", newline="") as f:
                detail_rows = list(csv.DictReader(f))

            append_seed_rows(results_csv, seed, row, detail_rows)

            elapsed = time.time() - t0
            logf.write(
                f"--- seed {seed} done in {elapsed:.1f}s. "
                f"floor={row['naive_floor_cm_rf']:.4f}cm "
                f"last_sig={row['last_significant_depth_cm_rf']} "
                f"anomalous_recovery={row['anomalous_recovery_rf']} ---\n"
            )
            logf.flush()
            print(f"SEED_DONE seed={seed} elapsed={elapsed:.1f}s floor={row['naive_floor_cm_rf']:.4f}cm",
                  flush=True)
            git_commit_and_push(REPO_ROOT, results_csv, checkpoint_log, seed, args.branch)

        logf.write(f"=== wrapper finished: seeds {args.start}-{args.end} ===\n")


if __name__ == "__main__":
    main()

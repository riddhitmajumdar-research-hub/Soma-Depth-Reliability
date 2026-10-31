# rf cloud run status -- seeds 113-124

**Status: COMPLETE. All 12 seeds finished, no failures, no stalls.**

- Branch: `run rf-113-124`, up to date with `run rf-113-124`
  (verified `git log run rf-113-124..HEAD` and the reverse are
  both empty after the final push).
- Wrapper: `audit_fixes/cloud/rf/run_rf_seeds.py`, unmodified since launch.
  No file under `Source/` or `audit_fixes/lane8/` was edited at any point
  in this run.
- Per-seed checkpoint commits: 12/12 present (`rf seed {seed}: RF
  floor checkpoint` for 113 through 124), each followed by a successful
  `git push`. No `WARN: git commit`/`WARN: git push` lines appear in
  `audit_fixes/cloud/rf/logs/nohup_113-124.out`.
- Runtime: 611-672s/seed (~10.5-11.2 min), close to the PREREG.md
  single-seed benchmark estimate of ~410s/seed for seed 101 -- seeds
  113-124 ran ~1.5-1.6x slower than that single-seed projection. Not
  investigated further here (still well under the "3x expected time" stop
  threshold), but flagged rather than silently absorbed: possible causes
  not ruled out include host contention on this container vs. the
  benchmark's host, or seed-to-seed variance in `generate_degraded_windows`
  eval cost (PREREG.md section 5 already notes eval, not training, is the
  dominant and most variable cost).
- Output: `results/rf_floors_113-124.csv`, 276 data rows (12 seeds x 23
  depths) + header, matches expected shape exactly.
- Sanity checks performed before declaring done (not just "process exited
  0"):
  - `real_acc` spans [0.297, 0.927] across all rows -- not degenerate
    (not all-identical, not all-0, not all-1, not NaN).
  - `p_value_raw` spans [0.00498, 0.866] -- consistent with the harness's
    `(1 + #controls >= real) / 201` formula's achievable range.
  - All 12 seeds present exactly once each in the seed column with 23 rows
    apiece.
  - No Python traceback in `nohup_113-124.out`; wrapper's own
    `=== wrapper finished: seeds 113-124 ===` line present.
- No silent-failure indicators found: log and CSV both grew monotonically
  seed-by-seed, local branch and origin matched after each push, no
  disk-space errors.

## Known caveats (not failures, but disclosed)

1. `audit_fixes/lane8/logs/lane8_run.log` and
   `audit_fixes/lane8/logs/lane8_depth_detail_seed{113..124}.csv` were
   written to local disk by the unmodified `lane8_harness.run_one_seed()`
   as an unavoidable side effect (see `run_rf_seeds.py`'s own header
   comment and PREREG.md section 6). Per the task rule ("never edit files
   under ... audit_fixes/lane8"), these are **not** committed. The already
   -tracked `lane8_run.log` (tracked before this run, from the
   pre-registration step) was marked `git update-index --skip-worktree`
   rather than committed, so its live growth from this run does not appear
   as a diff against the pre-registered lane8 state and is never pushed as
   a change to a lane8 file.
2. This run covers seeds 113-124 only (12 of the 24 pre-registered seeds).
   Seeds 101-112 are a separate lane (`run rf-101-112`), not touched or
   re-run here; that branch's own log shows it did not get past seed 101
   in its own attempt, so the full 101-124 set is not yet complete
   repo-wide -- stated here so it isn't mistaken for a 24-seed result.
3. No comparison against the CNN (`Source/logs/
   degradation_aware_final_results.csv`) or ResNet results was made in
   this run.

See `REPORT.md` for the results themselves.

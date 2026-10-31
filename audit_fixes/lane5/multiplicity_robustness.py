# Lane 5 -- multiplicity_robustness.py
# Post-processing ONLY over already-saved per-control results. Trains
# nothing, imports no training/eval functions, touches no running
# process. Usage:
#   python3 multiplicity_robustness.py --run main    # the 24-seed degradation-aware run
#   python3 multiplicity_robustness.py --run clean   # the clean comparison sweep
#
# For each seed, computes the aggregate (accuracy) floor under:
#   (a) BH within seed               -- already what the harness computed live
#   (b) BH pooled across seeds x depths -- one family of 24*23 tests
#   (c) Westfall-Young max-T at 0.05 -- FWER control via each control's
#       own max-over-depths accuracy as the resampling null
# and the same three for per-class recall WHERE the underlying per-depth
# raw p-values / real values were actually persisted by that run's
# harness (see "DATA AVAILABILITY" section below -- this is NOT available
# for the main run, only for the clean sweep, and the script says so
# rather than fabricating numbers).

import os
import sys
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Source"))
from degradation_aware_final_sweep import bh_fdr_correction  # noqa: E402 -- unmodified import
from physics_model import stiffness_classes  # noqa: E402

HERE = os.path.dirname(__file__)
MAIN_DIR = "recall_floor_main_run"
CLEAN_DIR = os.path.join("clean_trained_sweep", "clean_sweep")
CLASS_NAMES_SORTED = sorted(stiffness_classes.keys())
N_SHUFFLES = 200
ALPHA = 0.05


def bh_adjusted_pvalues(p_values):
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    raw_adj = ranked * n / np.arange(1, n + 1)
    adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0, 1)
    adjusted = np.empty(n)
    adjusted[order] = adj_sorted
    return adjusted


def reconstruct_real_stat(control_arr, p_value, n_shuffles=N_SHUFFLES):
    """Exact-ish reconstruction of the real model's per-depth statistic
    from the saved control array + the already-computed empirical
    p-value: p = (1 + k) / (n_shuffles + 1), where k = #{controls >=
    real_stat}. control_arr and real_stat live on the same quantized grid
    (both are accuracy/recall over a fixed, equal sample count), so k
    pins down real_stat to the k-th-largest control value in the typical
    (tie-at-boundary) case. Returns (value, k, ambiguous) -- ambiguous is
    True when ties push the actual count away from k, meaning the true
    value could be anywhere on the grid up to the next distinct control
    value down; NOT silently resolved, flagged in every output row."""
    sorted_desc = np.sort(control_arr)[::-1]
    k = int(round(p_value * (n_shuffles + 1) - 1))
    if k <= 0:
        # real stat exceeds every control; report control max as a lower bound
        return float(sorted_desc[0]), 0, False
    k = min(k, n_shuffles)
    candidate = float(sorted_desc[k - 1])
    actual_k = int(np.sum(control_arr >= candidate))
    ambiguous = actual_k != k
    return candidate, k, ambiguous


def load_seed_list(csv_path):
    seeds = []
    with open(csv_path, "r") as f:
        for row in csv.DictReader(f):
            seeds.append(int(row["seed"]))
    return sorted(seeds)


def load_aggregate_depth_detail(log_dir, prefix, seed):
    path = os.path.join(log_dir, f"{prefix}_aggregate_depth_detail_seed{seed}.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            rows.append({
                "depth_cm": float(row["depth_cm"]),
                "raw_p_value": float(row["raw_p_value"]),
                "bh_adjusted_p_value": float(row["bh_adjusted_p_value"]),
                "fdr_significant": row["fdr_significant"] == "True",
            })
    return sorted(rows, key=lambda r: r["depth_cm"])


def load_control_acc(controls_dir, seed, depth_cm):
    path = os.path.join(controls_dir, f"seed{seed}_depth{depth_cm:.2f}_ACC.npy")
    if os.path.exists(path):
        return np.load(path)
    # Main run never saved an _ACC.npy directly -- reconstruct it exactly
    # from the three per-class recall arrays. Valid because
    # N_EVAL_SAMPLES_PER_CELL=300 is split into exactly equal n_per_class
    # (100 each of 3 classes), so overall accuracy == unweighted mean of
    # the three per-class recalls, an algebraic identity, not an estimate.
    class_arrs = []
    for name in CLASS_NAMES_SORTED:
        p = os.path.join(controls_dir, f"seed{seed}_depth{depth_cm:.2f}_{name}.npy")
        if not os.path.exists(p):
            return None
        class_arrs.append(np.load(p))
    return np.mean(np.stack(class_arrs), axis=0)


def floor_from_sig(depths, sig_flags):
    for d, s in zip(depths, sig_flags):
        if not s:
            return d
    return depths[-1]


def analyze_run(run_name, log_dir, controls_dir, csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    seeds = load_seed_list(csv_path)
    print(f"[{run_name}] {len(seeds)} seeds found in {csv_path}: {seeds}")

    per_seed_detail = {}
    for seed in seeds:
        detail = load_aggregate_depth_detail(log_dir, f"lane5{'_clean' if run_name=='clean' else ''}", seed)
        if detail is None:
            print(f"  WARNING: no aggregate depth-detail file for seed {seed} -- skipping")
            continue
        per_seed_detail[seed] = detail

    if not per_seed_detail:
        print(f"[{run_name}] No aggregate depth-detail files found. Nothing to analyze.")
        return

    depths = [r["depth_cm"] for r in next(iter(per_seed_detail.values()))]

    # --- (a) BH within seed: already computed live by the harness; just re-read it. ---
    floors_a = {}
    for seed, detail in per_seed_detail.items():
        floors_a[seed] = floor_from_sig([r["depth_cm"] for r in detail], [r["fdr_significant"] for r in detail])

    # --- (b) BH pooled across seeds x depths: one family of len(seeds)*len(depths) tests. ---
    pooled_p = []
    pooled_index = []  # (seed, depth_cm)
    for seed, detail in per_seed_detail.items():
        for r in detail:
            pooled_p.append(r["raw_p_value"])
            pooled_index.append((seed, r["depth_cm"]))
    pooled_sig = bh_fdr_correction(pooled_p, alpha=ALPHA)
    pooled_sig_by_seed = {seed: {} for seed in per_seed_detail}
    for (seed, depth_cm), sig in zip(pooled_index, pooled_sig):
        pooled_sig_by_seed[seed][depth_cm] = bool(sig)
    floors_b = {}
    for seed, detail in per_seed_detail.items():
        depths_s = [r["depth_cm"] for r in detail]
        sig_s = [pooled_sig_by_seed[seed][d] for d in depths_s]
        floors_b[seed] = floor_from_sig(depths_s, sig_s)

    # --- (c) Westfall-Young max-T at 0.05, per seed. ---
    floors_c = {}
    min_p_cell_counts = {seed: 0 for seed in per_seed_detail}
    ambiguous_reconstructions = []
    for seed, detail in per_seed_detail.items():
        depths_s = [r["depth_cm"] for r in detail]
        control_matrix = []  # [n_controls, n_depths]
        real_stats = []
        any_missing = False
        for r in detail:
            arr = load_control_acc(controls_dir, seed, r["depth_cm"])
            if arr is None:
                any_missing = True
                break
            control_matrix.append(arr)
            real_val, k, ambiguous = reconstruct_real_stat(arr, r["raw_p_value"])
            real_stats.append(real_val)
            if ambiguous:
                ambiguous_reconstructions.append((seed, r["depth_cm"]))
            if k == 0:
                min_p_cell_counts[seed] += 0  # exceeds-all-controls case, not a min-p cell
            elif r["raw_p_value"] <= 1.0 / (N_SHUFFLES + 1) + 1e-12:
                min_p_cell_counts[seed] += 1
        if any_missing:
            print(f"  WARNING: control accuracy arrays missing for seed {seed} -- skipping Westfall-Young for this seed")
            continue
        control_matrix = np.stack(control_matrix, axis=1)  # [n_controls, n_depths]
        max_over_depths = control_matrix.max(axis=1)  # [n_controls] -- the max-T null
        wy_sig = []
        for depth_idx, real_val in enumerate(real_stats):
            adj_p = (np.sum(max_over_depths >= real_val) + 1) / (N_SHUFFLES + 1)
            wy_sig.append(adj_p <= ALPHA)
        floors_c[seed] = floor_from_sig(depths_s, wy_sig)

    # --- Write per-seed comparison table. ---
    out_csv = os.path.join(out_dir, f"multiplicity_robustness_{run_name}.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "floor_bh_within_seed_cm", "floor_bh_pooled_cm", "floor_westfall_young_cm", "floor_changed_b_vs_a", "floor_changed_c_vs_a"])
        w.writeheader()
        for seed in sorted(per_seed_detail.keys()):
            fa = floors_a.get(seed)
            fb = floors_b.get(seed)
            fc = floors_c.get(seed)
            w.writerow({
                "seed": seed, "floor_bh_within_seed_cm": fa, "floor_bh_pooled_cm": fb,
                "floor_westfall_young_cm": fc,
                "floor_changed_b_vs_a": (fb is not None and fa is not None and abs(fb - fa) > 1e-9),
                "floor_changed_c_vs_a": (fc is not None and fa is not None and abs(fc - fa) > 1e-9),
            })

    def summarize(name, floors_dict):
        vals = [v for v in floors_dict.values() if v is not None]
        if not vals:
            return f"  {name}: no data"
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        return f"  {name}: n={len(vals)} mean={mean:.3f}cm sd={sd:.3f}cm"

    print(f"\n[{run_name}] AGGREGATE (accuracy) floor summary:")
    print(summarize("(a) BH within seed", floors_a))
    print(summarize("(b) BH pooled across seeds x depths", floors_b))
    print(summarize("(c) Westfall-Young max-T @ 0.05", floors_c))

    changed_b = sum(1 for s in per_seed_detail if floors_a.get(s) is not None and floors_b.get(s) is not None and abs(floors_a[s] - floors_b[s]) > 1e-9)
    changed_c = sum(1 for s in per_seed_detail if floors_a.get(s) is not None and floors_c.get(s) is not None and abs(floors_a[s] - floors_c[s]) > 1e-9)
    print(f"  Seeds whose floor changed under (b) vs (a): {changed_b}/{len(per_seed_detail)}")
    print(f"  Seeds whose floor changed under (c) vs (a): {changed_c}/{len(per_seed_detail)}")

    total_cells = len(per_seed_detail) * len(depths)
    min_p_cells = sum(min_p_cell_counts.values())
    print(f"  Cells at the minimum possible p-value (1/201 = {1/(N_SHUFFLES+1):.5f}): {min_p_cells}/{total_cells}")
    if ambiguous_reconstructions:
        print(f"  WARNING: {len(ambiguous_reconstructions)} (seed, depth) real-accuracy reconstructions were "
              f"AMBIGUOUS (tie-inflated control count) -- Westfall-Young results for those cells carry extra "
              f"uncertainty. First few: {ambiguous_reconstructions[:5]}")
    print(f"  Saved per-seed detail to {out_csv}")

    # --- Per-class recall: only if the run's harness actually persisted the
    # needed per-depth per-class p-values / real recalls. ---
    pc_path = os.path.join(log_dir, f"lane5{'_clean' if run_name=='clean' else ''}_per_class_depth_detail_seed{seeds[0]}.csv")
    if run_name == "main" or not os.path.exists(pc_path):
        print(f"\n[{run_name}] PER-CLASS RECALL: NOT COMPUTABLE from saved artifacts.")
        print("  The main run's harness (lane5_harness.py) computes per-class per-depth")
        print("  raw p-values and real recall values in memory each seed, but only ever")
        print("  persists the FINAL FLOOR (first non-significant depth) per class per")
        print("  seed -- the per-depth p-values and real recall values themselves are")
        print("  discarded and never written to disk. There is nothing to reconstruct")
        print("  this from (unlike the aggregate case, no p-values survive at all for")
        print("  per-class). This is a genuine gap discovered in this pass, not")
        print("  resolved here -- flagged in REPORT.md. It has been fixed forward in")
        print("  lane5_harness_clean.py (the clean sweep), which DOES persist this")
        print("  detail (lane5_clean_per_class_depth_detail_seed{seed}.csv), so this")
        print("  analysis IS available when run with --run clean after that sweep runs.")
        return

    analyze_per_class(run_name, log_dir, controls_dir, seeds, depths, out_dir)


def analyze_per_class(run_name, log_dir, controls_dir, seeds, depths, out_dir):
    per_seed_pc = {}
    for seed in seeds:
        path = os.path.join(log_dir, f"lane5_clean_per_class_depth_detail_seed{seed}.csv")
        if not os.path.exists(path):
            continue
        rows = []
        with open(path, "r") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        per_seed_pc[seed] = sorted(rows, key=lambda r: float(r["depth_cm"]))

    if not per_seed_pc:
        print(f"\n[{run_name}] No per-class depth-detail files found yet.")
        return

    out_csv = os.path.join(out_dir, f"multiplicity_robustness_{run_name}_per_class.csv")
    with open(out_csv, "w", newline="") as f:
        fieldnames = ["seed", "class", "floor_bh_within_seed_cm", "floor_bh_pooled_cm", "floor_westfall_young_cm"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for name in CLASS_NAMES_SORTED:
            # (a) within seed
            floors_a = {}
            for seed, rows in per_seed_pc.items():
                sig = [row[f"{name}_fdr_significant"] == "True" for row in rows]
                d = [float(row["depth_cm"]) for row in rows]
                floors_a[seed] = floor_from_sig(d, sig)

            # (b) pooled across seeds x depths, per class
            pooled_p, pooled_index = [], []
            for seed, rows in per_seed_pc.items():
                for row in rows:
                    pooled_p.append(float(row[f"{name}_raw_p_value"]))
                    pooled_index.append((seed, float(row["depth_cm"])))
            pooled_sig = bh_fdr_correction(pooled_p, alpha=ALPHA)
            pooled_sig_by_seed = {seed: {} for seed in per_seed_pc}
            for (seed, depth_cm), sig in zip(pooled_index, pooled_sig):
                pooled_sig_by_seed[seed][depth_cm] = bool(sig)
            floors_b = {
                seed: floor_from_sig([float(r["depth_cm"]) for r in rows], [pooled_sig_by_seed[seed][float(r["depth_cm"])] for r in rows])
                for seed, rows in per_seed_pc.items()
            }

            # (c) Westfall-Young per class
            floors_c = {}
            for seed, rows in per_seed_pc.items():
                control_matrix = []
                real_vals = []
                missing = False
                for row in rows:
                    depth_cm = float(row["depth_cm"])
                    arr_path = os.path.join(controls_dir, f"seed{seed}_depth{depth_cm:.2f}_{name}.npy")
                    if not os.path.exists(arr_path):
                        missing = True
                        break
                    control_matrix.append(np.load(arr_path))
                    real_vals.append(float(row[f"{name}_recall_real"]))
                if missing:
                    continue
                control_matrix = np.stack(control_matrix, axis=1)
                max_over_depths = control_matrix.max(axis=1)
                sig = [(np.sum(max_over_depths >= rv) + 1) / (N_SHUFFLES + 1) <= ALPHA for rv in real_vals]
                floors_c[seed] = floor_from_sig([float(r["depth_cm"]) for r in rows], sig)

            for seed in sorted(per_seed_pc.keys()):
                w.writerow({
                    "seed": seed, "class": name,
                    "floor_bh_within_seed_cm": floors_a.get(seed),
                    "floor_bh_pooled_cm": floors_b.get(seed),
                    "floor_westfall_young_cm": floors_c.get(seed),
                })

            def summ(d):
                vals = [v for v in d.values() if v is not None]
                return f"mean={np.mean(vals):.3f}cm sd={np.std(vals, ddof=1) if len(vals)>1 else 0:.3f}cm" if vals else "no data"

            print(f"  [{name}] (a) within-seed: {summ(floors_a)} | (b) pooled: {summ(floors_b)} | (c) WY: {summ(floors_c)}")

    print(f"  Saved per-class detail to {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["main", "clean"], default="main")
    args = parser.parse_args()

    if args.run == "main":
        analyze_run(
            "main",
            log_dir=os.path.join(HERE, MAIN_DIR, "logs"),
            controls_dir=os.path.join(HERE, MAIN_DIR, "controls"),
            csv_path=os.path.join(HERE, MAIN_DIR, "logs", "lane5_per_class_results.csv"),
            out_dir=os.path.join(HERE, MAIN_DIR, "logs"),
        )
    else:
        analyze_run(
            "clean",
            log_dir=os.path.join(HERE, CLEAN_DIR, "logs"),
            controls_dir=os.path.join(HERE, CLEAN_DIR, "controls"),
            csv_path=os.path.join(HERE, CLEAN_DIR, "logs", "lane5_clean_per_class_results.csv"),
            out_dir=os.path.join(HERE, CLEAN_DIR, "logs"),
        )


if __name__ == "__main__":
    main()

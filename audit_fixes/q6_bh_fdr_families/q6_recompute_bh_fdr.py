"""
Audit fix lane: q6_bh_fdr_families
Q6: recompute BH-FDR for the clean CNN under three families and report the
floor under each. READ-ONLY on all pre-existing repo files -- this script
only reads:
  - Source/results/stage4_cnn_results.pkl        (floor_characterization_cnn.py's
    saved per-cell results: 82 cells = 4 freqs x {12,24,23,23} depths, fixed
    noise at SENSOR_SNR_DB)
  - outputs/stage4/data/noise_robustness_stress_test_clean_cnn.csv (clean-CNN
    SNR-crosscheck script's saved per-cell results: 5 SNR levels x 23 depths,
    fixed at 100Hz = constants.NOMINAL_FREQ_HZ)
Constants (SENSOR_SNR_DB, NOMINAL_FREQ_HZ) are imported from Source/constants.py,
not hardcoded.

No retraining. No existing file is modified.
"""
import csv
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "Source"))
from constants import SENSOR_SNR_DB, NOMINAL_FREQ_HZ  # noqa: E402

PKL_PATH = REPO_ROOT / "Source" / "results" / "stage4_cnn_results.pkl"
CSV_PATH = REPO_ROOT / "outputs" / "stage4" / "data" / "noise_robustness_stress_test_clean_cnn.csv"

OUT_DIR = Path(__file__).resolve().parent


def bh_fdr_correction(p_values, alpha=0.05):
    """Standard Benjamini-Hochberg step-up procedure. Re-implemented here
    (not imported) so this is an independent recomputation, but matches the
    same definition used in floor_characterization_cnn.py and
    noise_robustness_stress_test_clean_cnn.py: sort ascending, find the
    largest rank k with p_(k) <= (k/n)*alpha, reject all ranks <= k."""
    import numpy as np
    p_values = np.array(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool), thresholds, order
    max_passed_rank = int(np.max(np.where(passed)[0]))
    significant = np.zeros(n, dtype=bool)
    significant[order[:max_passed_rank + 1]] = True
    return significant, thresholds, order


def floor_from_family(rows_sorted_by_depth, sig_by_row):
    """First depth (ascending) that is NOT BH-significant = floor.
    Matches the floor definition already used in both source scripts."""
    for row, sig in zip(rows_sorted_by_depth, sig_by_row):
        if not sig:
            return row["depth_cm"]
    return None


def main():
    report = {}

    # ---- Load floor_characterization_cnn.py's saved pickle (82 cells) ----
    with open(PKL_PATH, "rb") as f:
        pkl_results = pickle.load(f)
    report["pkl_path"] = str(PKL_PATH.relative_to(REPO_ROOT))
    report["pkl_total_cells"] = len(pkl_results)

    freq_counts = {}
    for r in pkl_results:
        freq_counts[r["freq"]] = freq_counts.get(r["freq"], 0) + 1
    report["pkl_cells_by_freq"] = freq_counts

    # ---- Load clean-CNN SNR-crosscheck CSV ----
    with open(CSV_PATH, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    for r in csv_rows:
        r["snr_db"] = float(r["snr_db"])
        r["depth_cm"] = float(r["depth_cm"])
        r["p_value"] = float(r["p_value"])
        r["fdr_significant_ORIGINAL"] = r["fdr_significant"]
    report["csv_path"] = str(CSV_PATH.relative_to(REPO_ROOT))
    report["csv_total_rows"] = len(csv_rows)

    snr_levels = sorted(set(r["snr_db"] for r in csv_rows))
    report["csv_snr_levels"] = snr_levels

    target_snr = SENSOR_SNR_DB
    match_snr = min(snr_levels, key=lambda s: abs(s - target_snr))
    report["SENSOR_SNR_DB_from_constants"] = target_snr
    report["closest_csv_snr_level"] = match_snr
    report["snr_exact_match"] = "MATCH" if abs(match_snr - target_snr) < 1e-9 else "MISMATCH"

    report["NOMINAL_FREQ_HZ_from_constants"] = NOMINAL_FREQ_HZ

    # =========================================================
    # Family (a): pooled 82-cell family (all 4 freqs, floor_characterization_cnn.py)
    # =========================================================
    pooled_sorted_for_check = sorted(pkl_results, key=lambda r: (r["freq"], r["depth_cm"]))
    p_pooled = [r["p_value"] for r in pooled_sorted_for_check]
    sig_pooled, thr_pooled, order_pooled = bh_fdr_correction(p_pooled)

    pooled_by_freq_floor = {}
    for freq in sorted(freq_counts.keys()):
        idxs = [i for i, r in enumerate(pooled_sorted_for_check) if r["freq"] == freq]
        rows_f = sorted([pooled_sorted_for_check[i] for i in idxs], key=lambda r: r["depth_cm"])
        sig_f = [sig_pooled[pooled_sorted_for_check.index(r)] for r in rows_f]
        pooled_by_freq_floor[freq] = floor_from_family(rows_f, sig_f)

    report["family_A_pooled_82cell"] = {
        "n_cells": len(p_pooled),
        "n_rejected": int(sum(sig_pooled)),
        "floor_by_freq_cm": {str(k): (float(v) if v is not None else None) for k, v in pooled_by_freq_floor.items()},
        "floor_at_100Hz_under_pooled_correction_cm": (
            float(pooled_by_freq_floor[100]) if pooled_by_freq_floor[100] is not None else None
        ),
    }

    # =========================================================
    # Family (b): 100Hz-only family (24 cells) -- floor_characterization_cnn.py subset
    # =========================================================
    rows_100 = sorted([r for r in pkl_results if r["freq"] == 100], key=lambda r: r["depth_cm"])
    p_100 = [r["p_value"] for r in rows_100]
    sig_100, thr_100, order_100 = bh_fdr_correction(p_100)
    floor_100 = floor_from_family(rows_100, sig_100)

    report["family_B_100Hz_only_24cell"] = {
        "n_cells": len(p_100),
        "n_rejected": int(sum(sig_100)),
        "floor_cm": float(floor_100) if floor_100 is not None else None,
        "per_depth": [
            {"depth_cm": float(r["depth_cm"]), "p_value": float(r["p_value"]), "bh_reject": bool(s)}
            for r, s in zip(rows_100, sig_100)
        ],
    }

    # =========================================================
    # Family (c): 23-depth per-SNR family at match_snr (~SENSOR_SNR_DB), 100Hz fixed
    # This is the clean-CNN SNR-crosscheck script's OWN per-SNR grouping.
    # =========================================================
    rows_snr = sorted([r for r in csv_rows if r["snr_db"] == match_snr], key=lambda r: r["depth_cm"])
    p_snr = [r["p_value"] for r in rows_snr]
    sig_snr, thr_snr, order_snr = bh_fdr_correction(p_snr)
    floor_snr = floor_from_family(rows_snr, sig_snr)

    n = len(p_snr)
    # BH threshold table: sort ascending, pair rank k with threshold (k/n)*alpha, mark reject
    sort_idx = sorted(range(n), key=lambda i: p_snr[i])
    ranked_table = []
    for rank, i in enumerate(sort_idx, start=1):
        thr = (rank / n) * 0.05
        ranked_table.append({
            "rank": rank,
            "depth_cm": float(rows_snr[i]["depth_cm"]),
            "p_value": float(p_snr[i]),
            "bh_threshold_k_over_n_times_alpha": thr,
            "p_leq_threshold": bool(p_snr[i] <= thr),
        })

    per_depth_c = [
        {"depth_cm": float(r["depth_cm"]), "p_value": float(r["p_value"]),
         "bh_reject_RECOMPUTED": bool(s), "fdr_significant_ORIGINAL_CSV": r["fdr_significant_ORIGINAL"]}
        for r, s in zip(rows_snr, sig_snr)
    ]
    # cross-check recomputed vs the CSV's own saved fdr_significant column
    mismatches = [
        d for d in per_depth_c
        if d["bh_reject_RECOMPUTED"] != (str(d["fdr_significant_ORIGINAL_CSV"]).strip() == "True")
    ]

    report["family_C_23depth_per_SNR"] = {
        "snr_db_used": match_snr,
        "n_cells": n,
        "n_rejected": int(sum(sig_snr)),
        "floor_cm": float(floor_snr) if floor_snr is not None else None,
        "per_depth": per_depth_c,
        "ranked_bh_table": ranked_table,
        "recomputed_vs_original_csv_column": "MATCH" if not mismatches else "MISMATCH",
        "mismatches": mismatches,
    }

    out_json = OUT_DIR / "q6_bh_fdr_results.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ---- human-readable summary to stdout (captured to a log file too) ----
    print("=" * 78)
    print("Q6 -- BH-FDR recomputation, three families")
    print("=" * 78)
    print(f"SENSOR_SNR_DB (constants.py) = {target_snr}")
    print(f"Closest SNR level actually present in the clean-CNN CSV: {match_snr}  [{report['snr_exact_match']}]")
    print(f"NOMINAL_FREQ_HZ (constants.py) = {NOMINAL_FREQ_HZ}")
    print()
    print(f"Family A -- pooled 82-cell (floor_characterization_cnn.py, all 4 freqs, one BH pass over all 82):")
    print(f"  rejected: {report['family_A_pooled_82cell']['n_rejected']}/82")
    print(f"  floor by freq (cm): {report['family_A_pooled_82cell']['floor_by_freq_cm']}")
    print()
    print(f"Family B -- 100Hz-only (24 cells, BH pass restricted to the 100Hz subset):")
    print(f"  rejected: {report['family_B_100Hz_only_24cell']['n_rejected']}/24")
    print(f"  floor: {report['family_B_100Hz_only_24cell']['floor_cm']} cm")
    print()
    print(f"Family C -- 23-depth per-SNR (clean-CNN crosscheck CSV, snr={match_snr}, 100Hz fixed):")
    print(f"  rejected: {report['family_C_23depth_per_SNR']['n_rejected']}/23")
    print(f"  floor: {report['family_C_23depth_per_SNR']['floor_cm']} cm")
    print(f"  recomputed vs CSV's own saved fdr_significant column: {report['family_C_23depth_per_SNR']['recomputed_vs_original_csv_column']}")
    print()
    print("Per-depth p-values and BH outcome, Family C (23-depth, snr={:.4f}):".format(match_snr))
    for d in sorted(per_depth_c, key=lambda x: x["depth_cm"]):
        print(f"  depth={d['depth_cm']:7.4f}cm  p={d['p_value']:.6f}  "
              f"reject(recomputed)={d['bh_reject_RECOMPUTED']}  "
              f"orig_csv_flag={d['fdr_significant_ORIGINAL_CSV']}")
    print()
    print("BH ranked table (ascending p), Family C:")
    for row in ranked_table:
        print(f"  rank={row['rank']:2d}  depth={row['depth_cm']:7.4f}cm  p={row['p_value']:.6f}  "
              f"threshold=(k/n)*alpha={row['bh_threshold_k_over_n_times_alpha']:.6f}  "
              f"p<=threshold={row['p_leq_threshold']}")

    print()
    print(f"Wrote: {out_json.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

# Lane 5 -- pre-registered per-class certification harness.
# See audit_fixes/lane5/recall_floor_main_run/PREREG.md for the full protocol and disclosed
# judgment calls. This script BUILDS the harness; it does not launch the
# full 24-seed run (that requires explicit go-ahead -- see REPORT.md).
#
# Imports training/eval-data/FDR logic UNCHANGED from
# Source/degradation_aware_final_sweep.py -- does not reimplement it.
# Read-only w.r.t. everything under Source/.

import os
import sys
import csv
import time
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "Source"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # lane5_config.py lives one level up

from degradation_aware_final_sweep import (   # noqa: E402  -- unmodified imports
    build_degradation_aware_dataset,
    train_cnn,
    bh_fdr_correction,
    generate_degraded_windows,
    EVAL_DEPTHS_CM,
    FREQ,
)
from physics_model import stiffness_classes   # noqa: E402
from sklearn.preprocessing import LabelEncoder

import lane5_config as cfg

torch.set_num_threads(cfg.N_TORCH_THREADS)

HERE = os.path.dirname(__file__)
LOG_DIR = os.path.join(HERE, "logs")
WEIGHTS_DIR = os.path.join(HERE, "seed_weights")
CONTROLS_DIR = os.path.join(HERE, "controls")
CSV_PATH = os.path.join(LOG_DIR, "lane5_per_class_results.csv")
LOG_PATH = os.path.join(LOG_DIR, "lane5_run.log")
HEARTBEAT_PATH = os.path.join(LOG_DIR, "heartbeat.txt")
ORIGINAL_CSV_PATH = os.path.join(HERE, "..", "..", "..", "Source", "logs", "degradation_aware_final_results.csv")

for d in (LOG_DIR, WEIGHTS_DIR, CONTROLS_DIR):
    os.makedirs(d, exist_ok=True)


def touch_heartbeat(note=""):
    """Added after the main run died silently for ~3.75h with no trace
    (see REPORT.md) -- writes a timestamp file after every control batch
    so check_alive.sh can detect a hang/death quickly instead of only via
    process-list absence, which was missed for hours last time."""
    with open(HEARTBEAT_PATH, "w") as f:
        f.write(f"{time.time()} {time.strftime('%Y-%m-%d %H:%M:%S')} {note}\n")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lane5")

CLASS_NAMES_SORTED = sorted(stiffness_classes.keys())


def bh_adjusted_pvalues(p_values):
    """Standard Benjamini-Hochberg step-up ADJUSTED p-values (alpha-free;
    complementary to the imported bh_fdr_correction, which only returns a
    boolean significance flag at a fixed alpha). Same family/ordering
    convention as bh_fdr_correction: p_(1) <= ... <= p_(n),
    q_(i) = min_{j>=i} (n/j) * p_(j), enforced monotone, capped at 1.
    Added per Lane 2's gap report: per-depth flags/p-values were not
    persisted by the original sweep (see PREREG.md addendum)."""
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


def load_original_floor(seed):
    """MATCH/MISMATCH check source: the certified CSV row for this seed."""
    if not os.path.exists(ORIGINAL_CSV_PATH):
        return None
    with open(ORIGINAL_CSV_PATH, "r") as f:
        for row in csv.DictReader(f):
            if int(row["seed"]) == seed:
                return float(row["naive_floor_cm"])
    return None


def eval_model_per_class(model, X_cell_t, y_cell_encoded):
    """New (not in original): per-class one-vs-rest recall for one model
    on one already-built eval cell. Uses only torch inference, no new
    training/eval-data-generation logic."""
    with torch.no_grad():
        preds = model(X_cell_t).argmax(dim=1).numpy()
    per_class = {}
    for cls_idx, name in enumerate(CLASS_NAMES_SORTED):
        mask = y_cell_encoded == cls_idx
        per_class[name] = float(np.mean(preds[mask] == y_cell_encoded[mask])) if mask.sum() > 0 else float("nan")
    overall_acc = float(np.mean(preds == y_cell_encoded))
    return per_class, overall_acc


def run_one_seed(seed, n_eval_samples_per_cell=None, window_size=None):
    n_eval_samples_per_cell = n_eval_samples_per_cell or cfg.N_EVAL_SAMPLES_PER_CELL
    window_size = window_size or cfg.WINDOW
    if window_size != 100:
        raise NotImplementedError(
            "WINDOW != 100 requires passing window_size through to "
            "generate_degraded_windows -> extract_raw_windows; the original "
            "generate_degraded_windows (imported unchanged) does not expose "
            "this kwarg. Flagged in PREREG.md section 6; not resolved here."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    t_train0 = time.time()
    X_train, y_train_names = build_degradation_aware_dataset()
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train_names)
    model_real = train_cnn(X_train, y_train_encoded)
    torch.save(model_real.state_dict(), os.path.join(WEIGHTS_DIR, f"degradation_aware_seed{seed}.pt"))
    train_elapsed = time.time() - t_train0

    t_controls0 = time.time()
    seed_control_dir = os.path.join(WEIGHTS_DIR, f"controls_seed{seed}")
    os.makedirs(seed_control_dir, exist_ok=True)
    control_models = []
    for i in range(cfg.N_SHUFFLES):
        y_shuffled = np.random.permutation(y_train_encoded)
        model_c = train_cnn(X_train, y_shuffled)
        control_models.append(model_c)
        # Per user request: persist every shuffled-control model's weights
        # too (not just the real model), state_dicts, float32 (torch's
        # default dtype here -- SimpleCNN/train_cnn never cast to
        # anything else, confirmed by reading cnn_classifier.py).
        torch.save(model_c.state_dict(), os.path.join(seed_control_dir, f"control{i:03d}.pt"))
        touch_heartbeat(f"seed={seed} control_batch={i+1}/{cfg.N_SHUFFLES}")
    controls_elapsed = time.time() - t_controls0

    n_per_class = n_eval_samples_per_cell // len(CLASS_NAMES_SORTED)

    depths = list(EVAL_DEPTHS_CM)
    real_acc_by_depth = []
    real_recall_by_depth = []          # list of dict(class -> recall)
    control_recall_by_depth = []       # list of dict(class -> np.array[200])

    t_eval0 = time.time()
    for depth_cm in depths:
        depth_m = depth_cm / 100
        X_cell, y_cell_names = [], []
        for name in CLASS_NAMES_SORTED:
            kpa = stiffness_classes[name]
            windows = generate_degraded_windows(kpa, depth_m, FREQ, n_per_class)
            X_cell.extend(windows)
            y_cell_names.extend([name] * len(windows))
        X_cell = np.array(X_cell)
        y_cell_encoded = encoder.transform(y_cell_names)
        X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)

        real_recall, real_acc = eval_model_per_class(model_real, X_cell_t, y_cell_encoded)
        real_acc_by_depth.append(real_acc)
        real_recall_by_depth.append(real_recall)

        per_class_control_recalls = {name: [] for name in CLASS_NAMES_SORTED}
        for model_c in control_models:
            c_recall, _ = eval_model_per_class(model_c, X_cell_t, y_cell_encoded)
            for name in CLASS_NAMES_SORTED:
                per_class_control_recalls[name].append(c_recall[name])
        control_recall_by_depth.append({k: np.array(v, dtype=np.float32) for k, v in per_class_control_recalls.items()})
        touch_heartbeat(f"seed={seed} eval_depth={depth_cm}")
    eval_elapsed = time.time() - t_eval0

    # Aggregate floor (same logic/convention as original, recomputed here
    # for the seed-level MATCH/MISMATCH check against the certified CSV).
    # Aggregate control accuracy needs per-control OVERALL accuracy, not an
    # average of per-class recalls (not equivalent under class imbalance,
    # though n_per_class is equal here) -- recomputed honestly via a
    # second pass over the same control_models + freshly drawn eval cells
    # (the original script also redraws eval windows per depth per call).
    agg_p_values = []
    for i, depth_cm in enumerate(depths):
        depth_m = depth_cm / 100
        X_cell, y_cell_names = [], []
        for name in CLASS_NAMES_SORTED:
            kpa = stiffness_classes[name]
            windows = generate_degraded_windows(kpa, depth_m, FREQ, n_per_class)
            X_cell.extend(windows)
            y_cell_names.extend([name] * len(windows))
        # NOTE: this redraws fresh eval windows (matches the original
        # script's own behavior -- it also redraws per depth per call).
        X_cell = np.array(X_cell)
        y_cell_encoded = encoder.transform(y_cell_names)
        X_cell_t = torch.tensor(X_cell, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            preds_real = model_real(X_cell_t).argmax(dim=1).numpy()
        acc_real = np.mean(preds_real == y_cell_encoded)
        control_accs = []
        for model_c in control_models:
            with torch.no_grad():
                preds_c = model_c(X_cell_t).argmax(dim=1).numpy()
            control_accs.append(np.mean(preds_c == y_cell_encoded))
        control_accs = np.array(control_accs)
        p_value = (np.sum(control_accs >= acc_real) + 1) / (cfg.N_SHUFFLES + 1)
        agg_p_values.append(p_value)

    agg_sig = bh_fdr_correction(agg_p_values, alpha=cfg.FDR_ALPHA)
    agg_adj_p = bh_adjusted_pvalues(agg_p_values)
    agg_floor = None
    for depth_cm, sig in zip(depths, agg_sig):
        if not sig:
            agg_floor = depth_cm
            break
    if agg_floor is None:
        agg_floor = depths[-1]

    original_floor = load_original_floor(seed)
    floor_match = (original_floor is not None) and abs(agg_floor - original_floor) < 1e-6

    # ADDITIONAL REQUIREMENT (mid-run addendum, see PREREG.md): persist
    # the per-depth AGGREGATE raw p-value, BH-adjusted p-value, and
    # significance flag as their own per-seed checkpoint, not just the
    # single summary floor -- fixes the gap Lane 2 found (the original
    # sweep discarded every depth's flag except the first non-significant
    # one).
    depth_detail_path = os.path.join(LOG_DIR, f"lane5_aggregate_depth_detail_seed{seed}.csv")
    with open(depth_detail_path, "w", newline="") as f_detail:
        detail_writer = csv.DictWriter(
            f_detail, fieldnames=["seed", "depth_cm", "raw_p_value", "bh_adjusted_p_value", "fdr_significant"]
        )
        detail_writer.writeheader()
        for depth_cm, raw_p, adj_p, sig in zip(depths, agg_p_values, agg_adj_p, agg_sig):
            detail_writer.writerow({
                "seed": seed, "depth_cm": depth_cm, "raw_p_value": raw_p,
                "bh_adjusted_p_value": adj_p, "fdr_significant": bool(sig),
            })
    log.info(f"Seed {seed}: per-depth aggregate detail checkpointed to {depth_detail_path}")

    # Per-class p-values + per-class-per-seed BH-FDR family (PREREG 2).
    per_class_floor = {}
    per_class_first_below_045 = {}
    for name in CLASS_NAMES_SORTED:
        p_vals = []
        for i in range(len(depths)):
            real_r = real_recall_by_depth[i][name]
            ctrl_r = control_recall_by_depth[i][name]
            p = (np.sum(ctrl_r >= real_r) + 1) / (cfg.N_SHUFFLES + 1)
            p_vals.append(p)
        sig_flags = bh_fdr_correction(p_vals, alpha=cfg.FDR_ALPHA)
        floor_d = None
        for depth_cm, sig in zip(depths, sig_flags):
            if not sig:
                floor_d = depth_cm
                break
        per_class_floor[name] = floor_d if floor_d is not None else depths[-1]

        below = None
        for i, depth_cm in enumerate(depths):
            if real_recall_by_depth[i][name] < cfg.PER_CLASS_RECALL_FLOOR_THRESHOLD:
                below = depth_cm
                break
        per_class_first_below_045[name] = below

    # Save raw control-recall arrays at every depth (cheap; see PREREG 5).
    for i, depth_cm in enumerate(depths):
        for name in CLASS_NAMES_SORTED:
            np.save(
                os.path.join(CONTROLS_DIR, f"seed{seed}_depth{depth_cm:.2f}_{name}.npy"),
                control_recall_by_depth[i][name],
            )

    return {
        "seed": seed,
        "agg_floor_cm": agg_floor,
        "original_agg_floor_cm": original_floor,
        "floor_match": floor_match,
        "per_class_floor_cirrhosis": per_class_floor.get("cirrhosis"),
        "per_class_floor_healthy": per_class_floor.get("healthy"),
        "per_class_floor_significant_fibrosis": per_class_floor.get("significant_fibrosis"),
        "first_below_0.45_cirrhosis": per_class_first_below_045.get("cirrhosis"),
        "first_below_0.45_healthy": per_class_first_below_045.get("healthy"),
        "first_below_0.45_significant_fibrosis": per_class_first_below_045.get("significant_fibrosis"),
        "train_seconds": train_elapsed,
        "controls_seconds": controls_elapsed,
        "eval_seconds": eval_elapsed,
        "total_seconds": train_elapsed + controls_elapsed + eval_elapsed,
    }


def main():
    file_exists = os.path.exists(CSV_PATH)
    already_done = set()
    if file_exists:
        with open(CSV_PATH, "r") as f:
            for row in csv.DictReader(f):
                already_done.add(int(row["seed"]))
        log.info(f"Resuming: found existing results for seeds {sorted(already_done)}")

    fieldnames = [
        "seed", "agg_floor_cm", "original_agg_floor_cm", "floor_match",
        "per_class_floor_cirrhosis", "per_class_floor_healthy", "per_class_floor_significant_fibrosis",
        "first_below_0.45_cirrhosis", "first_below_0.45_healthy", "first_below_0.45_significant_fibrosis",
        "train_seconds", "controls_seconds", "eval_seconds", "total_seconds",
    ]
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        remaining = [s for s in cfg.SEEDS_TO_RUN if s not in already_done]
        n_total = len(cfg.SEEDS_TO_RUN)
        for idx, seed in enumerate(remaining):
            t0 = time.time()
            log.info(f"--- Seed {seed} starting ({len(already_done) + idx + 1}/{n_total}) ---")
            row = run_one_seed(seed)
            elapsed = time.time() - t0
            done_so_far = len(already_done) + idx + 1
            eta_seconds = elapsed * (n_total - done_so_far)
            log.info(
                f"Seed {seed} done in {elapsed:.0f}s. floor={row['agg_floor_cm']:.2f}cm "
                f"(original={row['original_agg_floor_cm']}, match={row['floor_match']}). "
                f"ETA for remaining {n_total - done_so_far} seeds: {eta_seconds/60:.1f} min"
            )
            writer.writerow(row)
            f.flush()

    log.info(f"All done. Full data in {CSV_PATH}")


if __name__ == "__main__":
    main()

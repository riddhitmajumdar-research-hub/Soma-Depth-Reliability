# Rerun ONLY condition 2b (amplitude randomization, corrected linear
# g~Uniform(0.5,1.5) distribution) for seeds 101-104. Condition 2's original
# results (log-uniform g, E[g]~1.08) are left untouched in results/ --
# 2b is appended alongside, never overwriting.
#
# Model weights from the original run were not persisted to disk, so this
# retrains the real model + 200 shuffled controls per seed from scratch
# (per PREREG.md protocol, same Source/ functions, same per-seed torch/np
# seeding as harness.run_one_seed) rather than reusing anything cached.

import os
import sys
import csv
import time
import logging
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(HERE, "..", "..", "..", "Source")
sys.path.insert(0, SOURCE_DIR)
sys.path.insert(0, HERE)

from degradation_aware_final_sweep import build_degradation_aware_dataset, train_cnn  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

import config as cfg  # noqa: E402
from harness import run_condition, RESULTS_DIR, LOG_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "battery_2b_run.log")), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("battery_2b")

SEED_TAG = "101-104"
FLOORS_CSV = os.path.join(RESULTS_DIR, f"floors_{SEED_TAG}.csv")
SEEDS = [101, 102, 103, 104]

COND_ID, COND_LABEL, COND_SHIFT = next(c for c in cfg.CONDITIONS if c[0] == "2b_amp_randomization_linear")

floors_fieldnames = ["seed", "condition", "condition_label", "floor_cm", "last_significant_cm", "censored"]
detail_fieldnames = ["seed", "condition", "depth_cm", "real_acc", "raw_p_value", "bh_adjusted_p_value",
                      "fdr_significant", "control_acc_min", "control_acc_median", "control_acc_max", "control_acc_std"]


def main():
    floors_file_exists = os.path.exists(FLOORS_CSV)
    with open(FLOORS_CSV, "a", newline="") as ff:
        floors_writer = csv.DictWriter(ff, fieldnames=floors_fieldnames)
        if not floors_file_exists:
            floors_writer.writeheader()

        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed)
            np.random.seed(seed)

            X_train, y_train_names = build_degradation_aware_dataset()
            encoder = LabelEncoder()
            y_train_encoded = encoder.fit_transform(y_train_names)
            model_real = train_cnn(X_train, y_train_encoded)
            train_elapsed = time.time() - t0
            log.info(f"seed={seed} real model retrained in {train_elapsed:.0f}s")

            t1 = time.time()
            control_models = []
            for i in range(cfg.N_SHUFFLES):
                y_shuffled = np.random.permutation(y_train_encoded)
                control_models.append(train_cnn(X_train, y_shuffled))
                if (i + 1) % 50 == 0:
                    log.info(f"seed={seed} controls retrained {i+1}/{cfg.N_SHUFFLES}")
            controls_elapsed = time.time() - t1
            log.info(f"seed={seed} 200 controls retrained in {controls_elapsed:.0f}s")

            depth_detail_path = os.path.join(RESULTS_DIR, f"depth_detail_{seed}.csv")
            detail_exists = os.path.exists(depth_detail_path)
            with open(depth_detail_path, "a", newline="") as df:
                detail_writer = csv.DictWriter(df, fieldnames=detail_fieldnames)
                if not detail_exists:
                    detail_writer.writeheader()

                row, confmats = run_condition(seed, COND_ID, COND_LABEL, COND_SHIFT, model_real,
                                               control_models, encoder, detail_writer, df)

            floors_writer.writerow(row)
            ff.flush()
            log.info(f"seed={seed} condition={COND_ID} floor={row['floor_cm']:.2f}cm "
                      f"censored={row['censored']} ({time.time()-t0:.0f}s)")

            confmat_path = os.path.join(RESULTS_DIR, f"confmat_{seed}.npz")
            existing = dict(np.load(confmat_path)) if os.path.exists(confmat_path) else {}
            for depth_cm, cm in confmats.items():
                existing[f"{COND_ID}__depth{depth_cm:.2f}"] = cm
            np.savez_compressed(confmat_path, **existing)

    log.info("CONDITION 2b RERUN COMPLETE")


if __name__ == "__main__":
    main()

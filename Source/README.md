# Source

Scripts, logs and saved models for the SOMA pipeline, Stages 1 to 4 and the checks added in Logs 18 to 25. Nothing here has been moved or renamed. This file is an index: one line per script, taken from the script's own header. Stage numbers follow the top-level `README.md`.

`constants.py` is the single place for the physical and sensor constants. Most scripts import from it.

Some scripts use absolute paths to this repo, and some load files by a path relative to `Source/` (for example `models/...`). Check the top of a script before running it.

## Core models and constants

| Script | What it does |
|---|---|
| `constants.py` | Physical and sensor constants, depth grids, seed, gate depth |
| `physics_model.py` | Stiffness classes, shear wave velocity, time of flight, `generate_signal` |
| `attenuation_model.py` | Kelvin-Voigt attenuation with class-dependent viscosity (Stage 2, Fix #3) |
| `attenuation_model_powerlaw.py` | Power-law attenuation, used for the sensitivity tests |

## Stage 3: classifiers

| Script | What it does |
|---|---|
| `source_classifier.py` | Feature-based baseline classifier on clean signals (3) |
| `cv_classifier.py` | K-fold cross-validation of the baseline (3a) |
| `viscosity_sensitivity.py` | Viscosity sensitivity sweep (3b) |
| `cnn_classifier.py` | Basic 1D CNN, and `extract_raw_windows` (3c) |
| `resnet_classifier.py` | ResNet1D (3e) |
| `stage3e_roc_pr.py` | ROC and precision-recall curves for 3e |
| `gradcam_resnet.py` | Grad-CAM for the classifier (3f) |
| `stage3f_saliency_featuremaps.py` | Saliency maps and feature maps (3f) |
| `joint_uncertainty.py` | Joint uncertainty propagation (3g) |
| `significance_testing.py` | Statistical significance tests (3h) |

## Stage 4: floors and degradation-aware training

| Script | What it does |
|---|---|
| `floor_characterization.py` | Floor sweep for the baseline, pre-registered protocol |
| `floor_characterization_cnn.py` | Floor sweep for the CNN, certified with shuffled controls |
| `extend_20hz.py` | 20 Hz depth extension |
| `stage4_depth_diagnostics.py` | Multi-depth confusion matrices and t-SNE |
| `degradation_aware_cnn.py` | Degradation-aware training, single seed, with the noise correction |
| `degradation_aware_cnn_calibration_retrain.py` | Same, with softmax probabilities logged for the reliability diagram |
| `degradation_aware_calibration_corrected.py` | Regenerates the calibration data with the noise and gate fixes |
| `degradation_aware_final_sweep.py` | The 24-seed sweep, 300 samples per cell |
| `degradation_aware_resnet_final_sweep.py` | The same sweep with ResNet1D |
| `retrain_seed121_verify.py` | Retrains and verifies seed 121 from the corrected sweep |
| `degradation_aware_per_class_recall_floor.py` | Per-class recall by depth, with 95% CIs |
| `noise_robustness_stress_test.py` | Sensor noise stress test on the certified CNN, certified with shuffles |
| `noise_robustness_stress_test_clean_cnn.py` | The same test with the clean-trained CNN |
| `power_law_sensitivity.py` | Certified CNN evaluated on power-law attenuation, no retraining |
| `power_law_sensitivity_certified.py` | Power-law sensitivity with the full 200-shuffle protocol |

## Checks from the external review (Log 25) and after

| Script | What it does |
|---|---|
| `repeated_grouped_cv_log11.py` | Repeated grouped 5-fold CV for the Log 11 comparisons |
| `resnet_per_class_metrics_grouped_split.py` | ResNet1D per-class metrics under the grouped split |
| `frozen_embedding_linear_probe.py` | Frozen-embedding linear probe at the 8.18 cm anomaly |
| `gradcam_counterfactual_occlusion.py` | Grad-CAM counterfactual occlusion test |
| `weak_damping_error_bound.py` | Leading-order error bound on the weak-damping approximation |
| `viscosity_sensitivity_eta0_extension.py` | Extends the viscosity sweep down to eta = 0 |
| `cirrhosis_viscosity_sensitivity.py` | Sweep over the cirrhosis viscosity magnitude |
| `regenerate_gate_statistic_verification.py` | Rebuilds `logs/gate_statistic_verification.csv` from the 24-seed floors |

## External validation and demo

| Script | What it does |
|---|---|
| `ets_montreal_path2_retrain.py` | Retrains the CNN on the ETS Montreal phantom RF data |
| `log21_figures.py` | Figures for the ETS Montreal validation (Log 21) |
| `run_worked_example_trained.py` | Runs simulated patient scans through the seed-121 model, with the depth gate |

## Figures

| Script | What it does |
|---|---|
| `plot_attenuation.py` | Stage 2 attenuation plots |
| `plot_classifier.py` | Stage 3 baseline result plots |
| `plot_cnn_classifier.py` | Stage 3c CNN plots |
| `plot_resnet_classifier.py` | Stage 3e ResNet1D plots |
| `plot_degradation_aware.py` | Three-way floor comparison |
| `plot_stage4.py` | Final Stage 4 plots |
| `make_figures.py` | Builds the figures in `outputs/figures_earlier/` from saved CSVs and pickles (methodology note 03) |
| `make_fig1_windowing.py` | Fig. 1, the beat-anchored window segmentation (note 03) |

## Subdirectories

| Directory | Contents |
|---|---|
| `logs/` | CSVs and text logs written by the scripts above |
| `results/` | Pickled Stage 4 results |
| `models/` | Four saved checkpoints (`.pt`, ignored by git) |
| `seed_weights_resnet/` | Per-seed ResNet1D weights (`.pt`, ignored by git) |
| `figures/` | Stage 3f and Stage 4 figure copies |
| `archive/` | `load_data.py`, an early Stage 1 script kept for history; see its README |

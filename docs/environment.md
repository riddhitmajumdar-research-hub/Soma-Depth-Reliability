# Environments

Three environments were used. This note records which, and what is known about the work each one ran. Everything here comes from the saved run logs and from the metadata embedded in the PNG files; where a link between an environment and a piece of work cannot be established from those, the table says "not recorded".

| Environment | Interpreter and packages | Work it ran |
|---|---|---|
| Local virtual environment | Python 3.10.0; torch 2.14.0, scipy 1.15.3, scikit-learn 1.7.2, matplotlib 3.10.9, numpy 2.2.6, pandas 2.3.3 (the pins in `requirements.txt`) | Most scripts. 35 PNG files record matplotlib 3.10.9. |
| Local Python 3.14 install | torch 2.13.0, scipy 1.18.1, scikit-learn 1.9.0, matplotlib 3.11.1, numpy 2.5.2 | Not recorded in any run log. 27 PNG files record matplotlib 3.11.1, the version installed here (listed below). |
| Remote compute runs | Python 3.11.15; torch 2.14.0+cu130 (CPU only in the runs), numpy 2.2.6, scipy 1.17.1, scikit-learn 1.9.1, matplotlib 3.11.2 | The 19-condition batteries on seeds 101 to 104 and 105 to 108 (`audit_fixes/cloud/battery/*/RUNLOG.md`), the smoke run and the random forest runs, which reuse the smoke-run setup (`audit_fixes/cloud/rf/REPORT_101-112.md`). No PNG file records matplotlib 3.11.2. |

The remote runs used scipy 1.17.1, scikit-learn 1.9.1 and matplotlib 3.11.2, and the local virtual environment used scipy 1.15.3, scikit-learn 1.7.2 and matplotlib 3.10.9. The torch release (2.14.0) and numpy (2.2.6) are the same in both.

## Figures whose PNG metadata records matplotlib 3.11.1

- `outputs/figures_earlier/fig3_02_amplitude_vs_depth_by_class.png`
- `outputs/figures_earlier/fig3_03_confusion_matrix_baseline.png`
- `outputs/figures_earlier/fig3_04_cnn_training_curves.png`
- `outputs/figures_earlier/fig3_07_floor_progression.png`
- `outputs/stage1/stage1_physics_signals.png`
- `outputs/stage2/stage2_amplitude_vs_depth.png`
- `outputs/stage2/stage2_before_after.png`
- `outputs/stage3/stage3_confusion_matrix.png`
- `outputs/stage3/stage3_feature_distribution.png`
- `outputs/stage3/stage3b_viscosity_sensitivity.png`
- `outputs/stage3/stage3c_confusion_matrix.png`
- `outputs/stage3/stage3c_training_curves.png`
- `outputs/stage3/stage3e_confusion_matrix.png`
- `outputs/stage3/stage3e_per_class_metrics.png`
- `outputs/stage3/stage3e_roc_pr_curves.png`
- `outputs/stage3/stage3e_training_curves.png`
- `outputs/stage3/stage3f_feature_maps.png`
- `outputs/stage3/stage3f_gradcam.png`
- `outputs/stage3/stage3f_gradcam_resolution.png`
- `outputs/stage3/stage3f_saliency_maps.png`
- `outputs/stage3/stage3g_joint_uncertainty.png`
- `outputs/stage3/stage3h_paired_differences.png`
- `outputs/stage3/stage3h_paired_trajectory.png`
- `outputs/stage3/stage3h_significance_comparison.png`
- `outputs/stage4/figures/stage4_baseline_vs_cnn.png`
- `outputs/stage4/figures/stage4_cnn_accuracy_vs_depth.png`
- `outputs/stage4/figures/stage4_full_progression.png`

## Figures whose PNG metadata records matplotlib 3.11.2

None.

## Other figure metadata

16 PNG files record matplotlib 3.10.7, which is the version of a further local Python 3.10.0 install outside the virtual environment, and 8 PNG files carry no software entry. The environment of both groups is not recorded.

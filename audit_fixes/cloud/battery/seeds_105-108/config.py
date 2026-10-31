# Battery-B config (seeds 105-108): replicates Battery-A's (run battery-
# 101-104) 19-condition protocol exactly, EXCEPT condition 2, which uses the
# CORRECTED linear-space gain draw g~Uniform(0.5,1.5) (E[g]=1.0 exactly)
# instead of 101-104's original log-uniform draw g=exp(U(ln .5, ln 2))
# (E[g]~1.08, a design error found after that run -- see PREREG.md). Seeds
# 105-108 only ever see the corrected condition 2; there is no "2 vs 2b"
# split on this branch (101-104's own 2b addendum already covers that
# comparison for seeds 101-104). Each condition is (condition_id, label,
# shift_dict). shift_dict keys are consumed by
# wrappers.generate_degraded_windows_shifted.

SEEDS_TO_RUN = [105, 106, 107, 108]

N_SHUFFLES = 200
N_TORCH_THREADS = 4
FDR_ALPHA = 0.05

CONDITIONS = [
    ("1_baseline", "baseline", {}),
    ("2_amp_randomization", "amplitude randomisation g~U(0.5,1.5) linear (corrected; E[g]=1.0)", {"amp_gain_linear": True}),
    ("3a_spreading_p0.5", "geometric spreading p=0.5", {"spreading_p": 0.5}),
    ("3b_spreading_p1.0", "geometric spreading p=1.0", {"spreading_p": 1.0}),
    ("4a_stiffness_cv_perclass", "within-class stiffness CV per class (0.115/0.056/0.245)",
        {"stiffness_cv_per_class": {"healthy": 0.115, "significant_fibrosis": 0.056, "cirrhosis": 0.245}}),
    ("4b_stiffness_cv_uniform", "within-class stiffness CV=0.15 all classes", {"stiffness_cv": 0.15}),
    ("5a_snr20", "SNR sweep 20dB", {"snr_db": 20.0}),
    ("5b_snr13.58", "SNR sweep 13.58dB (SENSOR_SNR_DB_WORST_CASE)", {"snr_db": 13.58}),
    ("5c_snr10", "SNR sweep 10dB", {"snr_db": 10.0}),
    ("5d_snr5", "SNR sweep 5dB", {"snr_db": 5.0}),
    ("5e_snr0", "SNR sweep 0dB", {"snr_db": 0.0}),
    ("6a_exact_atten", "exact complex-wavenumber attenuation, TOF unchanged", {"exact_atten": True}),
    ("6b_exact_atten_tof", "exact attenuation + exact TOF via c_exact", {"exact_tof": True}),
    ("7a_viscosity_healthy_fibrosis_1.4x", "eta healthy/fibrosis x1.4 (cirrhosis fixed 1.77)",
        {"eta_override_per_class": {"healthy": 0.5 * 1.4, "significant_fibrosis": 0.5 * 1.4}}),
    ("7b_viscosity_healthy_fibrosis_1.8x", "eta healthy/fibrosis x1.8 (cirrhosis fixed 1.77)",
        {"eta_override_per_class": {"healthy": 0.5 * 1.8, "significant_fibrosis": 0.5 * 1.8}}),
    ("7c_viscosity_healthy_fibrosis_2.6x", "eta healthy/fibrosis x2.6 (cirrhosis fixed 1.77)",
        {"eta_override_per_class": {"healthy": 0.5 * 2.6, "significant_fibrosis": 0.5 * 2.6}}),
    ("7d_viscosity_cirrhosis_1.2", "eta cirrhosis=1.2 (healthy/fibrosis fixed 0.5)",
        {"eta_override_per_class": {"cirrhosis": 1.2}}),
    ("7e_viscosity_cirrhosis_2.18", "eta cirrhosis=2.18 (healthy/fibrosis fixed 0.5)",
        {"eta_override_per_class": {"cirrhosis": 2.18}}),
    ("7f_viscosity_cirrhosis_2.5", "eta cirrhosis=2.5 (healthy/fibrosis fixed 0.5)",
        {"eta_override_per_class": {"cirrhosis": 2.5}}),
]

# Condition 8 (pure-noise probe) is handled separately -- no floor/p-values,
# runs last, see noise_probe.py.

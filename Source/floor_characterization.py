# SOMA Signal Fidelity Investigation
# Stage 4 -- Floor Characterization Sweep (Monte Carlo, pre-registered protocol)
# File: Source/floor_characterization.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from physics_model import generate_signal, stiffness_classes
from attenuation_model import attenuate_signal
from source_classifier import extract_pulses_with_features, build_dataset
from constants import SENSOR_SNR_DB, SEED

np.random.seed(SEED)

DEPTHS_CM = np.linspace(0, 15, 30)
FREQUENCIES_HZ = [20, 100, 250, 500]
N_SAMPLES_PER_CELL = 100
N_SHUFFLES = 200

# --- FIX #1: sensor noise floor ---------------------------------------------
# PREVIOUS: 67.03dB, measured at 1kHz for a free-field microphone -- outside
# SOMA's 20-500Hz band and the wrong sensor category (not skin-contact).
# REPLACED: Yang et al. 2022, Biosensors (PMC9312695) -- wearable
# skin-contact MEMS heart-sound sensor, 10-630Hz band (covers SOMA's band
# almost exactly), SNR=27.05dB, validated against a 3M Littmann stethoscope
# (21.80dB) on a real subject. Corroborated by two other independent
# wearable MEMS heart-sound papers in the same band (26.47dB, 13.58dB).
# CONSEQUENCE: ~40dB gap = ~2 orders of magnitude in noise floor. Expect the
# certified depth floor to shrink substantially -- re-run, don't just re-cite.
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)


def generate_degraded_samples(kpa, depth_m, frequency_hz, n_needed, jitter_std=0.03):
    features_list = []
    while len(features_list) < n_needed:
        jittered_hr = 1.15 + np.random.normal(0, jitter_std)
        time, clean_signal, tof = generate_signal(kpa, heart_rate_hz=jittered_hr)
        attenuated, alpha = attenuate_signal(clean_signal, kpa, depth_m, frequency_hz)
        noisy = attenuated + np.random.normal(0, SENSOR_NOISE_STD, size=attenuated.shape)
        rows = extract_pulses_with_features(noisy, time, jittered_hr, tof)
        features_list.extend(rows)
    return features_list[:n_needed]


def bh_fdr_correction(p_values, alpha=0.05):
    p_values = np.array(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not passed.any():
        return np.zeros(n, dtype=bool)
    max_passed_rank = np.max(np.where(passed)[0])
    significant = np.zeros(n, dtype=bool)
    significant[order[:max_passed_rank + 1]] = True
    return significant


if __name__ == "__main__":
    print("Training the real classifier (clean signals, exactly as Stage 3)...")
    X_clean, y_clean = build_dataset()
    scaler_real = StandardScaler()
    X_clean_scaled = scaler_real.fit_transform(X_clean)
    model_real = LogisticRegression(max_iter=1000)
    model_real.fit(X_clean_scaled, y_clean)

    print(f"Training {N_SHUFFLES} independently-shuffled control classifiers...")
    control_models = []
    for i in range(N_SHUFFLES):
        y_shuffled = np.random.permutation(y_clean)
        scaler_c = StandardScaler()
        X_c_scaled = scaler_c.fit_transform(X_clean)
        model_c = LogisticRegression(max_iter=1000)
        model_c.fit(X_c_scaled, y_shuffled)
        control_models.append((model_c, scaler_c))

    class_names = sorted(stiffness_classes.keys())
    n_per_class = N_SAMPLES_PER_CELL // len(class_names)

    results = []

    print(f"\nRunning sweep: {len(DEPTHS_CM)} depths x {len(FREQUENCIES_HZ)} frequencies "
          f"x {N_SAMPLES_PER_CELL} samples/cell (+ {N_SHUFFLES} permutation controls each)...")
    for freq in FREQUENCIES_HZ:
        for depth_cm in DEPTHS_CM:
            depth_m = depth_cm / 100
            X_cell = []
            y_cell = []
            for name in class_names:
                kpa = stiffness_classes[name]
                rows = generate_degraded_samples(kpa, depth_m, freq, n_per_class)
                X_cell.extend(rows)
                y_cell.extend([name] * len(rows))
            X_cell = np.array(X_cell)
            y_cell = np.array(y_cell)
            n_total = len(y_cell)

            preds_real = model_real.predict(scaler_real.transform(X_cell))
            acc_real = np.mean(preds_real == y_cell)

            control_accs = []
            for model_c, scaler_c in control_models:
                preds_c = model_c.predict(scaler_c.transform(X_cell))
                control_accs.append(np.mean(preds_c == y_cell))
            control_accs = np.array(control_accs)

            p_value = (np.sum(control_accs >= acc_real) + 1) / (N_SHUFFLES + 1)

            results.append({
                "freq": freq, "depth_cm": depth_cm,
                "acc_real": acc_real,
                "control_mean": control_accs.mean(),
                "control_min": control_accs.min(),
                "control_max": control_accs.max(),
                "p_value": p_value,
            })
        print(f"  Completed frequency {freq}Hz")

    all_p_values = [r["p_value"] for r in results]
    significant_flags = bh_fdr_correction(all_p_values)
    for r, sig in zip(results, significant_flags):
        r["fdr_significant"] = sig

    print("\n" + "="*70)
    print("FLOOR IDENTIFICATION (first depth where accuracy is no longer")
    print("FDR-significant against the EMPIRICAL permutation null)")
    print("="*70)
    for freq in FREQUENCIES_HZ:
        freq_results = [r for r in results if r["freq"] == freq]
        floor_depth = None
        for r in freq_results:
            if not r["fdr_significant"]:
                floor_depth = r["depth_cm"]
                break
        if floor_depth is not None:
            print(f"  {freq:4d}Hz  ->  floor at {floor_depth:.2f}cm")
        else:
            print(f"  {freq:4d}Hz  ->  never reaches the null within 0-15cm")

    print("\nEmpirical control (chance) accuracy by depth range:")
    shallow = [r["control_mean"] for r in results if r["depth_cm"] < 2]
    deep = [r["control_mean"] for r in results if r["depth_cm"] > 13]
    print(f"  Shallow (<2cm) mean control: {np.mean(shallow):.2%}")
    print(f"  Deep (>13cm) mean control:   {np.mean(deep):.2%}")

    import pickle
    with open("results/stage4_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("\nFull results saved to results/stage4_results.pkl for plotting.")

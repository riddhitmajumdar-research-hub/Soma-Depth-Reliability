import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

project_files = glob.glob("**/physics_model.py", recursive=True) + glob.glob("../**/physics_model.py", recursive=True)
if project_files:
    sys.path.append(os.path.dirname(os.path.abspath(project_files[0])))

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED, NOMINAL_FREQ_HZ, TRAINING_DEPTHS_CM, MAX_VALID_DEPTH_CM

torch.manual_seed(SEED)
np.random.seed(SEED)

# FIX (post-audit): was 67.03dB, the pre-Fix-#1 value. Log 18 corrected this
# everywhere else but this worked-example demo was never updated.
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)
TRAIN_DEPTHS_CM = TRAINING_DEPTHS_CM
MODEL_WEIGHTS_PATH = "models/soma_certified_model_seed121_corrected_noise.pt"

class_labels = sorted(stiffness_classes.keys())

print(f"Loading verified seed-121 model weights (one seed from the certified 8.48cm +/- 1.81cm distribution) from {MODEL_WEIGHTS_PATH}...")
model = SimpleCNN(n_classes=3)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
model.eval()
print("Certified model loaded (no retraining performed).\n")


def evaluate_patient_scan(patient_id, true_kpa, true_class, depth_cm):
    print("=" * 70)
    print(f"PATIENT TELEMETRY REPORT | ID: {patient_id}")
    print(f"Target Depth: {depth_cm:.1f} cm | True Condition: {true_class} ({true_kpa:.1f} kPa)")
    print("=" * 70)

    if depth_cm > MAX_VALID_DEPTH_CM:
        print("[STATUS: REJECTED / UNRELIABLE]")
        print(f"  CRITICAL WARNING: Target depth ({depth_cm:.1f} cm) exceeds certified domain-of-validity floor ({MAX_VALID_DEPTH_CM:.1f} cm).")
        print("  Model output is uncalibrated at this depth; prediction aborted per safety boundary.")
        print("  (Rejection is based on measured depth, not model confidence -- confidence calibration")
        print("   at this depth is known to be unreliable per stage4_calibration_data.csv analysis.)\n")
        return

    depth_m = depth_cm / 100.0
    jittered_hr = 1.18
    time, clean_signal, tof = generate_signal(true_kpa, heart_rate_hz=jittered_hr)
    attenuated, alpha = attenuate_signal(clean_signal, true_kpa, depth_m, NOMINAL_FREQ_HZ)
    noisy_signal = attenuated + np.random.normal(0, SENSOR_NOISE_STD, size=attenuated.shape)

    windows = extract_raw_windows(noisy_signal, time, jittered_hr)
    x_sample = torch.tensor(windows[0], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x_sample)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    pred_idx = np.argmax(probs)
    pred_label = class_labels[pred_idx]
    confidence = probs[pred_idx]

    print("[STATUS: VALIDATED IN-DOMAIN]")
    print(f"  Signal-to-Noise Ratio : {SENSOR_SNR_DB:.2f} dB")
    print(f"  Attenuation Alpha     : {alpha:.2f} /m")
    print(f"  Predicted Condition   : {pred_label.upper()}")
    print(f"  Softmax Confidence    : {confidence * 100:.2f}%")
    print("  Class Probabilities   :")
    for label, prob in zip(class_labels, probs):
        print(f"    - {label:<20s}: {prob * 100:5.2f}%")
    print("\n")


if __name__ == "__main__":
    evaluate_patient_scan("PATIENT_A_IN_DOMAIN", stiffness_classes["significant_fibrosis"], "significant_fibrosis", 6.0)
    evaluate_patient_scan("PATIENT_B_OUT_OF_DOMAIN", stiffness_classes["cirrhosis"], "cirrhosis", 11.0)

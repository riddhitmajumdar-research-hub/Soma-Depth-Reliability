import csv
import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Locate project modules
project_files = glob.glob("**/physics_model.py", recursive=True) + glob.glob("../**/physics_model.py", recursive=True)
if project_files:
    sys.path.append(os.path.dirname(os.path.abspath(project_files[0])))

from physics_model import stiffness_classes, generate_signal
from attenuation_model import attenuate_signal
from cnn_classifier import SimpleCNN, extract_raw_windows
from constants import SENSOR_SNR_DB, SEED, NOMINAL_FREQ_HZ, MAX_VALID_DEPTH_CM

# =========================================================
# STEP 7: GENERATE RELIABILITY DIAGRAM
# FIX (post-audit): previously reimplemented UCE/binning independently with
# a different bin range (0.3-1.0) than the canonical calibration_analysis.py
# (0-1.0), risking silently different UCE numbers from the same CSV. Now
# delegates to the single-sourced script instead.
# =========================================================
import subprocess
import glob as _glob

csv_matches = _glob.glob("**/stage4_calibration_data.csv", recursive=True) + _glob.glob("../**/stage4_calibration_data.csv", recursive=True)
if csv_matches:
    csv_path = csv_matches[0]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(csv_path))) if "outputs" in csv_path else os.path.dirname(os.path.abspath(csv_path))
    calibration_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_analysis.py")
    print("[STEP 7] Delegating to calibration_analysis.py (single source of truth)...")
    subprocess.run(["python3", calibration_script, csv_path], check=True)


# =========================================================
# STEP 11: TRAINED WORKED EXAMPLE (PATIENT A & B)
# FIX (post-audit): previously retrained a fresh model inline here, directly
# contradicting Log 16's stated claim that the worked example "loads the
# certified weights rather than retraining." Now actually does that.
# =========================================================
torch.manual_seed(SEED)
np.random.seed(SEED)

# FIX (post-audit): was 67.03dB, the pre-Fix-#1 value.
SENSOR_NOISE_STD = 10 ** (-SENSOR_SNR_DB / 20)

MODEL_WEIGHTS_PATH = _glob.glob("**/soma_certified_model_seed121_corrected_noise.pt", recursive=True) + _glob.glob("../**/soma_certified_model_seed121_corrected_noise.pt", recursive=True)
assert MODEL_WEIGHTS_PATH, "Certified model weights not found -- cannot run worked example without them"
MODEL_WEIGHTS_PATH = MODEL_WEIGHTS_PATH[0]

class_labels = sorted(stiffness_classes.keys())

print(f"[STEP 11] Loading certified model weights from {MODEL_WEIGHTS_PATH} (no retraining)...")
model = SimpleCNN(n_classes=3)
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
model.eval()
print("Certified model loaded.\n")

def evaluate_patient_scan(patient_id, true_kpa, true_class, depth_cm):
    print("=" * 70)
    print(f"PATIENT TELEMETRY REPORT | ID: {patient_id}")
    print(f"Target Depth: {depth_cm:.1f} cm | True Condition: {true_class} ({true_kpa:.1f} kPa)")
    print("=" * 70)
    
    if depth_cm > MAX_VALID_DEPTH_CM:
        print("[STATUS: REJECTED / UNRELIABLE]")
        print(f"  CRITICAL WARNING: Target depth ({depth_cm:.1f} cm) exceeds certified domain-of-validity floor ({MAX_VALID_DEPTH_CM:.1f} cm).")
        print("  Model output is uncalibrated at this depth; prediction aborted per safety boundary.\n")
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

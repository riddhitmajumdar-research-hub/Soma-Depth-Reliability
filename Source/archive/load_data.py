# SOMA Local Processing Engine
# Stage 1 — Data Representation
# File: Source/load_data.py

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
SAMPLE_RATE    = 500
DURATION       = 3
HEART_RATE_HZ  = 1.2
NOISE_STRENGTH = 0.3
WINDOW_SIZE    = 100
STEP_SIZE      = 50

# --- Time axis ---
time = np.linspace(0, DURATION, SAMPLE_RATE * DURATION)

# --- Clean heartbeat wave ---
heartbeat = np.sin(2 * np.pi * HEART_RATE_HZ * time)

# --- Gaussian noise ---
noise = np.random.normal(loc=0, scale=NOISE_STRENGTH, size=time.shape)

# --- Noisy signal ---
noisy_heartbeat_signal = heartbeat + noise

# --- Windowing ---
windows = []

for start in range(0, len(noisy_heartbeat_signal) - WINDOW_SIZE + 1, STEP_SIZE):
    window = noisy_heartbeat_signal[start : start + WINDOW_SIZE]
    windows.append(window)

windows = np.array(windows)

# --- Inspect arrays ---
print(f"Window array shape         : {windows.shape}")
print(f"Total windows created      : {windows.shape[0]}")
print(f"Samples per window         : {windows.shape[1]}")
print(f"Seconds per window         : {windows.shape[1] / SAMPLE_RATE}")
print(f"First window, first 5 vals : {windows[0, :5].round(4)}")
print(f"Second window, first 5 vals: {windows[1, :5].round(4)}")

# --- Four-panel plot ---
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

axes[0, 0].plot(time, heartbeat, color='crimson', linewidth=1.5)
axes[0, 0].set_title("Full Clean Signal")
axes[0, 0].set_xlabel("Time (seconds)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(time, noisy_heartbeat_signal, color='steelblue', linewidth=0.8)
axes[0, 1].set_title("Full Noisy Signal")
axes[0, 1].set_xlabel("Time (seconds)")
axes[0, 1].set_ylabel("Amplitude")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(windows[0], color='darkorange', linewidth=1.5)
axes[1, 0].set_title("Window 1 — Samples 0 to 99")
axes[1, 0].set_xlabel("Sample index")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(windows[1], color='darkorange', linewidth=1.5)
axes[1, 1].set_title("Window 2 — Samples 50 to 149")
axes[1, 1].set_xlabel("Sample index")
axes[1, 1].set_ylabel("Amplitude")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("SOMA — Signal Windowing (Stage 1, Step 3)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('Data/stage1_pipeline_output.png', dpi=300, bbox_inches='tight')
plt.show()
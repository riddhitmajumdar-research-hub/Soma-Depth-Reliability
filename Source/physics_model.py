import numpy as np                 # arrays and math functions
import matplotlib.pyplot as plt    # plotting

def shear_wave_velocity_m_s(youngs_modulus_kpa, rho=1060):
    """
    Converts stiffness (Young's modulus, kPa) to shear wave velocity (m/s).
    rho: tissue density, kg/m^3 (standard soft-tissue estimate: 1060)
    """
    E_pa = youngs_modulus_kpa * 1000   # kPa -> Pa
    G_pa = E_pa / 3                     # E = 3G -> solve for G
    return (G_pa / rho) ** 0.5          # G = rho*c^2 -> solve for c


def time_of_flight_s(youngs_modulus_kpa, distance_m=0.010):
    """
    Time for a shear wave to cross a fixed distance, for a given stiffness.
    distance_m: representative tracking distance (10mm), based on real elastography devices.
    """
    c = shear_wave_velocity_m_s(youngs_modulus_kpa)
    return distance_m / c


def generate_signal(youngs_modulus_kpa, duration=3.0, heart_rate_hz=1.15, sample_rate=500):
    """
    Builds a synthetic signal for one stiffness value.
    Each heartbeat launches a shear wave pulse; the pulse arrives at the
    sensor after a stiffness-dependent delay (time-of-flight).
    """
    tof = time_of_flight_s(youngs_modulus_kpa)          # delay per pulse, for this stiffness
    beat_times = np.arange(0, duration, 1.0 / heart_rate_hz)  # one heartbeat per cycle

    n_samples = int(duration * sample_rate)
    time = np.linspace(0, duration, n_samples)
    signal = np.zeros(n_samples)                        # empty signal to fill in

    for beat_t in beat_times:
        arrival_t = beat_t + tof     # pulse arrives tof seconds after the heartbeat
        pulse_window = 0.15          # pulse duration, seconds

        mask = (time >= arrival_t) & (time <= arrival_t + pulse_window)
        local_t = time[mask] - arrival_t
        pulse_shape = np.exp(-15 * local_t) * np.sin(2 * np.pi * 100 * local_t)
        signal[mask] += pulse_shape

    return time, signal, tof


# --- FIX #2: stiffness-class citation correction ---------------------------
# PREVIOUS comment ("sourced in README, Log 02") pointed to Yoo et al. (earlier recorded as 'Lee et al.') 2021
# (PMC8535143, GE LOGIQ S8/E9 2D-SWE), which reports diagnostic CUTOFFS
# (6.42-6.70 kPa significant fibrosis, 8.88-9.15 kPa cirrhosis) -- not
# representative per-class values -- and defines no healthy baseline at all.
#
# Replaced with a single internally-consistent source reporting
# representative kPa per fibrosis stage directly (biopsy-validated):
#   Zayadeen et al., Egyptian Liver Journal (King Hussein Medical Center),
#   n=95: F0-F1=5.39+/-0.62 kPa, F2=7.32+/-0.41 kPa, F3=8.46+/-0.33 kPa,
#   F4=11.42+/-2.8 kPa.
#
# DECISION MADE HERE: kept original numeric values (within ~15-20% of
# Zayadeen et al.) rather than renumbering and invalidating every
# downstream Stage 1-4 result. Tightened alternative if you want it instead:
# {"healthy": 5.4, "significant_fibrosis": 7.3, "cirrhosis": 11.4}
stiffness_classes = {
    "healthy": 4.5,
    "significant_fibrosis": 7.5,
    "cirrhosis": 12.0,
}

if __name__ == "__main__":
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for ax, (name, kpa) in zip(axes, stiffness_classes.items()):
        time, signal, tof = generate_signal(kpa)
        ax.plot(time, signal)
        ax.set_title(f"{name} (E = {kpa} kPa, time-of-flight = {tof*1000:.2f} ms)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    plt.suptitle("Stage 1 — Clean signals by stiffness class")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig("stage1_physics_signals.png", dpi=200, bbox_inches="tight")
    plt.show()

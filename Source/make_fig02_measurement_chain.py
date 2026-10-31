# Figure 2 (measurement chain), drawn with matplotlib. All numbers are computed from the repo's own code; nothing is typed by hand.
#   E per class        : physics_model.stiffness_classes
#   rho, d             : default arguments of physics_model.shear_wave_velocity_m_s / time_of_flight_s (read with inspect)
#   G, c, TOF          : physics_model.shear_wave_velocity_m_s, physics_model.time_of_flight_s
#   carrier frequency  : parsed from the source of physics_model.generate_signal (np.sin(2*np.pi*100*local_t))
#   sample rate, beats : generate_signal defaults; window length: cnn_classifier.extract_raw_windows default
#   SNR, sigma         : constants.SENSOR_SNR_DB
import sys, os, inspect
sys.dont_write_bytecode = True
from pathlib import Path as _P; REPO = str(_P(__file__).resolve().parents[1])
sys.path.insert(0, REPO + "/Source")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import physics_model as pm, cnn_classifier as cc, constants as C

OUT = sys.argv[1] if len(sys.argv) > 1 else REPO + "/outputs/figures_final/fig02_measurement_chain.png"
dflt = lambda f, k: inspect.signature(f).parameters[k].default
RHO = dflt(pm.shear_wave_velocity_m_s, "rho")            # kg/m^3
D_M = dflt(pm.time_of_flight_s, "distance_m")            # m
FS = dflt(pm.generate_signal, "sample_rate")
WIN = dflt(cc.extract_raw_windows, "window_size")
names = list(pm.stiffness_classes)
E = [pm.stiffness_classes[n] for n in names]
G = [e / 3 for e in E]
c = [pm.shear_wave_velocity_m_s(e) for e in E]
tof = [pm.time_of_flight_s(e) * 1000 for e in E]
sigma = 10 ** (-C.SENSOR_SNR_DB / 20)
import re
CARRIER = int(re.search(r'2 \* np\.pi \* (\d+) \* local_t', inspect.getsource(pm.generate_signal)).group(1))
slash = lambda v, f: " / ".join(f.format(x) for x in v)
print("E", E, "G", G, "c", [round(x, 2) for x in c], "TOF ms", [round(x, 2) for x in tof], "rho", RHO, "d", D_M, "win", WIN, "fs", FS, "SNR", C.SENSOR_SNR_DB, "sigma", round(sigma, 4))

boxes = [
 f"Stiffness E\n{slash(E, '{:g}')} kPa",
 f"Shear modulus G = E/3\n{slash(G, '{:.2f}')} kPa\nSpeed c = sqrt(G/rho)\nrho = {RHO} kg/m³\n{slash(c, '{:.2f}')} m/s",
 f"Time of flight\nTOF = d/c, d = {D_M*1000:g} mm\n{slash(tof, '{:.2f}')} ms",
 f"Pulse, {CARRIER} Hz carrier\nstarts at beat + TOF",
 "Attenuation\nA = A0 exp(-α·depth)\nα = ω²·η / (2·G·c)",
 f"Sensor noise\nσ = 10^(-SNR/20)\nSNR {C.SENSOR_SNR_DB:g} dB, σ = {sigma:.4f}",
 f"{WIN}-sample window\nat heartbeat trigger\n{WIN/FS:g} s at {FS} Hz",
 "Classifier\nCNN, ResNet1D,\nrandom forest",
]
fig, ax = plt.subplots(figsize=(15.2, 7.2)); ax.set_xlim(0, 152); ax.set_ylim(0, 72); ax.axis("off")
plt.rcParams["font.family"] = "serif"
w, h, gap = 33, 26, 4
xs = [2 + i * (w + gap) for i in range(4)]
for i, t in enumerate(boxes):
    x = xs[i % 4]; y = 42 if i < 4 else 4
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.4", fc="#EEF3F8", ec="#1F3A5F", lw=2.2))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=10.5, family="serif")
    if i % 4 != 3:
        ax.annotate("", xy=(x + w + gap - 0.4, y + h / 2), xytext=(x + w + 0.4, y + h / 2), arrowprops=dict(arrowstyle="->", color="#1F3A5F", lw=2))
x3 = xs[3] + w / 2
ax.plot([x3, x3, xs[0] + w / 2], [42, 36, 36], color="#1F3A5F", lw=2)
ax.annotate("", xy=(xs[0] + w / 2, 30.4), xytext=(xs[0] + w / 2, 36), arrowprops=dict(arrowstyle="->", color="#1F3A5F", lw=2))
fig.text(0.5, 0.01, "Values in the boxes are classes in the order: " + " / ".join(n.replace("_", " ") for n in names) + ".", ha="center", fontsize=9, style="italic")
import csv
with open(OUT.replace(".png", "_values.csv"), "w", newline="") as _f:
    _w = csv.writer(_f); _w.writerow(["class", "E_kPa", "G_kPa", "c_m_per_s", "TOF_ms"])
    for _i, _n in enumerate(names): _w.writerow([_n, E[_i], G[_i], c[_i], tof[_i]])
    _w.writerow([]); _w.writerow(["rho_kg_m3", "tracking_distance_m", "carrier_Hz", "sample_rate_Hz", "window_samples", "SNR_dB", "sigma"])
    _w.writerow([RHO, D_M, CARRIER, FS, WIN, C.SENSOR_SNR_DB, sigma])
fig.savefig(OUT, dpi=150, bbox_inches="tight"); print("saved", OUT)

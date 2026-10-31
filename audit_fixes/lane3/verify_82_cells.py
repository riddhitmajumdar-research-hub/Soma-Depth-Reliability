"""
Lane 3 audit — trace the "82 frequency x depth cells" FDR family claim.
Reproduces the exact grid construction from Source/floor_characterization_cnn.py
(lines 20-35, values copied verbatim, imports BASE_EVAL_DEPTHS_CM from
Source/constants.py) to independently count the (freq, depth) cells that
`bh_fdr_correction` is applied across jointly in that script.
Run from repo root: python3 audit_fixes/lane3/verify_82_cells.py
"""
import numpy as np

# from Source/constants.py
BASE_EVAL_DEPTHS_CM = np.linspace(0, 15, 12)

# from Source/floor_characterization_cnn.py lines 20-35 (copied verbatim)
BASE_DEPTHS_CM = BASE_EVAL_DEPTHS_CM
FINE_NEAR_ORIGIN_CM = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2, 1.5, 1.8, 2.1]
DENSE_DEPTHS_CM = np.sort(np.unique(np.concatenate([BASE_DEPTHS_CM, FINE_NEAR_ORIGIN_CM])))

FINE_100HZ_CM = [4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5, 5.7, 5.9, 6.1, 6.3, 6.5]
DENSE_100HZ_CM = np.sort(np.unique(np.concatenate([BASE_DEPTHS_CM, FINE_100HZ_CM])))

FREQUENCIES_HZ = [20, 100, 250, 500]
DEPTHS_BY_FREQ = {
    20: BASE_DEPTHS_CM,
    100: DENSE_100HZ_CM,
    250: DENSE_DEPTHS_CM,
    500: DENSE_DEPTHS_CM,
}

for f in FREQUENCIES_HZ:
    print(f"{f}Hz: {len(DEPTHS_BY_FREQ[f])} depths")

total_cells = sum(len(DEPTHS_BY_FREQ[f]) for f in FREQUENCIES_HZ)
print("TOTAL (freq, depth) CELLS in the single joint bh_fdr_correction(all_p_values) family:", total_cells)

import numpy as np, glob, os

CONTROLS_DIR = "audit_fixes/lane5/clean_trained_sweep/clean_sweep/controls" if os.path.isdir("audit_fixes/lane5/clean_trained_sweep/clean_sweep/controls") else None
if CONTROLS_DIR is None:
    # try to locate it
    hits = glob.glob("audit_fixes/lane5/**/seed124_depth0.00_ACC.npy", recursive=True)
    print("Located control files at:", hits)
    if hits:
        CONTROLS_DIR = os.path.dirname(hits[0])
print("CONTROLS_DIR:", CONTROLS_DIR)

acc = np.load(os.path.join(CONTROLS_DIR, "seed124_depth0.00_ACC.npy"))
print("Control ACCURACY distribution at depth=0, seed 124:")
print(" n =", len(acc), "mean =", acc.mean(), "sd =", acc.std(), "min =", acc.min(), "max =", acc.max())
print(" first 10 values:", acc[:10])

healthy = np.load(os.path.join(CONTROLS_DIR, "seed124_depth0.00_healthy.npy"))
print("\nControl HEALTHY recall distribution at depth=0, seed 124:")
print(" n =", len(healthy), "mean =", healthy.mean(), "sd =", healthy.std())
print(" first 10 values:", healthy[:10])

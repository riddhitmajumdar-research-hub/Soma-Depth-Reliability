import pandas as pd, numpy as np, glob, os, re, csv

LOG_DIR = "audit_fixes/lane5/clean_trained_sweep/clean_sweep/logs"
CLASS_NAMES_SORTED = ["cirrhosis", "healthy", "significant_fibrosis"]
FDR_ALPHA = 0.05  # matches cfg.FDR_ALPHA used elsewhere in this pipeline; flag if this project uses a different value

def bh_fdr_correction(p_values, alpha=0.05):
    p = np.asarray(p_values, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresh
    sig = np.zeros(n, dtype=bool)
    if below.any():
        max_i = np.max(np.where(below)[0])
        sig[order[:max_i + 1]] = True
    return sig

detail_files = sorted(glob.glob(os.path.join(LOG_DIR, "lane5_clean_per_class_depth_detail_seed*.csv")))
print(f"Found {len(detail_files)} per-class depth-detail files")

results = []
for fpath in detail_files:
    seed = int(re.search(r"seed(\d+)\.csv", fpath).group(1))
    df = pd.read_csv(fpath)
    df = df.sort_values("depth_cm").reset_index(drop=True)
    depths = df["depth_cm"].values

    per_class_floor = {}
    for name in CLASS_NAMES_SORTED:
        p_vals = df[f"{name}_raw_p_value"].values
        sig_flags = bh_fdr_correction(p_vals, alpha=FDR_ALPHA)
        floor_d = None
        for depth_cm, sig in zip(depths, sig_flags):
            if sig:
                floor_d = depth_cm
        per_class_floor[name] = floor_d if floor_d is not None else 0.0

    results.append({"seed": seed, **{f"per_class_floor_{k}": v for k, v in per_class_floor.items()}})

out = pd.DataFrame(results).sort_values("seed")
print(out.to_string(index=False))
print()
for name in CLASS_NAMES_SORTED:
    x = out[f"per_class_floor_{name}"].values
    print(f"{name}: n={len(x)} mean={x.mean():.2f} sd={x.std(ddof=1):.2f} median={np.median(x):.2f} range={x.min():.2f}-{x.max():.2f} zeros={int((x==0).sum())}")

out.to_csv("/tmp/clean_per_class_floors_FIXED.csv", index=False)
print("\nSaved to /tmp/clean_per_class_floors_FIXED.csv")

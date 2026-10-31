import pandas as pd, numpy as np, glob, re

files = sorted(glob.glob("audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/logs/perclass_balacc_depth_detail_seed*.csv"))
print(f"Found {len(files)} seed files")

CLASSES = ["cirrhosis", "healthy", "significant_fibrosis"]
rows = []
for fpath in files:
    seed = int(re.search(r"seed(\d+)\.csv", fpath).group(1))
    df = pd.read_csv(fpath)
    for cls in CLASSES:
        sub = df[df["class"] == cls].sort_values("depth_cm")
        sig_depths = sub[sub["fdr_significant"] == True]["depth_cm"].values
        floor = sig_depths.max() if len(sig_depths) > 0 else 0.0
        never_sig = len(sig_depths) == 0
        rows.append({"seed": seed, "class": cls, "floor_cm": floor, "never_significant": never_sig})

out = pd.DataFrame(rows)
print(out.pivot(index="seed", columns="class", values="floor_cm").to_string())
print()
for cls in CLASSES:
    x = out[out["class"] == cls]["floor_cm"].values
    n_never = out[out["class"] == cls]["never_significant"].sum()
    print(f"{cls}: n=24 mean={x.mean():.2f} sd={x.std(ddof=1):.2f} median={np.median(x):.2f} range={x.min():.2f}-{x.max():.2f} | never-significant seeds: {n_never}/24")

out.to_csv("/tmp/balacc_floors_by_class.csv", index=False)

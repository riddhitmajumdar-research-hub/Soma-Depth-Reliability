import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu
L5 = pd.read_csv("audit_fixes/lane5/recall_floor_main_run/logs/lane5_per_class_results.csv")
RN = pd.read_csv("Source/logs/degradation_aware_resnet_final_results.csv")
CL = pd.read_csv("audit_fixes/lane5/clean_trained_sweep/clean_sweep/logs/lane5_clean_per_class_results.csv")
WY = pd.read_csv("audit_fixes/lane5/recall_floor_main_run/logs/multiplicity_robustness_main.csv")
def s(x):
    x = np.asarray(x, float)
    return f"n={len(x)} mean={x.mean():.2f} sd={x.std(ddof=1):.2f} median={np.median(x):.2f} range={x.min():.2f}-{x.max():.2f}"
print("LANE5 rerun agg floor      :", s(L5.agg_floor_cm))
print("LANE5 original agg floor   :", s(L5.original_agg_floor_cm))
print("seeds where rerun==original:", int(L5.floor_match.sum()), "of", len(L5))
print("RESNET naive floor         :", s(RN.naive_floor_cm))
print("RESNET last_significant    :", s(RN.last_significant_depth_cm))
for name, cnn in [("original", L5.original_agg_floor_cm), ("rerun", L5.agg_floor_cm)]:
    for col in ["naive_floor_cm", "last_significant_depth_cm"]:
        u, p = mannwhitneyu(RN[col], cnn, alternative="two-sided")
        print(f"MWU ResNet[{col}] vs CNN[{name}]: U={u:.1f} p={p:.4f}")
print("CLEAN agg floor            :", s(CL.agg_floor_cm))
print("clean columns              :", list(CL.columns))
print("WY floor                   :", s(WY.floor_westfall_young_cm), "| zeros:", int((WY.floor_westfall_young_cm == 0).sum()))
print("BH within vs pooled changed:", int(WY.floor_changed_b_vs_a.sum()), "of 24")

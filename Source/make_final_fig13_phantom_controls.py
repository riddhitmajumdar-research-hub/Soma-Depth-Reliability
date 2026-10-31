# Final Fig. 13: ETS Montreal phantom control variants, held-out accuracy on Acqui6, 3 model seeds.
# Reads the saved file outputs/external_validation_controls/ets_montreal_controls_results.json (provenance: PROVENANCE.md in that folder).
# V0 = original design (all columns); V2 = depth labels only, no inclusion in either class; V1 = inclusion vs background at equal depth.
# V1 and V2 use column-mapping mode "M" (primary). This tests the pipeline on real signal only; it says nothing about stiffness sensing
# or the depth boundary. No permutation null was run.
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = f"{ROOT}/outputs/external_validation_controls/ets_montreal_controls_results.json"
OUT = f"{ROOT}/outputs/figures_final/fig13_phantom_controls.png"
res = json.load(open(SRC))
pick = lambda var, mode: next(r for r in res if r["variant"] == var and r["mode"] == mode)
conds = [("Original design\n(all columns)", pick("V0", "-"), "#A23B3B"),
         ("Depth labels only\n(no inclusion in either class)", pick("V2", "M"), "#1F3A5F"),
         ("Inclusion vs background\nat equal depth", pick("V1", "M"), "#2E7D4F")]
fig, ax = plt.subplots(figsize=(7.6, 5.2))
rng = np.random.default_rng(0)
for i, (name, r, color) in enumerate(conds):
    accs = np.array(r["accs"]) * 100
    assert len(accs) == 3
    ax.bar(i, accs.mean(), width=0.6, color=color, alpha=0.55)
    ax.errorbar(i, accs.mean(), yerr=accs.std(ddof=1), color="black", capsize=5, lw=1.3)
    ax.scatter(i + rng.uniform(-0.12, 0.12, 3), accs, color="black", s=22, zorder=4, label="Individual model seed (n = 3)" if i == 0 else None)
    ax.text(i, 3, f"{accs.mean():.2f} ± {accs.std(ddof=1):.2f}%", ha="center", color="white", fontweight="bold", fontsize=10)
    print(f"{name.splitlines()[0]}: mean {accs.mean():.2f} +/- {accs.std(ddof=1):.2f} (JSON acc_mean {r['acc_mean']*100:.2f}, acc_sd {r['acc_sd']*100:.2f}); seeds {np.round(accs,2).tolist()}")
ax.axhline(50, color="gray", linestyle="--", lw=1.2, label="Chance (50%)")
ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in conds], fontsize=9)
ax.set_ylim(0, 105)
ax.set_ylabel("Held-out accuracy, test run Acqui6 (%)")
ax.set_title("ETS Montreal phantom: control variants (3 seeds)")
ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.78), fontsize=8.5)
ax.grid(alpha=0.3, axis="y")
fig.text(0.5, -0.03, "Bars: mean; whiskers: SD over 3 seeds. Tests that the pipeline runs on real signal; no permutation null was run.\nIt does not test stiffness sensing or the depth boundary.",
         ha="center", fontsize=8, style="italic")
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", os.path.abspath(OUT))

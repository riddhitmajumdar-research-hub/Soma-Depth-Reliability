# audit_fixes

Work from the audit. Each directory is one lane of that work.

`docs/methodology/README.md` is the index. It has the table that maps lane names to note numbers (the two numberings differ), the topic finder, and the list of open gaps.

| Directory | Question | Note |
|---|---|---|
| [lane1](lane1/README.md) | How far is the weak-damping attenuation formula from the exact solution? | [04](../docs/methodology/04-lane1-exact-attenuation.md) |
| [lane2](lane2/README.md) | Do the reported p-values and effect sizes reproduce, and does significance survive variance correction? | [05](../docs/methodology/05-lane2-and-q6-statistical-rechecks.md) |
| [q6_bh_fdr_families](q6_bh_fdr_families/README.md) | What is the BH family size, and can 200 shuffles pass it? | [05](../docs/methodology/05-lane2-and-q6-statistical-rechecks.md) |
| [lane3](lane3/README.md) | Do nine factual claims about the code and logs hold? | [06](../docs/methodology/06-lane3-fact-checks.md) |
| [lane4](lane4/README.md) | Do SVM, random forest and GRU baselines match the CNN on the clean task? | [07](../docs/methodology/07-lane4-baseline-architectures.md) |
| [lane5](lane5/README.md) | Can the aggregate floor protocol certify per-class floors over 24 seeds? | [09](../docs/methodology/09-lane5-per-class-certification.md) |
| [lane6](lane6/README.md) | Which feature of the synthetic windows works without any signal shape? | [08](../docs/methodology/08-lane6-beat-offset-mechanism.md) |
| [lane7](lane7/README.md) | Is the ResNet1D floor distribution different from the CNN's? | [10](../docs/methodology/10-lane7-resnet1d-vs-cnn.md) |
| [lane8](lane8/README.md) | Is the depth floor a property of the CNN or of the physics and noise? | [11](../docs/methodology/11-lane8-and-cloud-random-forest.md) |

## What is not in this tree

- Model weights. `lane5/recall_floor_main_run/seed_weights/`, `lane5/clean_trained_sweep/clean_sweep/seed_weights/` and `lane5/recall_floor_main_run/quarantine_seed103_partial/` exist on the machine that ran them and are ignored by git (about 580 MB).
- The per-seed confusion matrices of the robustness battery. The 24-seed random forest results and the battery CSVs are in `audit_fixes/cloud/rf/` and `audit_fixes/cloud/battery/results/`. See notes 11 and 12.

## Reading order

Start with `docs/methodology/README.md`, open the note for the lane, then the lane's own README. Lane 5 and Lane 7 each have a `REPORT.md` that predates their final results; both carry a notice at the top saying where the current numbers are.

# Provenance: ETS Montreal phantom control variants (Fig. 13)

| Item | Value |
|---|---|
| Saved result file | `ets_montreal_controls_results.json` (byte copy of the scratch file below) |
| Original location | `~/soma_audit_scratch/trackB_local/results.json` (outside the repo) |
| Producing script | `~/soma_audit_scratch/trackB_local/run_variants.py`, copied here as `run_variants_scratch_copy.py`. The same script is in the repository as `Source/ets_controls_run_variants.py`, with its helper `Source/ets_controls_common.py`. It reuses `train_cnn` and `evaluate`, extracted by AST from the repo's `Source/ets_montreal_path2_retrain.py`. |
| Report | `REPORT_trackB_scratch_copy.md` (byte copy of the scratch REPORT.md) |
| Data | Public ETS Montreal / Borealis ultrasound phantom dataset (`data/ets_montreal/`, gitignored raw data) |
| Split | Acqui1-5 train, Acqui6 test; classes balanced by subsampling; 3 model seeds (1, 2, 3); SD over the 3 seeds (ddof = 1) |
| Variants | V0 = original design, all 256 columns, inclusion vs background; V2 = depth labels only (rows 1526-1626 vs 300-400), no inclusion columns in either class; V1 = inclusion vs background at equal depth. V1 and V2 shown here use mode M (the primary column mapping, which assumes the 256 A-lines span the displayed image width: an assumption, not documented) |
| Not run | No permutation null; chance is 50%; no collapsed model in any run |
| Status | The plotting script (`Source/make_final_fig13_phantom_controls.py`) reads the JSON copy in this folder. The generating script is in the repository (`Source/ets_controls_run_variants.py`), but it writes its result to a path outside the repository, so the JSON in this folder is a copy of that output. |

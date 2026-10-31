# Deviations from PREREG.md

1. `summarize_proxy.py` gained a `--reference-root` option (default: the repository root). Reason: the consistency checks of PREREG.md section 6 read result files of the robustness battery, the certified sweep and the clean-trained sweep, which are on other branches and are not in this branch's tree; the checks were run with the reference root set to the release folder. This is an execution fix made after the results existed. No threshold, seed, condition, test or floor was changed.
2. Hardware and execution (declared before any result in PREREG.md section 3): eight single-thread processes on one Mac. No restart, interruption or failed seed occurred.

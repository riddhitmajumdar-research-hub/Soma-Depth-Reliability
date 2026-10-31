# Q6 — BH-FDR under three families (clean CNN, 100 Hz, 27.05 dB)

## What was done

1. Located the two scripts named in the question with `find`/`grep` (no names assumed):
   - [`Source/floor_characterization_cnn.py`](../../Source/floor_characterization_cnn.py) — degradation sweep over 4 frequencies × depth, fixed noise at `SENSOR_SNR_DB`.
   - [`Source/noise_robustness_stress_test_clean_cnn.py`](../../Source/noise_robustness_stress_test_clean_cnn.py) — the clean-CNN SNR-crosscheck, fixed at `NOMINAL_FREQ_HZ` (100 Hz), swept over `SNR_SWEEP_DB` × depth.
2. Checked whether either script saves raw per-cell p-values to a CSV:
   - `floor_characterization_cnn.py` does **not** write a CSV. It only pickles its 82 result rows to `Source/results/stage4_cnn_results.pkl` (identical copy also at `outputs/stage4/data/stage4_cnn_results.pkl` — verified byte-for-byte equal, `r1==r2` True). The pickle does contain the raw `p_value` per cell, so I used it read-only (no retraining) rather than stopping at "not CSV."
   - `noise_robustness_stress_test_clean_cnn.py` **does** save raw p-values to CSV: [`outputs/stage4/data/noise_robustness_stress_test_clean_cnn.csv`](../../outputs/stage4/data/noise_robustness_stress_test_clean_cnn.csv), column `p_value`, 115 rows = 5 SNR levels × 23 depths.
3. Wrote and ran [`q6_recompute_bh_fdr.py`](q6_recompute_bh_fdr.py) (heredoc-complete, imports `SENSOR_SNR_DB` and `NOMINAL_FREQ_HZ` from `Source/constants.py`, no hardcoded constants). It re-implements the standard BH step-up procedure independently (not imported from the repo) and applies it under three families. Raw stdout saved to [`q6_recompute_bh_fdr_stdout.log`](q6_recompute_bh_fdr_stdout.log); full structured output at [`q6_bh_fdr_results.json`](q6_bh_fdr_results.json).

Files read (all pre-existing, none modified):
- `Source/constants.py`
- `Source/floor_characterization_cnn.py`
- `Source/noise_robustness_stress_test_clean_cnn.py`
- `Source/results/stage4_cnn_results.pkl`
- `outputs/stage4/data/stage4_cnn_results.pkl`
- `outputs/stage4/data/noise_robustness_stress_test_clean_cnn.csv`
- `outputs/figures_earlier/manifest.md` (to check the certified number, see below)

## (a) Floor under each family

| Family | Source | n | # BH-rejected | Floor |
|---|---|---|---|---|
| **Pooled 82-cell** | `stage4_cnn_results.pkl`, all 4 freqs, one BH pass over all 82 p-values | 82 | 23/82 | **20Hz: none in tested range; 100Hz: 4.30 cm; 250Hz: 0.75 cm; 500Hz: 0.30 cm** |
| **100Hz-only (24 cells)** | same pickle, BH pass restricted to the 24 rows with `freq==100` | 24 | 0/24 | **0.00 cm** |
| **23-depth per-SNR** | crosscheck CSV, rows with `snr_db==27.05` (100Hz fixed) | 23 | 0/23 | **0.00 cm** |

**Check value / MATCH-MISMATCH:** the record's manifest (`outputs/figures_earlier/manifest.md:52`) states "the same run that produced the certified 4.30cm floor... lives in `Source/results/stage4_cnn_results.pkl`". My recomputation of the **pooled 82-cell family** reproduces **4.30 cm at 100Hz exactly** → **MATCH**. The certified number is the pooled-family number, not the 100Hz-only or per-SNR number.

The 100Hz-only and 23-depth families both collapse the floor to 0 cm — **not because accuracy fails at the shallowest depth**, but because *no cell in either narrower family survives BH at all* (0/24 and 0/23 rejected). Under the code's own floor definition (`first depth, scanning from shallowest, that is not BH-significant`), a family with zero significant cells anywhere reports the shallowest tested depth as the floor by construction. This is a definitional artifact of narrowing the family, not a re-measurement of degraded accuracy.

## (b) 23-depth family: raw p-values, BH rejections, and why the floor reads 0 cm

Per-depth p-values (snr=27.05, 100Hz), sorted by depth:

| depth (cm) | p-value | BH reject (recomputed) | original CSV `fdr_significant` |
|---|---|---|---|
| 0.0000 | 0.004975 | False | False |
| 1.3636 | 0.004975 | False | False |
| 2.7273 | 0.009950 | False | False |
| 3.5000 | 0.019900 | False | False |
| 4.0000 | 0.019900 | False | False |
| 4.0909 | 0.029851 | False | False |
| 4.3000 | 0.089552 | False | False |
| 4.5000 | 0.069652 | False | False |
| 5.0000 | 0.084577 | False | False |
| 5.4545 | 0.194030 | False | False |
| 5.5000 | 0.149254 | False | False |
| 6.0000 | 0.189055 | False | False |
| 6.5000 | 0.746269 | False | False |
| 6.8182 | 0.751244 | False | False |
| 7.0000 | 0.756219 | False | False |
| 7.5000 | 0.820896 | False | False |
| 8.0000 | 0.830846 | False | False |
| 8.1818 | 0.830846 | False | False |
| 9.5455 | 0.885572 | False | False |
| 10.9091 | 0.900498 | False | False |
| 12.2727 | 0.900498 | False | False |
| 13.6364 | 0.875622 | False | False |
| 15.0000 | 0.945274 | False | False |

**Check value:** recomputed BH flags match the CSV's own saved `fdr_significant` column at every one of the 23 depths → **MATCH** (the original script's per-SNR BH pass was applied correctly; this is a re-derivation, not a fix).

**BH mechanics — none rejected:**

BH-FDR rejects hypothesis at rank *k* (p-values sorted ascending) if `p_(k) ≤ (k/n)·α`, and if so, rejects all ranks `≤` the largest such *k*. Here n=23, α=0.05, so rank-1 threshold = `(1/23)·0.05 = 0.002174`.

- The smallest p-value in this family is **0.004975**, at depth 0.0 cm — the *best possible* case (shallowest depth, highest real accuracy). `0.004975 > 0.002174` → **fails at rank 1**.
- Rank 2 (also depth 1.3636 cm, tied p=0.004975): threshold `= (2/23)·0.05 = 0.004348`. `0.004975 > 0.004348` → **still fails**.
- Every subsequent rank fails likewise (see full ranked table in the log/JSON) — no p-value in the family ever drops below its own rank's threshold, so **zero cells are rejected anywhere in the 23-depth family**, regardless of depth.

**Root cause — this is a permutation-resolution ceiling, not a depth effect:** the p-value is computed as `(count(control_acc ≥ acc_real) + 1) / (N_SHUFFLES + 1)` with `N_SHUFFLES = 200`, so the *smallest p-value the test can ever produce* is `1/201 = 0.0049751...` — exactly the value observed at depth 0.0 cm. The rank-1 BH threshold for a 23-member family is `0.05/23 = 0.0021739`, which is **smaller than the test's own achievable minimum p-value**. It is mathematically impossible for any single cell in a 23-member (or 24-member, threshold `0.05/24 = 0.0020833`) family built from 200 shuffles to survive BH correction, no matter how large the real-vs-control accuracy gap is. That is why the floor reads 0 cm in both the 100Hz-only and 23-depth families: not because the CNN fails at the surface, but because narrowing the family to ≤24 members combined with `N_SHUFFLES=200` puts every possible p-value above the smallest attainable BH threshold.

By contrast, the pooled 82-cell family's rank-*k* thresholds grow more slowly at low rank (`k/82` vs `k/23`), but because 23/82 cells across all four frequencies share p-values at or near the 1/201 floor, enough of them cluster at low ranks that the crossing point (largest *k* with `p_(k) ≤ (k/82)·0.05`) is reached — which is why pooling recovers non-trivial floors (4.30 cm at 100 Hz) while the narrower families recover none.

## Open questions / things I did not resolve

- **This is a genuine methodological sensitivity worth flagging to you, not something I fixed or chose a "right" family for.** The certified 4.30 cm number is reproduced exactly under the pooled-family correction (MATCH with the manifest), so nothing in the certified result is contradicted. But the 100Hz-only and per-SNR-at-27.05dB framings both silently floor to 0 cm purely from permutation-count resolution vs. family size — this is a property of the statistical test design (`N_SHUFFLES=200` inside `bh_fdr_correction`), not of the underlying model or physics. I did not increase `N_SHUFFLES` or otherwise attempt to fix this — that would mean retraining/rerunning the existing sweep scripts, which is out of scope for a read-only audit lane.
- I have not seen the record's actual prose for how the "clean CNN at 100Hz/27.05dB" is characterized outside the pooled floor — if the record (or any other section) separately quotes a 100Hz-only or per-SNR floor number distinct from 4.30 cm, that would need to be checked against these results before this can be called fully resolved. I did not search exhaustively beyond `outputs/figures_earlier/manifest.md` for this.

## File list

- `audit_fixes/q6_bh_fdr_families/q6_recompute_bh_fdr.py` — analysis script (read-only on the repo)
- `audit_fixes/q6_bh_fdr_families/q6_recompute_bh_fdr_stdout.log` — raw stdout from running it
- `audit_fixes/q6_bh_fdr_families/q6_bh_fdr_results.json` — full structured output (all three families, ranked BH tables, per-depth data)
- `audit_fixes/q6_bh_fdr_families/REPORT.md` — this report

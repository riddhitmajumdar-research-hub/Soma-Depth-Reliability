# Track B: V0 / V1 / V2 lateral-vs-depth audit

Source: `run_variants.py`, log `run_log.txt`, numbers `results.json`. All numbers below come from that run.
Reused from `Source/ets_montreal_path2_retrain.py` (extracted via ast, loop not run): `train_cnn`, `evaluate`, row windows 1526-1626 / 300-400, 60 epochs, SimpleCNN, per-window z-score (same formula, vectorised).
Split: Acqui1-5 train, Acqui6 test. Classes balanced by subsampling in train and test (rng seed = model seed). Seeds 1,2,3. SD is over 3 seeds (ddof=1).
**No permutation null was run.** Chance = 50%. No model collapsed (a class never predicted) in any run.

## Column definitions
Primary (M) assumes the 256 A-lines span the displayed 303-px image width (x=169-471); this is an assumption, not documented. Columns = floor((x-169)/303*256).
Inclusion columns per sequence (measured span shrunk 10): Acqui1 82-174 (93), Acqui2 75-179 (105), Acqui3 71-172 (102), Acqui4 78-175 (98), Acqui5 85-179 (95), Acqui6 85-179 (95). Non-inclusion = outside span widened 10 (111-123 columns each). All above the 15-column minimum.
Secondary: H-a inclusion 113-143, non-inclusion 0-50 and 206-255. H-b (my reading of Step 3: H-b span 60.6-195.4) inclusion 71-185, non-inclusion 0-50 and 206-255.

## Results (test accuracy on Acqui6)
| Variant | Columns | Acc mean +- SD | Recall label1 / label0 | Train windows (l1/l0 before balance) | Test windows (l1/l0) |
|---|---|---|---|---|---|
| V0 | all 256 | 96.78 +- 0.68% | 97.34 / 96.22 | 45056/45056 | 8192/8192 |
| V1 | M (primary) | 59.10 +- 0.61% | 57.05 / 61.14 | 17342/20674 | 3040/3872 |
| V2 | M (primary) | 95.99 +- 0.82% | 96.04 / 95.95 | 20674/20674 | 3872/3872 |
| V1 | H-a (secondary) | 59.69 +- 1.26% | 51.58 / 67.81 | 5456/17776 | 992/3232 |
| V1 | H-b (secondary) | 61.29 +- 0.55% | 56.40 / 66.18 | 20240/17776 | 3680/3232 |
| V2 | H-a (secondary) | 95.55 +- 0.91% | 95.58 / 95.53 | 17776/17776 | 3232/3232 |
| V2 | H-b (secondary) | 95.55 +- 0.91% | identical | identical | identical |

V2 under H-a and H-b are the same computation (same non-inclusion columns, same seeds), so the identical numbers are expected, not two independent results.

## RF energy (raw, mean squared per window, M columns, mean over 6 sequences)
a (inclusion cols, rows 1526-1626) = 4.93e5; b (non-inclusion, same rows) = 1.415e5; c (non-inclusion, rows 300-400) = 8.207e3.
a/b = 3.48 (per sequence 2.82-4.13). b/c = 17.25 (per sequence 13.97-21.08).

## What each result implies, and what it does not show
- **V0 ~97% reproduced (96.8%).** Implies the original headline is reproducible under this loader. Does not show the model senses the inclusion: V0 confounds depth region, lateral position (all 256 columns, of which only about 20-50% lie over the inclusion) and echo strength.
- **V2 ~96% with no inclusion columns in either class.** Implies depth region alone (rows 1526-1626 vs 300-400) separates the classes at near the V0 level, so the V0 accuracy does not require inclusion tissue. Does not show which feature carries it; raw energy differs 17x (b/c) but inputs are per-window z-scored, so amplitude is not directly available, and what remains (spectral content, attenuation/depth-dependent texture, TGC effects) was not tested.
- **V1 ~59% (H-a 59.7%, H-b 61.3%).** Implies that at equal depth the model separates inclusion from background only weakly; the signal is well below V0/V2. Does not show a statistically significant effect, since no permutation null was run (3 seeds only, and Acqui6 windows are correlated), and a high V1 would in any case not prove stiffness sensing: inclusion and background also differ in echo strength (a/b = 3.48) and speckle texture. Does not show sensing of stiffness in either direction.
- Mislabelling: originally all-256-column "inclusion" windows were 80% (H-a) or 47% (H-b) lateral background (Track B step 4). V1 removes that; the drop from ~97% to ~59% is consistent with the original accuracy being mostly depth-driven.
- All M results depend on the pixel-to-column assumption. Secondary H-a/H-b are hypothesis-dependent. Results are for one held-out sequence (Acqui6), 3 seeds, one architecture.

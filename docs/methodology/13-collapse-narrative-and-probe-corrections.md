# 13. Collapse narrative and frozen-probe corrections

Two corrections found while regenerating Fig. 10 and Fig. 11. Neither changes a certified number; both change what the text says about them.

## 1. Which class collapses into which (Fig. 10)

**What was wrong.** The README (Log 14, line 548, and the "reproduced independently" paragraph near line 773) says that at the collapse depth every healthy sample is misclassified as cirrhosis, and ties this to healthy tissue attenuating fastest. That came from one model and one seed's confusion matrix.

**What the data showed.** `Source/make_fig10_pooled_confusion.py` sums the saved seed 101 to 124 confusion matrices (`audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/confusion_matrices`, degradation-aware CNN, 100 Hz, 2400 windows per class per depth). No model is re-evaluated. Healthy stays healthy:

| Depth (cm) | healthy correct | healthy to cirrhosis |
|---|---|---|
| 0.00 | 81.38% | 0.00% |
| 2.73 | 93.17% | 0.00% |
| 4.30 | 99.46% | 0.38% |
| 6.82 | 95.21% | 4.79% |

The errors run the other way. At 6.82 cm, 1135 of 2400 cirrhosis windows are called healthy and 579 of 2400 fibrosis windows are called healthy (884 go to cirrhosis). Healthy acts as the sink class, which fits the noise-probe result in note 12 (healthy takes 48 to 100% of pulse-free noise).

**What changed.** `outputs/stage4/figures/stage4_multidepth_confusion_degaware_pooled.png` replaces the single-seed picture. The README text has not been rewritten; the old Log 14 paragraph is the historical record and should be read against this note.

**Limits.** The saved matrices stop at 6.82 cm. The 5.45 cm and 8.18 cm cases in the old text are not covered by the pooled figure, and the pooled matrices are for the degradation-aware model, whereas the old 8.18 cm collapse was the clean-trained one. So this contradicts the "healthy into cirrhosis" claim for the certified model on 0 to 6.82 cm; it does not by itself say what the clean-trained model does at 8.18 cm.

## 2. Frozen-embedding probe under the wrong noise level (Fig. 11)

**What was wrong.** `Source/frozen_embedding_linear_probe.py` generated its held-out set at 67.03 dB, the superseded sensor noise level. The corrected level is 27.05 dB (`SENSOR_SNR_DB` in `constants.py`). The old result, probe 100% against deployed 33.3% at 8.18 cm (`Source/logs/frozen_embedding_linear_probe_8p18cm.csv`), was therefore measured on nearly noiseless data.

**What the rerun showed.** `Source/frozen_embedding_linear_probe_27p05dB.py` is a copy with only the noise level, depth argument and CSV path changed. The CNN is trained on clean windows, so its weights do not depend on the noise level. 200 held-out windows per class:

| Depth (cm) | deployed model | frozen probe | chance |
|---|---|---|---|
| 8.18 | 33.3% | 75.8% | 33.3% |
| 8.00 | 33.3% | 77.1% | 33.3% |

The probe still beats chance by 42.5 and 43.8 points, so the representation keeps class information the argmax boundary has lost. It is not 100%.

**What changed.** New CSVs `Source/logs/frozen_embedding_linear_probe_27p05dB_{8p18,8p0}cm.csv`, new script `Source/make_fig11_corrected_probe.py`, new figure `outputs/figures_earlier/fig3_06_frozen_probe_vs_deployed_27p05dB.png`. The 67.03 dB script, CSV and figure are untouched as the historical record.

**Limits.** One seed, one held-out draw per depth, no confidence interval on the 75.8 and 77.1 figures.

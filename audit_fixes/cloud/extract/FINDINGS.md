# extract — FINDINGS.md

Read-only extraction from `results/floors_105-108.csv` on
`run battery-105-108` (latest commit `<commit>` at time of this
check). No retraining, no recomputation — every number below is read
directly out of that CSV. Condition label for the 2.6x healthy/fibrosis
viscosity multiplier confirmed against that branch's PREREG.md: it is
`7c_viscosity_healthy_fibrosis_2.6x`.

---

## (a) baseline | condition 2 (amplitude randomization, corrected) | shift

| seed | baseline | condition 2 | shift |
|---|---|---|---|
| 105 | 8.000 | 8.182 | +0.182 |
| 106 | 9.545 | 10.909 | +1.364 |
| 107 | 7.500 | 8.000 | +0.500 |
| 108 | 10.909 | 13.636 | +2.727 |

## (b) baseline | viscosity 2.6x (condition 7c) | shift

| seed | baseline | 7c (visc 2.6x) | shift |
|---|---|---|---|
| 105 | 8.000 | 10.909 | +2.909 |
| 106 | 9.545 | 9.545 | +0.000 |
| 107 | 7.500 | 9.545 | +2.045 |
| 108 | 10.909 | 13.636 | +2.727 |

---

## Raw CSV rows, unreformatted (for direct double-check against source)

Baseline rows:
```
105,1_baseline,baseline,8.0,7.5,False
106,1_baseline,baseline,9.545454545454545,13.636363636363635,False
107,1_baseline,baseline,7.5,7.0,False
108,1_baseline,baseline,10.909090909090908,15.0,False
```

Condition-2 rows:
```
105,2_amp_randomization,"amplitude randomisation g~U(0.5,1.5) linear (corrected; E[g]=1.0)",8.181818181818182,8.0,False
106,2_amp_randomization,"amplitude randomisation g~U(0.5,1.5) linear (corrected; E[g]=1.0)",10.909090909090908,9.545454545454545,False
107,2_amp_randomization,"amplitude randomisation g~U(0.5,1.5) linear (corrected; E[g]=1.0)",8.0,7.5,False
108,2_amp_randomization,"amplitude randomisation g~U(0.5,1.5) linear (corrected; E[g]=1.0)",13.636363636363635,12.272727272727272,False
```

Condition 7c (viscosity healthy/fibrosis x2.6) rows:
```
105,7c_viscosity_healthy_fibrosis_2.6x,eta healthy/fibrosis x2.6 (cirrhosis fixed 1.77),10.909090909090908,12.272727272727272,False
106,7c_viscosity_healthy_fibrosis_2.6x,eta healthy/fibrosis x2.6 (cirrhosis fixed 1.77),9.545454545454545,8.181818181818182,False
107,7c_viscosity_healthy_fibrosis_2.6x,eta healthy/fibrosis x2.6 (cirrhosis fixed 1.77),9.545454545454545,13.636363636363635,False
108,7c_viscosity_healthy_fibrosis_2.6x,eta healthy/fibrosis x2.6 (cirrhosis fixed 1.77),13.636363636363635,15.0,False
```

(CSV column order: `seed,condition,condition_label,floor_cm,last_significant_cm,censored`.)

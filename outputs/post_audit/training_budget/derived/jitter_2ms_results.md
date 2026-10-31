# Stage 2D results: timing jitter J = 2.0 ms vs J = 1.0 ms, 200 epochs, Phase 3 CNN, seeds 101-108, K = 1, B = 719
J is the half-width of a uniform [-J, +J] distribution (SD = J/sqrt(3): 0.577 ms at J = 1.0, 1.155 ms at J = 2.0). Floors in cm, new rule.

| seed | floor J = 1.0 (Stage 2 draw 0) | floor J = 2.0 | difference (2.0 - 1.0) |
|---|---|---|---|
| 101 | 13.6364 | 10.9091 | -2.7273 |
| 102 | 12.2727 | 10.9091 | -1.3636 |
| 103 | 16.3636 | 12.2727 | -4.0909 |
| 104 | 13.6364 | 10.9091 | -2.7273 |
| 105 | 13.6364 | 13.6364 | +0.0000 |
| 106 | 15.0000 | 12.2727 | -2.7273 |
| 107 | 13.6364 | 13.6364 | +0.0000 |
| 108 | 15.0000 | 13.6364 | -1.3636 |

(* censored at 20 cm)

## J = 2.0 ms across seeds
n = 8; mean 12.273; SD (ddof=1) 1.262; median 12.273; min 10.909; max 13.636. J = 1.0 reference: mean 14.148; SD 1.249; median 13.636.
Paired difference: 0 up / 2 equal / 6 down; exact two-sided binomial sign test, zero differences dropped, n = 6, p = 0.03125.
Seeds with floor 0.00 at J = 2.0: none.

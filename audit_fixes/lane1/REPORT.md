# Lane 1 — Weak-Damping Approximation Error Quantification

## Scope

Quantify how much the weak-damping approximation used everywhere in this
codebase's Kelvin-Voigt attenuation model (`Source/attenuation_model.py`)
diverges from the exact complex-wavenumber solution, across the operating
band and three tissue classes, and answer three follow-on sanity questions.
Read-only against the existing codebase; all new code and outputs live under
`audit_fixes/lane1/`.

## Files used (discovered via `find`/`grep`, not assumed)

- `Source/constants.py` — `SEED=42`, `NOMINAL_FREQ_HZ=100`,
  `MAX_VALID_DEPTH_CM=6.50`, comment documenting the 24-seed mean floor
  `8.48cm +/- 1.81cm`.
- `Source/physics_model.py` — `stiffness_classes = {healthy: 4.5, significant_fibrosis: 7.5, cirrhosis: 12.0}` kPa, `rho=1060` default, `G = E/3`, `c0 = sqrt(G/rho)`.
- `Source/attenuation_model.py` — `LIVER_VISCOSITY_BY_CLASS = {healthy: 0.5, significant_fibrosis: 0.5, cirrhosis: 1.77}` Pa·s, and the weak-damping formula `alpha = omega**2 * eta / (2*G*c)`. `attenuate_signal()` confirms amplitude decay is `exp(-alpha * depth_m)` — matches the task's assumed form.
- `Source/joint_uncertainty.py` — existing MC machinery: `N_SAMPLES=5000`, `VISCOSITY_RANGE=(0.5, 1.77)` uniform, `DENSITY_RANGE=(1060*0.92, 1060*1.08)` uniform, seeded with `constants.SEED`. Reused verbatim for Task 5 (imported, not retyped).
- `Source/weak_damping_error_bound.py` — a pre-existing *perturbative* (leading-order, small-`x`) error bound at 250/500Hz only. Different method from this lane's exact solution; not overwritten, cross-referenced below.
- `Source/logs/degradation_aware_final_results.csv` — verified the 8.48cm/1.81cm figure independently: `naive_floor_cm` column, n=24, mean=8.4754cm→8.48cm, SD(ddof=1)=1.8122cm→1.81cm. **MATCH.**

## Constants verification against `constants.py` and physics scripts

All physical constants (E per class, rho, eta per class) were pulled by
import from `physics_model.py` / `attenuation_model.py`, not retyped from the
protocol. They matched the protocol's stated values exactly: E={4.5, 7.5, 12.0}
kPa, rho=1060, eta={0.5, 0.5, 1.77} Pa·s. Amplitude-at-depth form confirmed
as `exp(-alpha*x)` directly from `attenuation_model.attenuate_signal()`.

## Task 1 — Exact vs. weak-damping attenuation

Script: [`task1_exact_vs_weak.py`](task1_exact_vs_weak.py) · stdout: [`task1_stdout.log`](task1_stdout.log) · data: [`table1_exact_vs_weak.csv`](table1_exact_vs_weak.csv)

Exact solution: `k = omega*sqrt(rho/(G + i*omega*eta))`, `alpha_exact = |Im(k)|`, `c_exact = omega/Re(k)`. (Note: `numpy.sqrt`'s principal branch on this complex input returns `Im(k)<0`, which is the non-physical growing-wave branch; the script takes `abs(Im(k))` to select the decaying-wave branch, documented inline in the script.)

| class | freq (Hz) | ω·η/G | α_weak (1/m) | α_exact (1/m) | ratio | c0 (m/s) | c_exact (m/s) |
|---|---|---|---|---|---|---|---|
| healthy | 20 | 0.042 | 2.21 | 2.21 | 1.001 | 1.190 | 1.190 |
| healthy | 100 | 0.209 | 55.31 | 53.85 | 1.027 | 1.190 | 1.209 |
| healthy | 250 | 0.524 | 345.70 | 296.85 | 1.165 | 1.190 | 1.302 |
| healthy | 500 | 1.047 | 1382.79 | 863.19 | 1.602 | 1.190 | 1.557 |
| significant_fibrosis | 20 | 0.025 | 1.03 | 1.03 | 1.000 | 1.536 | 1.536 |
| significant_fibrosis | 100 | 0.126 | 25.71 | 25.46 | 1.010 | 1.536 | 1.545 |
| significant_fibrosis | 250 | 0.314 | 160.67 | 151.47 | 1.061 | 1.536 | 1.591 |
| significant_fibrosis | 500 | 0.628 | 642.66 | 521.09 | 1.233 | 1.536 | 1.737 |
| cirrhosis | 20 | 0.056 | 1.80 | 1.80 | 1.002 | 1.943 | 1.945 |
| cirrhosis | 100 | 0.278 | 44.96 | 42.92 | 1.048 | 1.943 | 1.997 |
| cirrhosis | 250 | 0.695 | 281.03 | 219.13 | 1.282 | 1.943 | 2.247 |
| cirrhosis | 500 | 1.390 | 1124.10 | 563.66 | 1.994 | 1.943 | 2.857 |

**Checks against independently supplied values — all MATCH:**

| check | expected | actual | result |
|---|---|---|---|
| ratio, healthy, 100Hz | 1.027 | 1.02722 | MATCH |
| ratio, significant_fibrosis, 100Hz | 1.010 | 1.00984 | MATCH |
| ratio, cirrhosis, 100Hz | 1.048 | 1.04773 | MATCH |
| α_weak, healthy, 100Hz | 55.31 | 55.31154 | MATCH |
| α_weak, cirrhosis, 100Hz | 44.96 | 44.96410 | MATCH |
| ratio, cirrhosis, 250Hz | 1.28 | 1.28245 | MATCH |
| ratio, cirrhosis, 500Hz | 1.99 | 1.99429 | MATCH |

Interpretation: the weak-damping approximation over-estimates attenuation
(`alpha_weak > alpha_exact`) at every class/frequency tested, and the
divergence grows with `omega*eta/G` as expected. At the certified operating
point (100Hz) the over-estimate is small (2.7%–4.8%), but at 250/500Hz it
grows to 16%–99%. This is directionally consistent with, but not identical
in method to, the pre-existing perturbative bound in
`Source/weak_damping_error_bound.py`, which independently flags 250/500Hz as
outside the small-`x` regime — the exact solution here confirms that flag
quantitatively rather than only via the leading-order series.

## Task 2 — First-order floor-shift bound at 100Hz

Script: [`task2_floor_shift_bound.py`](task2_floor_shift_bound.py) · stdout: [`task2_stdout.log`](task2_stdout.log) · data: [`task2_floor_shift_bound.csv`](task2_floor_shift_bound.csv)

`x_eq = x_ref * (alpha_weak/alpha_exact)` at 100Hz, for `x_ref` = 8.48cm (24-seed mean floor) and 6.50cm (`MAX_VALID_DEPTH_CM`):

| class | reference | x_ref (cm) | x_eq (cm) | shift (cm) |
|---|---|---|---|---|
| healthy | 8.48cm floor | 8.48 | 8.711 | +0.231 |
| healthy | 6.50cm gate | 6.50 | 6.677 | +0.177 |
| significant_fibrosis | 8.48cm floor | 8.48 | 8.564 | +0.084 |
| significant_fibrosis | 6.50cm gate | 6.50 | 6.564 | +0.064 |
| cirrhosis | 8.48cm floor | 8.48 | 8.885 | +0.405 |
| cirrhosis | 6.50cm gate | 6.50 | 6.810 | +0.310 |

**Label: this is a bound on physics-model error only** (how far the exact
attenuation law shifts an amplitude-matched depth relative to the weak-damping
law), **not a re-certified accuracy floor.** Re-certifying the floor requires
re-running the eval pipeline on the saved seed121 weights
(`outputs/stage4/models/soma_certified_model_seed121_corrected_noise.pt`),
which is explicitly out of scope for this lane.

Worst case is cirrhosis: up to +0.41cm shift relative to the 8.48cm floor,
+0.31cm relative to the 6.50cm safety gate — both shifts are in the direction
of the exact model tolerating slightly *more* depth than the weak model
implies (since alpha_exact < alpha_weak), not less.

## Task 3 — Arrival-time (TOF) sensitivity and class separability at fs=500Hz

Script: [`task3_arrival_time_sensitivity.py`](task3_arrival_time_sensitivity.py) · stdout: [`task3_stdout.log`](task3_stdout.log) · data: [`table3_arrival_time_sensitivity.csv`](table3_arrival_time_sensitivity.csv)

`fs=500Hz` (→2ms/sample) was verified directly from
`physics_model.generate_signal`'s `sample_rate` default (assert in script),
not assumed. TOF computed for both `c0` and `c_exact` (100Hz) at d ∈
{5,7.5,10,15,20}mm; class-pair gaps compared to the 2ms sample period.

**Result: 14 of 30 (distance × class-pair × velocity-model) combinations
fall below 1 sample of gap and are NOT reliably separable after
quantisation to 2ms samples.** Specifically:
- At d=5mm, **no** class pair clears 1 sample under either velocity model (max gap 0.82 samples, healthy-vs-cirrhosis).
- At d=7.5mm and 10mm, only healthy-vs-cirrhosis clears 1 sample; the other two pairs (healthy-vs-fibrosis, fibrosis-vs-cirrhosis) do not.
- From d=15mm onward, all three class pairs clear 1 sample under both velocity models.
- Using `c_exact` instead of `c0` changes individual gaps by a few percent but does **not** change which combinations pass/fail the 1-sample threshold at any tested distance.

Practical reading: short tracking distances (≤10mm) risk losing class
separability to sample-rate quantisation independent of which attenuation
model (weak vs. exact) is used — this is a TOF/sampling limitation, not
something the weak-damping question resolves either way.

## Task 4 — 20Hz sanity check (heuristic, not a classifier prediction)

Script: [`task4_20hz_sanity_check.py`](task4_20hz_sanity_check.py) · stdout: [`task4_stdout.log`](task4_stdout.log) · data: [`table4_20hz_sanity_check.csv`](table4_20hz_sanity_check.csv)

Using `alpha_exact` at 20Hz and 100Hz: `x20 = x100_ref * (alpha_exact_100/alpha_exact_20)`.

| class | reference | x100 (cm) | x20 heuristic (cm) |
|---|---|---|---|
| healthy | 8.48cm | 8.48 | 206.6 |
| healthy | 6.50cm | 6.50 | 158.4 |
| significant_fibrosis | 8.48cm | 8.48 | 210.0 |
| significant_fibrosis | 6.50cm | 6.50 | 161.0 |
| cirrhosis | 8.48cm | 8.48 | 202.7 |
| cirrhosis | 6.50cm | 6.50 | 155.4 |

**Sanity flag (not a code bug, a physical-plausibility flag):** the
20Hz-equivalent depths are 1.5–2.1 **meters** — far beyond any physically
meaningful tissue depth for this application. This is an expected
consequence of attenuation scaling roughly as ω² in the Kelvin-Voigt model
(20Hz attenuates ~20–25× less than 100Hz here), but it means the "20Hz
sanity check" as posed doesn't land on a physically interpretable depth —
it confirms the model is self-consistent at 20Hz, not that 20Hz is
practically equivalent to the certified 100Hz operating point at any real
tissue depth. Flagging for awareness; not resolving further per the protocol rules.

## Task 5 — Monte Carlo (5,000 samples), reusing `joint_uncertainty.py`'s exact ranges

Script: [`task5_monte_carlo.py`](task5_monte_carlo.py) · stdout: [`task5_stdout.log`](task5_stdout.log) · data: [`task5_monte_carlo_samples.csv`](task5_monte_carlo_samples.csv) (5,000 rows), [`task5_monte_carlo_summary.csv`](task5_monte_carlo_summary.csv)

Ranges reused verbatim by importing them from `Source/joint_uncertainty.py`
(not retyped): `VISCOSITY_RANGE=(0.5, 1.77)` Pa·s uniform, `DENSITY_RANGE=(975.2, 1144.8)` kg/m³ uniform (i.e. 1060±8%), `N_SAMPLES=5000`, seeded with `constants.SEED=42` — same seeding call `joint_uncertainty.py` itself uses. Class fixed to `healthy` (E=4.5kPa) to match `joint_uncertainty.py`'s own `HEALTHY_KPA` and the 8.48cm reference's nominal (healthy, 100Hz) framing.

Target quantity: for each MC draw of (η, ρ), the depth `x_eq` at which the
100Hz exact-model amplitude at (η,ρ) equals the *nominal* (η=0.5, ρ=1060)
exact-model amplitude at the reference depth 8.48cm.

**Result:**
- mean = **4.627 cm**
- SD (ddof=1) = **1.417 cm**
- 5th percentile = **3.138 cm**
- 95th percentile = **7.613 cm**

**Label: heuristic amplitude-equivalence, not a re-certified floor or a
classifier prediction.** The distribution sits well below both the 8.48cm
nominal reference and the 6.50cm safety gate for most of its mass (95th
percentile 7.61cm is still under 8.48cm) — this is expected and mechanical:
the viscosity range spans up to cirrhosis's η=1.77 Pa·s, and higher η under
uncertainty increases attenuation, which by definition pulls the
amplitude-matched depth down relative to the nominal η=0.5 baseline. This is
not a contradiction of any record number; it's a description of how sensitive
the amplitude-equivalence heuristic is to viscosity uncertainty within the
already-adopted MC sampling range.

## Cross-check with existing analysis

`Source/weak_damping_error_bound.py` (perturbative, leading-order, 250/500Hz
only) independently flags that the small-parameter expansion `x=omega*eta/G`
breaks down at 250/500Hz for these classes (`x` up to 1.39, not `<<1`). This
lane's exact-solution ratios (Task 1) are consistent with that flag: the
weak/exact divergence at 500Hz reaches 60–99%, i.e. large, exactly where the
perturbative script says its own bound stops being trustworthy. No
contradiction found between the two methods; they agree on where the
approximation is fragile.

## Contradictions with record numbers

**None found.** The 8.48cm/1.81cm and 6.50cm reference values used above
were independently re-derived from `Source/logs/degradation_aware_final_results.csv`
and `Source/constants.py` respectively and matched exactly (MATCH, not
MISMATCH) — see verification section above. Per protocol rule #6, if a
mismatch had appeared it would be reported here unresolved; none did.

## Open questions / recommendations (not resolved here, per scope)

1. Task 2's floor-shift bound is a physics-error bound, not a re-certified
   floor — actually re-running eval on the seed121 weights with the exact
   attenuation model (rather than the weak approximation) would be needed to
   turn this into a certified number.
2. Task 3's separability loss at short tracking distances (≤10mm) is
   independent of the weak-vs-exact question and may warrant its own
   investigation of minimum viable tracking distance.
3. Task 4's 20Hz heuristic lands on physically implausible depths (1.5–2m);
   worth deciding whether the 20Hz "sanity check" framing itself needs
   rethinking, or whether it's only meant to confirm internal model
   consistency (which it does).
4. Task 5's MC result is sensitive to the choice of using cirrhosis's η as
   the uncertainty range's upper bound (inherited from `joint_uncertainty.py`,
   not chosen fresh here) — if that range was intended to represent
   per-class uncertainty rather than a healthy-to-cirrhosis span, the MC
   result would differ substantially; flagging the inherited range's scope
   rather than resolving it.

## File list (all new, under `audit_fixes/lane1/`)

```
task1_exact_vs_weak.py, task1_stdout.log, table1_exact_vs_weak.csv
task2_floor_shift_bound.py, task2_stdout.log, task2_floor_shift_bound.csv
task3_arrival_time_sensitivity.py, task3_stdout.log, table3_arrival_time_sensitivity.csv
task4_20hz_sanity_check.py, task4_stdout.log, table4_20hz_sanity_check.csv
task5_monte_carlo.py, task5_stdout.log, task5_monte_carlo_samples.csv, task5_monte_carlo_summary.csv
REPORT.md (this file)
```

No existing file was modified. Nothing was committed (protocol rule #7).

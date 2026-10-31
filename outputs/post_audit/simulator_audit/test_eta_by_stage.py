"""Regression and correctness tests for the eta_by_stage flag in Source/attenuation_model.py.
(1) Flag off: output equals the pre-edit file (outputs/phase1/backups/attenuation_model.py.bak_*)
    bit-for-bit over a grid of kPa, frequencies and depths, with eta given and not given.
(2) Flag on: off-class kPa values get the stage eta, on-class values equal the old result,
    and a missing stage raises. Also writes the demonstration table data/eta_bug_demo.csv."""
import csv, glob, importlib.util, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(WT, "Source"))
import attenuation_model as new
bak = sorted(glob.glob(os.path.join(WT, "outputs/phase1/backups/attenuation_model.py.bak_*")))[0]
from importlib.machinery import SourceFileLoader
spec = importlib.util.spec_from_loader("attenuation_model_old", SourceFileLoader("attenuation_model_old", bak))
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

ok = True
kpas = [4.5, 7.5, 12.0, 3.0, 4.4999, 5.9, 7.51, 10.0, 11.99, 12.0001, 14.0, 15.0]
sig = np.sin(np.linspace(0, 20, 500))
n = 0
for k in kpas:
    for f in (20, 100, 250, 500):
        for eta in (None, 0.3, 1.77):
            a, b = old.attenuation_coefficient(k, f, eta=eta), new.attenuation_coefficient(k, f, eta=eta)
            ok &= (a == b); n += 1
        for d in (0.0, 0.045, 0.15):
            sa, aa = old.attenuate_signal(sig, k, d, f)
            sb, ab = new.attenuate_signal(sig, k, d, f)
            ok &= bool(np.array_equal(sa, sb)) and aa == ab; n += 1
print(f"flag off: {n} comparisons, identical = {ok}")

# flag on
cls = {"healthy": 4.5, "significant_fibrosis": 7.5, "cirrhosis": 12.0}
ok_on = True
rows = []
for stage, kc in cls.items():
    for k in (kc, kc * 0.8, kc * 0.9, kc + 1e-4, kc * 1.1, kc * 1.2):
        a_off = new.attenuation_coefficient(k, 100)
        a_on = new.attenuation_coefficient(k, 100, eta_by_stage=True, stage=stage)
        expect = new.attenuation_coefficient(k, 100, eta=new.LIVER_VISCOSITY_BY_CLASS[stage])
        ok_on &= (a_on == expect)
        if k == kc:
            ok_on &= (a_on == a_off)
        rows.append(dict(stage=stage, kpa=round(k, 4), eta_off=old._eta_for_kpa(k), eta_on=new.LIVER_VISCOSITY_BY_CLASS[stage],
                         alpha_off_per_m=a_off, alpha_on_per_m=a_on, ratio_on_over_off=a_on / a_off))
try:
    new.attenuation_coefficient(10.0, 100, eta_by_stage=True)
    ok_on = False
except ValueError:
    pass
new.ETA_BY_STAGE = True
ok_on &= new.attenuation_coefficient(10.0, 100, stage="cirrhosis") == new.attenuation_coefficient(10.0, 100, eta=1.77)
new.ETA_BY_STAGE = False
print("flag on: correct for off-class kPa, equal to old at class kPa, raises without stage:", ok_on)
os.makedirs(os.path.join(WT, "outputs/phase1/data"), exist_ok=True)
with open(os.path.join(WT, "outputs/phase1/data/eta_bug_demo.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    if r["stage"] == "cirrhosis":
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
print("TEST", "PASS" if ok and ok_on else "FAIL")
sys.exit(0 if ok and ok_on else 1)

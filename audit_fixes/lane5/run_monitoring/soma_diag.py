import os
os.environ['OMP_NUM_THREADS'] = '2'
import re, sys, csv, time, inspect, traceback
import numpy as np

from pathlib import Path as _P; R = str(_P(__file__).resolve().parents[3])
L = R + '/audit_fixes/lane5/recall_floor_main_run/logs'
print('run at', time.strftime('%Y-%m-%d %H:%M:%S'))

cert = {}
with open(R + '/Source/logs/degradation_aware_final_results.csv') as f:
    for row in csv.DictReader(f):
        cert[int(row['seed'])] = float(row['naive_floor_cm'])

print('\n=== 1. CLEAN-MODEL FLOORS SO FAR vs CERTIFIED DEGRADATION-AWARE ===')
clean = {}
for p in (L + '/lane5_chain_run_nohup.log', L + '/lane5_chain_run.log'):
    if os.path.exists(p):
        for line in open(p, errors='ignore'):
            m = re.search(r'\[clean\] Seed (\d+) done in (\d+)s\. floor=([\d.]+)cm', line)
            if m:
                clean[int(m.group(1))] = float(m.group(3))
if not clean:
    print('no clean-sweep lines found')
else:
    print('seed  clean_floor  certified_floor')
    for s in sorted(clean):
        flag = '  <- at grid max (censored)' if clean[s] >= 14.99 else ''
        print('%4d  %10.2f  %14.2f%s' % (s, clean[s], cert.get(s, float('nan')), flag))
    v = np.array([clean[s] for s in sorted(clean)])
    c = np.array([cert[s] for s in sorted(clean) if s in cert])
    print('n=%d clean: mean=%.2f median=%.2f min=%.2f max=%.2f at_max=%d' % (len(v), v.mean(), np.median(v), v.min(), v.max(), (v >= 14.99).sum()))
    print('same seeds, certified: mean=%.2f median=%.2f' % (c.mean(), np.median(c)))

print('\n=== 2. PER-CLASS BALANCED-ACCURACY FLOORS SO FAR ===')
pc = {}
pp = L + '/perclass_balacc_nohup.log'
if os.path.exists(pp):
    for line in open(pp, errors='ignore'):
        m = re.search(r'Seed (\d+) done in (\d+)s\. balacc floors: cirrhosis=([\d.]+), healthy=([\d.]+), significant_fibrosis=([\d.]+)', line)
        if m:
            pc[int(m.group(1))] = tuple(float(m.group(i)) for i in (3, 4, 5))
print('seed  cirrhosis  healthy  fibrosis')
for s in sorted(pc):
    print('%4d  %9.1f  %7.1f  %8.1f' % ((s,) + pc[s]))
for j, nm in enumerate(['cirrhosis', 'healthy', 'fibrosis']):
    vals = [pc[s][j] for s in pc]
    if vals:
        print('%s: n=%d exactly0=%d exactly15=%d other=%d' % (nm, len(vals), sum(x == 0.0 for x in vals), sum(x >= 14.99 for x in vals), sum(0.0 < x < 14.99 for x in vals)))

print('\n=== 3. RECENT OUTPUT FILES (last 12 h, no weights) ===')
now = time.time()
for root, dirs, files in os.walk(R + '/audit_fixes/lane5'):
    if 'seed_weights' in root or 'quarantine' in root or '__pycache__' in root:
        continue
    for fn in files:
        fp = os.path.join(root, fn)
        if now - os.path.getmtime(fp) < 12 * 3600 and not fn.endswith('.pt'):
            print('%9d  %s' % (os.path.getsize(fp), fp.replace(R + '/', '')))

print('\n=== 4. FIBROSIS-PERSISTENCE DIAGNOSTIC (main-run weights) ===')
try:
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, R + '/Source')
    import degradation_aware_final_sweep as das
    from cnn_classifier import SimpleCNN
    from physics_model import stiffness_classes
    sig = inspect.signature(das.generate_degraded_windows)
    print('generate_degraded_windows', sig)
    print('SimpleCNN', inspect.signature(SimpleCNN))
    names = sorted(stiffness_classes)
    print('class order (alphabetical):', names)
    FREQ = getattr(das, 'FREQ', 100)
    pnames = list(sig.parameters)
    use_cm = any('cm' in q for q in pnames[:3])
    print('depth argument treated as', 'cm' if use_cm else 'metres', '| params:', pnames)

    def make_model():
        for ctor in (lambda: SimpleCNN(), lambda: SimpleCNN(3), lambda: SimpleCNN(num_classes=3)):
            try:
                return ctor()
            except Exception:
                continue
        raise RuntimeError('cannot construct SimpleCNN')

    def load(path):
        m = make_model()
        sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        m.load_state_dict(sd)
        m.eval()
        return m

    def predict(m, X):
        t = torch.tensor(np.asarray(X), dtype=torch.float32)
        last = None
        with torch.no_grad():
            for shp in (lambda z: z.unsqueeze(1), lambda z: z):
                try:
                    return m(shp(t)).argmax(1).numpy()
                except Exception as e:
                    last = e
        raise last

    def conf(y, p):
        C = np.zeros((3, 3), int)
        for a, b in zip(y, p):
            C[a, b] += 1
        return C

    def balacc(C, c):
        tp = C[c, c]
        fn = C[c].sum() - tp
        fp = C[:, c].sum() - tp
        tn = C.sum() - tp - fn - fp
        return 0.5 * (tp / (tp + fn) + tn / (tn + fp))

    def build_test(depth_cm, n_per_class, seed):
        Xs, ys = [], []
        for ci, nm in enumerate(names):
            np.random.seed(seed + 7 * ci)
            d = depth_cm if use_cm else depth_cm / 100.0
            w = das.generate_degraded_windows(stiffness_classes[nm], d, FREQ, n_per_class)
            if isinstance(w, tuple):
                w = w[0]
            w = np.asarray(w)
            Xs.append(w)
            ys += [ci] * len(w)
        return np.concatenate(Xs), np.array(ys)

    depths = [0.0, 4.0, 8.181818, 12.272727, 15.0]
    for seed in (106, 107):
        wp = R + '/audit_fixes/lane5/recall_floor_main_run/seed_weights/degradation_aware_seed%d.pt' % seed
        if not os.path.exists(wp):
            print('seed', seed, 'weights missing')
            continue
        model = load(wp)
        ctrls = []
        for i in range(40):
            cp = R + '/audit_fixes/lane5/recall_floor_main_run/seed_weights/controls_seed%d/control%03d.pt' % (seed, i)
            if os.path.exists(cp):
                ctrls.append(load(cp))
        print('\n--- seed %d: real model + %d controls ---' % (seed, len(ctrls)))
        for dcm in depths:
            X, y = build_test(dcm, 300, 1000 * seed + int(dcm * 10))
            C = conf(y, predict(model, X))
            acc = np.trace(C) / C.sum()
            rec = [C[i, i] / C[i].sum() for i in range(3)]
            print('depth %.2f cm: acc=%.3f recall(cirr,heal,fib)=%.3f,%.3f,%.3f' % ((dcm, acc) + tuple(rec)))
            print('  confusion (rows true cirr,heal,fib; cols predicted):', C.tolist())
            cconf = [conf(y, predict(m, X)) for m in ctrls]
            for ci, nm in enumerate(names):
                cb = [balacc(cc, ci) for cc in cconf]
                print('  %s balacc real=%.3f controls min/median/max=%.3f/%.3f/%.3f' % (nm, balacc(C, ci), min(cb), float(np.median(cb)), max(cb)))
            if dcm == 0.0 and acc < 0.5:
                print('  WARNING: low accuracy at 0 cm; label order or input shape may be wrong, ignore this section')
except Exception:
    traceback.print_exc()
print('\nDONE')

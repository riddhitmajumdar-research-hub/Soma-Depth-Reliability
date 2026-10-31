"""Shared floor rule for Stage 2C/2D analysis (same functions as stage2b_results.py / p3_analyze.py: bh_int, new_floor; integer BH at q = 0.05, B = 719)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_grid import GRID
P3 = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); G = np.array(GRID); M = len(G)
def bh_int(cnt, B):
    cnt = np.asarray(cnt, np.int64); o = np.argsort(cnt, kind="stable"); ok = [(int(cnt[o[k-1]]) + 1) * M * 20 <= k * (B + 1) for k in range(1, M + 1)]
    kmax = max([k for k, v in enumerate(ok, 1) if v], default=0); sig = np.zeros(M, bool); sig[o[:kmax]] = True; return sig
def new_floor(sig):
    for i in range(M - 1):
        if not sig[i] and not sig[i + 1]: return (0.0, False) if i == 0 else (float(G[i - 1]), False)
    return float(G[-1]), True
def floor_from_file(path, epochs=200):
    z = np.load(path); assert int(z["B"]) == 719 and int(z["epochs"]) == epochs, path
    return new_floor(bh_int((z["control_counts"].astype(np.int64) >= z["real_counts"].astype(np.int64)[None, :]).sum(0), 719))
def sign_test(d):
    """Exact two-sided binomial sign test, zero differences dropped. Returns (n_up, n_down, n_zero, n, p)."""
    from scipy.stats import binomtest
    d = np.asarray(d, float); up = int((d > 0).sum()); dn = int((d < 0).sum()); z = int((d == 0).sum()); n = up + dn
    return up, dn, z, n, (float(binomtest(up, n, 0.5).pvalue) if n else float("nan"))

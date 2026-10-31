"""Phase 3 permutation machinery.

A control model is the same architecture and training protocol as the real model, trained on the same
training windows with the class labels randomly permuted (one permutation per control). The statistic at a
depth is the accuracy of a model on that depth's evaluation cell (300 windows, 100 per class). The
p-value at a depth is (number of controls with accuracy >= the real model's accuracy + 1) / (B + 1).

Controls are trained in Python threads (torch intra-op threads = 1 per process). Every random choice comes
from a stable-hash stream: crc32 of "<protocol>|<family>|<seed>|<draw>|<what>|<index>". Dropout inside a
control uses a private torch.Generator, so results do not depend on thread scheduling.
Training protocol is that of the certified trainers: SimpleCNN 60 epochs, ResNet1D 150 epochs (no retry for
controls), Adam lr 1e-3, full batch, dropout 0.3; RandomForest 300 trees with sklearn defaults.
"""
import os, sys, threading, zlib
import numpy as np
import torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p3_common import *  # noqa
from cnn_classifier import SimpleCNN
from resnet_classifier import ResNet1D

PROTOCOL = "p3_protocol_v1"
_init_lock = threading.Lock()


class GenDropout(nn.Module):
    def __init__(self, p, gen):
        super().__init__(); self.p = p; self.gen = gen
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape, generator=self.gen) >= self.p).to(x.dtype)
        return x * keep / (1.0 - self.p)


def stable_seed(*parts):
    return zlib.crc32("|".join(str(p) for p in (PROTOCOL,) + parts).encode())


def train_control_torch(kind, X_t, y, seed_init, seed_drop, dropout_p=0.3):
    """y: int labels (already permuted). Returns eval-mode model."""
    with _init_lock:                       # default init draws from the global generator; serialise it
        torch.manual_seed(seed_init)
        model = SimpleCNN(n_classes=3) if kind == "cnn" else ResNet1D(n_classes=3)
    model.dropout = GenDropout(dropout_p, torch.Generator().manual_seed(seed_drop))
    y_t = torch.tensor(y, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(60 if kind == "cnn" else 150):
        opt.zero_grad(); crit(model(X_t), y_t).backward(); opt.step()
    model.eval()
    return model


def control_counts_torch(kind, X_t, y_perm, cells_t, cells_y, s_init, s_drop):
    m = train_control_torch(kind, X_t, y_perm, s_init, s_drop)
    out = np.zeros(len(cells_t), dtype=np.uint16)
    with torch.no_grad():
        for i, (xc, yc) in enumerate(zip(cells_t, cells_y)):
            out[i] = int((m(xc).argmax(1).numpy() == yc).sum())
    return out


def control_counts_rf(Xtr, y_perm, cells_x, cells_y, rstate, n_jobs=1):
    from sklearn.ensemble import RandomForestClassifier
    m = RandomForestClassifier(n_estimators=300, n_jobs=n_jobs, random_state=rstate).fit(Xtr, y_perm)
    return np.array([int((m.predict(x) == y).sum()) for x, y in zip(cells_x, cells_y)], dtype=np.uint16)


def run_controls(family, seed, draw, B, Xtr, ytr, cells_x, cells_y, workers):
    """Returns uint16 counts [B, n_depths]."""
    rng = np.random.RandomState(stable_seed(family, seed, draw, "perm"))
    perms = [rng.permutation(ytr) for _ in range(B)]
    if family == "rf":
        jobs = [(Xtr, perms[j], cells_x, cells_y, stable_seed(family, seed, draw, "rf", j) % (2 ** 31 - 1)) for j in range(B)]
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda a: control_counts_rf(*a), jobs))
    else:
        X_t = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1)
        cells_t = [torch.tensor(x, dtype=torch.float32).unsqueeze(1) for x in cells_x]
        jobs = [(family, X_t, perms[j], cells_t, cells_y, stable_seed(family, seed, draw, "init", j), stable_seed(family, seed, draw, "drop", j)) for j in range(B)]
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(lambda a: control_counts_torch(*a), jobs))
    return np.stack(res)

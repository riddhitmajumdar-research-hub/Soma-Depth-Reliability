"""Phase 4.2 per PROTOCOL_PHASE4.md. Writes data4/attr_profiles.csv (columns: net, method, depth_cm, group, seed, t0..t99).
net = trained | init (same init seed, untrained) | input (mean |x| normalised; method 'abs_input')."""
import csv, os, sys
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__)); P4 = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(P4, "..", "phase3", "code"))
from p3_common import *  # noqa
from p3_grid import GRID
from p3_run import eval_cells
from cnn_classifier import SimpleCNN
DEPTHS = [0.0, 4.0, 6.5, 8.0, 10.0]; IDX = {d: list(GRID).index(d) for d in DEPTHS}
def gradcam(m, x, tgt):
    x = x.clone().requires_grad_(True)
    a = m.relu(m.conv2(m.pool(m.relu(m.conv1(x))))); a.retain_grad()
    out = m.fc2(m.relu(m.fc1(m.pool(a).view(len(x), -1)))); out.gather(1, tgt[:, None]).sum().backward()
    cam = F.relu((a.grad.mean(2, keepdim=True) * a).sum(1, keepdim=True)); return F.interpolate(cam, size=100, mode="linear", align_corners=False)[:, 0].detach().numpy()
def ig(m, x, tgt, steps=32):
    tot = torch.zeros_like(x)
    for k in range(1, steps + 1):
        xi = (x * k / steps).clone().requires_grad_(True); m(xi).gather(1, tgt[:, None]).sum().backward(); tot += xi.grad
    return (x * tot / steps).abs().detach().numpy()[:, 0]
w = csv.writer(open(os.path.join(P4, "data4", "attr_profiles.csv"), "w", newline="")); w.writerow(["net", "method", "depth_cm", "group", "seed"] + [f"t{i}" for i in range(100)])
def emit(net, meth, d, seed, A, y):
    A = A / (A.sum(1, keepdims=True) + 1e-12)
    w.writerow([net, meth, d, "all", seed, *np.round(A.mean(0), 6)])
    for k in range(3): w.writerow([net, meth, d, f"class{k}", seed, *np.round(A[y == k].mean(0), 6)])
for seed in range(101, 125):
    nets = {}
    m = SimpleCNN(n_classes=3); m.load_state_dict(torch.load(os.path.join(WORKTREE, f"outputs/phase2/checkpoints/cnn/cnn_p2_primary_v1_seed{seed}.pt"))); m.eval(); nets["trained"] = m
    torch.manual_seed(seed); np.random.seed(seed); m0 = SimpleCNN(n_classes=3); m0.eval(); nets["init"] = m0
    cx, cy = eval_cells(seed)
    for d in DEPTHS:
        x = torch.tensor(cx[IDX[d]], dtype=torch.float32).unsqueeze(1); y = cy[IDX[d]]
        emit("input", "abs_input", d, seed, np.abs(cx[IDX[d]]), y)
        for net, mm in nets.items():
            tgt = mm(x).argmax(1)
            emit(net, "gradcam", d, seed, gradcam(mm, x, tgt), y); emit(net, "ig", d, seed, ig(mm, x, tgt), y)
print("done")

# Probe of what retrained models predict on pure noise and at three depths. Imports the repo's own data and training code unchanged; writes only into this folder.
# Real model only (no shuffled controls). Seeds 101-108.
import os, sys, csv, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO + "/Source")
import numpy as np, torch
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from physics_model import stiffness_classes
import degradation_aware_final_sweep as sw
import degradation_aware_resnet_final_sweep as rs

OUT = HERE
SEEDS = list(range(101, 109))
NAMES = sorted(stiffness_classes.keys())  # cirrhosis, healthy, significant_fibrosis (LabelEncoder order)
N_PER_CLASS = 300
N_NOISE = 300
MEDIAN_FLOOR = {"rf": 12.272727272727272, "resnet": 10.909090909090908}  # from the saved 24-seed floors
which = sys.argv[1]

def depths_for(kind):
    d = [("6.50", 6.5), ("median_floor", MEDIAN_FLOOR[kind]), ("15.00", 15.0)]
    grid = list(sw.EVAL_DEPTHS_CM)
    for lab, v in d[1:]:
        assert any(abs(v - g) < 1e-9 for g in grid), (lab, v)  # all three are on the sweep's own grid
    return d

def train(kind, X, y, seed):
    if kind == "rf":
        # identical to lane8_harness.train_rf with lane8_config values (not imported: lane8_harness opens a log file in the repo)
        m = RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=seed); m.fit(X, y); return m, ""
    m, collapses, attempt = rs.train_resnet_with_retry(X, y)
    return m, f"collapses={collapses};escaped_on_attempt={attempt}"

def predict(kind, m, X):
    if kind == "rf": return m.predict(X)
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32).unsqueeze(1)).argmax(dim=1).numpy()

rows = []
t0 = time.time()
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    X_train, y_names = sw.build_degradation_aware_dataset()
    enc = LabelEncoder(); y = enc.fit_transform(y_names)
    assert list(enc.classes_) == NAMES
    model, note = train(which, X_train, y, seed)
    # pure-noise windows: same construction as the battery's condition 8 (cloud/battery wrappers.build_pure_noise_windows)
    rng = np.random.RandomState((seed * 999983 + 8) % (2**32 - 1))
    Xn = rng.normal(0, sw.SENSOR_NOISE_STD, size=(N_NOISE, 100))
    c = np.bincount(predict(which, model, Xn), minlength=3)
    rows.append(dict(classifier=which, seed=seed, condition="pure_noise", true_class="none", n=N_NOISE,
                     pred_cirrhosis=c[0], pred_healthy=c[1], pred_fibrosis=c[2], note=note))
    for lab, depth_cm in depths_for(which):
        for name in NAMES:
            w = np.array(sw.generate_degraded_windows(stiffness_classes[name], depth_cm / 100, sw.FREQ, N_PER_CLASS))
            c = np.bincount(predict(which, model, w), minlength=3)
            rows.append(dict(classifier=which, seed=seed, condition=f"depth_{lab}_{depth_cm:.4f}cm", true_class=name, n=N_PER_CLASS,
                             pred_cirrhosis=c[0], pred_healthy=c[1], pred_fibrosis=c[2], note=""))
    print(f"{which} seed {seed} done, {time.time()-t0:.0f}s {note}", flush=True)
    with open(f"{OUT}/raw_counts_{which}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

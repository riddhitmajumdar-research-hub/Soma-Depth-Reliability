"""Phase 2 trainer for cnn, resnet, rf, logreg. Resumable: a seed with a record row is skipped.
Usage: p2_run.py --family cnn --config p2_primary_v1 --seeds 101 102 ... [--threads N]
Each seed writes: dataset cache (read-only), checkpoint (read-only) + manifest, one record row.
Sanity evaluation only: accuracy at the clean cell and at 0, 6.5, 8, 10 cm. No floors, no p-values."""
import argparse, csv, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2_common import *  # noqa
from p2_features import lr_features, FEATURE_NAMES
import joblib

FIELDS = ["family", "config_id", "seed", "split_hash", "config_hash", "git_commit", "code_dirty", "epochs_or_trees",
          "train_acc", "val_acc", "acc_eval_clean", "acc_eval_d0", "acc_eval_d6.5", "acc_eval_d8", "acc_eval_d10",
          "train_wall_s", "total_wall_s", "predict_s_per_1500_windows", "data_build_s", "power_label",
          "checkpoint_file", "checkpoint_sha256", "reload_check", "collapse_count", "resnet_attempt"]


def torch_predict(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).unsqueeze(1))


def run_one(family, config_id, seed):
    import degradation_aware_final_sweep as sweep
    cfg, cfg_hash, _ = load_config(config_id)
    t_total = time.time(); p_start = power_state()
    t = time.time(); ds, shash, built = get_dataset(config_id, seed); t_data = time.time() - t
    Xtr, ytr = ds["train"]; Xva, yva = ds["val"]
    commit, dirty = git_commit()
    ck_dir = os.path.join(P2, "checkpoints", {"cnn": "cnn", "resnet": "resnet", "rf": "rf", "logreg": "logreg"}[family])
    ext = {"cnn": ".pt", "resnet": ".pt", "rf": ".joblib", "logreg": ".joblib"}[family]
    ck_path = os.path.join(ck_dir, f"{family}_{config_id}_seed{seed}{ext}")
    collapse, attempt = "", ""
    torch.manual_seed(seed); np.random.seed(seed)
    t = time.time()
    if family == "cnn":
        model = sweep.train_cnn(Xtr, ytr); epochs = 60
        pred = lambda X: torch_predict(model, X).argmax(1).numpy()
        raw = lambda X: torch_predict(model, X).numpy()
    elif family == "resnet":
        import degradation_aware_resnet_final_sweep as rs
        model, collapse, attempt = rs.train_resnet_with_retry(Xtr, ytr); epochs = 150
        pred = lambda X: torch_predict(model, X).argmax(1).numpy()
        raw = lambda X: torch_predict(model, X).numpy()
    elif family == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=seed).fit(Xtr, ytr); epochs = 300
        pred = lambda X: model.predict(X)
        raw = lambda X: model.predict_proba(X)
    else:
        from sklearn.preprocessing import RobustScaler
        from sklearn.linear_model import LogisticRegression
        sc = RobustScaler().fit(lr_features(Xtr))
        lr = LogisticRegression(max_iter=1000).fit(sc.transform(lr_features(Xtr)), ytr); epochs = 0
        model = {"scaler": sc, "model": lr, "features": FEATURE_NAMES}
        pred = lambda X: lr.predict(sc.transform(lr_features(X)))
        raw = lambda X: lr.predict_proba(sc.transform(lr_features(X)))
    t_train = time.time() - t

    fixed = np.concatenate([Xva[:64], ds["eval_clean"][0][:64]])
    before = raw(fixed)
    if os.path.exists(ck_path):
        raise RuntimeError(f"{ck_path} exists; not overwriting")
    if family in ("cnn", "resnet"):
        torch.save(model.state_dict(), ck_path)
    else:
        joblib.dump(model, ck_path)
    os.chmod(ck_path, 0o444)
    # reload check
    if family == "cnn":
        from cnn_classifier import SimpleCNN
        m2 = SimpleCNN(n_classes=3); m2.load_state_dict(torch.load(ck_path)); m2.eval()
        after = torch_predict(m2, fixed).numpy()
    elif family == "resnet":
        from resnet_classifier import ResNet1D
        m2 = ResNet1D(n_classes=3); m2.load_state_dict(torch.load(ck_path)); m2.eval()
        after = torch_predict(m2, fixed).numpy()
    else:
        m2 = joblib.load(ck_path)
        after = (m2.predict_proba(fixed) if family == "rf" else
                 m2["model"].predict_proba(m2["scaler"].transform(lr_features(fixed))))
    reload_ok = bool(np.array_equal(before, after)) and bool(np.array_equal(before.argmax(1), after.argmax(1)))

    rec = dict(family=family, config_id=config_id, seed=seed, split_hash=shash, config_hash=cfg_hash,
               git_commit=commit, code_dirty=dirty, epochs_or_trees=epochs,
               train_acc=acc(pred(Xtr), ytr), val_acc=acc(pred(Xva), yva),
               collapse_count=collapse, resnet_attempt=attempt)
    t = time.time()
    for nm in EVAL_NAMES:
        X, y = ds[nm]
        rec[f"acc_{nm}"] = acc(pred(X), y)
    rec["predict_s_per_1500_windows"] = time.time() - t
    rec.update(train_wall_s=t_train, total_wall_s=time.time() - t_total, data_build_s=t_data if built else 0.0,
               power_label=power_label(p_start, power_state()), checkpoint_file=os.path.basename(ck_path),
               checkpoint_sha256=sha256_file(ck_path), reload_check="exact_match" if reload_ok else "MISMATCH")
    return rec, ck_dir, ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rp = os.path.join(P2, "records", f"seed_records_{a.family}_{a.config}.csv")
    done = set()
    if os.path.exists(rp):
        done = {int(r["seed"]) for r in csv.DictReader(open(rp))}
    for s in a.seeds:
        if s in done:
            print(f"{a.family} {a.config} seed {s}: record exists, skipping", flush=True); continue
        rec, ck_dir, ext = run_one(a.family, a.config, s)
        new = not os.path.exists(rp)
        with open(rp, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new: w.writeheader()
            w.writerow(rec); f.flush(); os.fsync(f.fileno())
        write_manifest(ck_dir, ext if ext != ".joblib" else ".joblib")
        print(f"{a.family} {a.config} seed {s}: train {rec['train_acc']:.3f} val {rec['val_acc']:.3f} "
              f"clean {rec['acc_eval_clean']:.3f} d0 {rec['acc_eval_d0']:.3f} d6.5 {rec['acc_eval_d6.5']:.3f} "
              f"d8 {rec['acc_eval_d8']:.3f} d10 {rec['acc_eval_d10']:.3f} | fit {rec['train_wall_s']:.1f}s "
              f"total {rec['total_wall_s']:.1f}s | {rec['power_label']} | reload {rec['reload_check']}", flush=True)


if __name__ == "__main__":
    main()

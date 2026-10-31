"""Nadeau-Bengio correction on the saved Lane 4 v2 per-fold accuracies.

Uses nadeau_bengio_test() from audit_fixes/lane2/analysis.py unchanged. That file
runs its whole analysis and rewrites its outputs on import, so the function is
pulled out of the source with ast instead of importing the module.
Nothing is retrained. Run from the repo root: python3 audit_fixes/lane4/nb_v2.py
"""
import ast, csv, os
import numpy as np
from scipy import stats

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src = open(os.path.join(R, "audit_fixes/lane2/analysis.py")).read()
fn = [n for n in ast.parse(src).body
      if isinstance(n, ast.FunctionDef) and n.name == "nadeau_bengio_test"][0]
ns = {"np": np, "stats": stats}
exec(compile(ast.Module([fn], []), "lane2_nb", "exec"), ns)
nb = ns["nadeau_bengio_test"]

load = lambda p: list(csv.DictReader(open(os.path.join(R, p))))
a = load("Source/logs/log11_repeated_grouped_cv_raw.csv")
b = load("audit_fixes/lane4/lane4_v2_repeated_grouped_cv_raw.csv")
assert [(r["repetition"], r["fold"]) for r in a] == [(r["repetition"], r["fold"]) for r in b]
col = lambda rows, k: np.array([float(r[k]) for r in rows])
M = {"baseline": col(a, "baseline_acc"), "cnn": col(a, "cnn_acc"), "resnet1d": col(a, "resnet1d_acc"),
     "svm": col(b, "svm_acc"), "rf": col(b, "rf_acc"), "gru": col(b, "gru_acc")}
for k, v in M.items():
    print(f"{k}: mean {v.mean()*100:.2f}%")
for x, y in [("svm", "baseline"), ("rf", "baseline"), ("svm", "cnn"), ("rf", "cnn"),
             ("svm", "resnet1d"), ("rf", "resnet1d"), ("rf", "svm"), ("gru", "baseline")]:
    r = nb(M[x] - M[y])
    print(f"{x} vs {y}: diff={r['mean_diff']*100:+.2f}pp t={r['t_stat']:.3f} df={r['df']} "
          f"p={r['p_value']:.6g} 95% CI=[{r['ci_95_lo']*100:.2f}, {r['ci_95_hi']*100:.2f}]")

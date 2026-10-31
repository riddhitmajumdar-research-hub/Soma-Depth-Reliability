# SCRATCH ORIGIN: copied from ~/soma_audit_scratch/trackB_local/common.py, with only the repo-root path changed.
# Purpose: helper module from the same scratch folder (not imported by run_variants.py; used by its sibling scripts).
# NOT RE-EXECUTED IN THE REPO: this copy has never been run from this location; the saved results came from the scratch folder run.
# The repo-root path was generalised to a repo-relative root derived from this file's location; other paths are as copied.
import ast, sys, numpy as np, torch, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
from pathlib import Path as _P; SRC=(str(_P(__file__).resolve().parents[1]) + "/Source")
sys.path.insert(0,SRC)
from cnn_classifier import SimpleCNN
# extract train_cnn / evaluate and constants from the original script via ast (nothing executed)
_tree=ast.parse(open(f"{SRC}/ets_montreal_path2_retrain.py").read())
_ns=dict(torch=torch,nn=nn,np=np,SimpleCNN=SimpleCNN,Path=Path,loadmat=loadmat)
for n in _tree.body:
    if isinstance(n,ast.FunctionDef) and n.name in("train_cnn","evaluate","extract_windows"): pass
    if isinstance(n,ast.Assign) and getattr(n.targets[0],"id","") in("DATA_ROOT","TRAIN_SEQUENCES","TEST_SEQUENCES","INCLUSION_ROWS","BACKGROUND_ROWS","N_EPOCHS"):
        exec(compile(ast.Module([n],[]),"x","exec"),_ns)
_fn=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in("train_cnn",)]
exec(compile(ast.Module(_fn,[]),"x","exec"),_ns)
train_cnn=_ns["train_cnn"]; DATA_ROOT=_ns["DATA_ROOT"]; TRAIN=_ns["TRAIN_SEQUENCES"]; TEST=_ns["TEST_SEQUENCES"]
IR=_ns["INCLUSION_ROWS"]; BR=_ns["BACKGROUND_ROWS"]
def zs(w): return (w-w.mean(-1,keepdims=True))/(w.std(-1,keepdims=True)+1e-8)
CACHE="cache.npz"
def build_cache():
    out={}
    for s in TRAIN+TEST:
        inc=[];bg=[]
        for p in sorted((DATA_ROOT/s/"RF").glob("RF*.mat")):
            rf=loadmat(str(p))["RF"].astype(np.float32)
            inc.append(rf[IR[0]:IR[1]].T); bg.append(rf[BR[0]:BR[1]].T)   # (256,100)
        out[s+"_inc"]=np.stack(inc); out[s+"_bg"]=np.stack(bg)            # (files,256,100)
    np.savez(CACHE,**out)
def load(): return np.load(CACHE)
def recall_acc(model,X,y):
    with torch.no_grad(): p=model(torch.tensor(X,dtype=torch.float32).unsqueeze(1)).argmax(1).numpy()
    return (p==y).mean(), *(float((p[y==c]==c).mean()) for c in (0,1))

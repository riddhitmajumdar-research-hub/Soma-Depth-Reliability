# SCRATCH ORIGIN: copied from ~/soma_audit_scratch/trackB_local/run_variants.py, with only the repo-root path changed.
# Purpose: generator of outputs/external_validation_controls/ets_montreal_controls_results.json (Fig. 13).
# NOT RE-EXECUTED IN THE REPO: this copy has never been run from this location; the saved results came from the scratch folder run.
# The repo-root path was generalised to a repo-relative root derived from this file's location; other paths are as copied.
import ast, sys, math, json, time
import numpy as np, torch, torch.nn as nn
from scipy.io import loadmat
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/"Source"))
from cnn_classifier import SimpleCNN
torch.set_num_threads(4)
# ---- extract (not run) train_cnn/evaluate/constants from the original script via ast
src=(REPO/"Source/ets_montreal_path2_retrain.py").read_text(); tree=ast.parse(src)
keep=[n for n in tree.body if (isinstance(n,ast.FunctionDef) and n.name in("train_cnn","evaluate"))
      or (isinstance(n,ast.Assign) and any(getattr(t,'id','') in("DATA_ROOT","INCLUSION_ROWS","BACKGROUND_ROWS","N_EPOCHS") for t in n.targets))]
ns=dict(np=np,torch=torch,nn=nn,SimpleCNN=SimpleCNN,Path=Path)
exec(compile(ast.Module(body=keep,type_ignores=[]),"extracted","exec"),ns)
DATA_ROOT,INC,BG,train_cnn,evaluate=ns["DATA_ROOT"],ns["INCLUSION_ROWS"],ns["BACKGROUND_ROWS"],ns["train_cnn"],ns["evaluate"]
print("extracted:",DATA_ROOT,INC,BG,"N_EPOCHS",ns["N_EPOCHS"])
def z(w): return (w-w.mean(1,keepdims=True))/(w.std(1,keepdims=True)+1e-8)   # same per-window z-score as original
SEQ=[f"Acqui{i}" for i in range(1,7)]
# raw slices per sequence: (frames,256,100)
RAW={}
for s in SEQ:
    a=[];b=[]
    for f in sorted((DATA_ROOT/s/"RF").glob("RF*.mat")):
        rf=loadmat(str(f))["RF"].astype(np.float32)
        a.append(rf[INC[0]:INC[1]].T.copy()); b.append(rf[BG[0]:BG[1]].T.copy())
    RAW[s]=(np.stack(a),np.stack(b)); print(s,"frames",len(a),flush=True)
# ---- column definitions
X0,W=169,303; PX={"Acqui1":(255,387),"Acqui2":(246,393),"Acqui3":(242,385),"Acqui4":(250,389),"Acqui5":(258,393),"Acqui6":(258,393)}
def col(x): return math.floor((x-X0)/W*256)
def defM(s):
    c0,c1=col(PX[s][0]),col(PX[s][1]); return list(range(c0+10,c1-10+1)), [c for c in range(256) if c<c0-10 or c>c1+10],(c0,c1)
COLS={}
print("\nPRIMARY (M, image-measured; ASSUMES 256 A-lines span the displayed 303-px image width)")
for s in SEQ:
    i,n,(c0,c1)=defM(s); COLS[s]=(i,n)
    print(f"{s}: measured span cols {c0}-{c1}; incl (shrunk10) {i[0]}-{i[-1]} n={len(i)}; non-incl (widened10) 0-{c0-11} and {c1+11}-255 n={len(n)}")
    if len(i)<15: print("STOP: <15 inclusion columns"); sys.exit(1)
HA=(list(range(113,144)),list(range(0,51))+list(range(206,256)))
HB=(list(range(71,186)),list(range(0,51))+list(range(206,256)))   # H-b span 60.6-195.4 shrunk 10 -> 71-185; outside widened 10
print("H-a: incl 113-143, non 0-50,206-255 | H-b (my reading of Step 3): incl 71-185, non 0-50,206-255")
def cols_of(mode,s): return COLS[s] if mode=="M" else (HA if mode=="Ha" else HB)
# ---- energy report (raw, before z-score)
print("\nENERGY (mean of squared raw RF per window, all frames, per-sequence column definitions M)")
E={"a":[],"b":[],"c":[]}
for s in SEQ:
    i,n=COLS[s]; A,B=RAW[s]
    E["a"].append((A[:,i]**2).mean()); E["b"].append((A[:,n]**2).mean()); E["c"].append((B[:,n]**2).mean())
    print(f" {s}: a={E['a'][-1]:.4g} b={E['b'][-1]:.4g} c={E['c'][-1]:.4g} a/b={E['a'][-1]/E['b'][-1]:.2f} b/c={E['b'][-1]/E['c'][-1]:.2f}")
ea,eb,ec=[np.mean(E[k]) for k in "abc"]
print(f" MEAN over seqs: a={ea:.4g} b={eb:.4g} c={ec:.4g} a/b={ea/eb:.2f} b/c={eb/ec:.2f}",flush=True)
# ---- variant window builders: return (X,y) unbalanced
def build(variant,mode,seqs):
    X=[];y=[]
    for s in seqs:
        A,B=RAW[s]; i,n=cols_of(mode,s)
        if variant=="V0":
            X+= [z(A.reshape(-1,100)),z(B.reshape(-1,100))]; y+=[1]*(A.shape[0]*256)+[0]*(B.shape[0]*256)
        elif variant=="V1":
            X+=[z(A[:,i].reshape(-1,100)),z(A[:,n].reshape(-1,100))]; y+=[1]*(A.shape[0]*len(i))+[0]*(A.shape[0]*len(n))
        else:
            X+=[z(A[:,n].reshape(-1,100)),z(B[:,n].reshape(-1,100))]; y+=[1]*(A.shape[0]*len(n))+[0]*(B.shape[0]*len(n))
    return np.concatenate(X).astype(np.float32),np.array(y,dtype=np.int64)
def balance(X,y,rng):
    i1=np.where(y==1)[0];i0=np.where(y==0)[0];m=min(len(i1),len(i0))
    idx=np.concatenate([rng.choice(i1,m,replace=False),rng.choice(i0,m,replace=False)])
    return X[idx],y[idx],len(i1),len(i0),m
RES=[]
def run(variant,mode):
    Xtr0,ytr0=build(variant,mode,SEQ[:5]); Xte0,yte0=build(variant,mode,["Acqui6"])
    accs=[];r1=[];r0=[]
    for seed in (1,2,3):
        rng=np.random.default_rng(seed)
        Xtr,ytr,a1,a0,m=balance(Xtr0,ytr0,rng); Xte,yte,b1,b0,mt=balance(Xte0,yte0,rng)
        if seed==1: print(f"  windows before balancing: train {a1}/{a0} (label1/label0) test {b1}/{b0}; after: train {m}+{m}, test {mt}+{mt}")
        torch.manual_seed(seed); np.random.seed(seed)
        model=train_cnn(Xtr,ytr); Xt=torch.tensor(Xte).unsqueeze(1)
        with torch.no_grad(): p=model(Xt).argmax(1).numpy()
        acc=(p==yte).mean(); assert abs(acc-evaluate(model,Xt,yte))<1e-9
        rc1=(p[yte==1]==1).mean(); rc0=(p[yte==0]==0).mean()
        flag=" COLLAPSED (a class never predicted)" if (p==1).sum()==0 or (p==0).sum()==0 else ""
        print(f"  seed {seed}: acc={acc:.4f} recall1={rc1:.4f} recall0={rc0:.4f}{flag}",flush=True)
        accs.append(acc);r1.append(rc1);r0.append(rc0)
    r=dict(variant=variant,mode=mode,acc_mean=float(np.mean(accs)),acc_sd=float(np.std(accs,ddof=1)),rec1=float(np.mean(r1)),rec0=float(np.mean(r0)),accs=[float(a) for a in accs])
    print(f"  => {variant}/{mode}: acc {r['acc_mean']:.4f} +- {r['acc_sd']:.4f} (SD, n=3), recall1 {r['rec1']:.4f}, recall0 {r['rec0']:.4f}\n",flush=True); RES.append(r)
t0=time.time()
for v,m in [("V0","-"),("V1","M"),("V2","M"),("V1","Ha"),("V1","Hb"),("V2","Ha"),("V2","Hb")]:
    print(f"=== {v} [{'primary M' if m=='M' else ('n/a' if m=='-' else 'secondary, hypothesis-dependent '+m)}]",flush=True); run(v,m)
json.dump(RES,open(Path.home()/"soma_audit_scratch/trackB_local/results.json","w"),indent=1)
print("elapsed",time.time()-t0)

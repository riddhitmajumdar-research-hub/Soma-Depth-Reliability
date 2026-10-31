import os, glob, csv, re
import numpy as np
from scipy import stats
from pathlib import Path as _P; R=str(_P(__file__).resolve().parents[4])
def bh(p, q=0.05):
    p=np.asarray(p); n=len(p); o=np.argsort(p); ranked=p[o]
    thr=q*np.arange(1,n+1)/n
    ok=np.where(ranked<=thr)[0]
    sig=np.zeros(n,bool)
    if len(ok): sig[o[:ok.max()+1]]=True
    return sig
def floor_from(depths, sig):
    for d,s in zip(depths,sig):
        if not s: return d, False
    return depths[-1], True
def load(path):
    rows=list(csv.DictReader(open(path)))
    cols=list(rows[0].keys())
    ac=[c for c in cols if 'acc' in c.lower()][0]
    sg=[c for c in cols if 'sig' in c.lower()][0]
    print('   using columns:',ac,sg) if not hasattr(load,'p') else None
    load.p=1
    rows.sort(key=lambda r: float(r['depth_cm']))
    return (np.array([float(r['depth_cm']) for r in rows]), np.array([float(r[ac]) for r in rows]),
            np.array([r[sg].strip()=='True' for r in rows]))
def report(tag, files, N=300):
    print('\n=====',tag,'=====')
    print('seed  perm_floor  binom300_floor  binom75_floor  | acc at 8.18/12.27/15.0')
    out=[]
    for f in files:
        s=int(re.search(r'seed(\d+)',f).group(1))
        d,a,sigp=load(f)
        pf,pc=floor_from(d,sigp)
        res=[]
        for n in (N,75):
            k=np.round(a*n).astype(int)
            p=stats.binom.sf(k-1,n,1/3)
            res.append(floor_from(d,bh(p)))
        def at(x):
            i=np.argmin(abs(d-x)); return a[i]
        print('%4d  %6.2f%s  %10.2f%s  %12.2f%s  |  %.3f / %.3f / %.3f'%(s,pf,'*' if pc else ' ',res[0][0],'*' if res[0][1] else ' ',res[1][0],'*' if res[1][1] else ' ',at(8.18),at(12.27),at(15.0)))
        out.append((pf,res[0][0],res[1][0]))
    if out:
        o=np.array(out); print('median floors: perm=%.2f binom300=%.2f binom75=%.2f   (* = no non-significant depth)'%tuple(np.median(o,axis=0)))
cf=sorted(glob.glob(R+'/audit_fixes/lane5/clean_trained_sweep/clean_sweep/logs/lane5_clean_aggregate_depth_detail_seed*.csv'))
mf=sorted(glob.glob(R+'/audit_fixes/lane5/recall_floor_main_run/logs/lane5_a*seed*.csv'))
print('found clean files:',len(cf),' main-run files:',len(mf))
report('CLEAN-TRAINED MODEL (matched protocol)',cf)
report('DEGRADATION-AWARE CNN, replicate run',mf)
print('\n===== CONTROL COLLAPSE AT DEPTH (clean sweep confusion matrices) =====')
cm=R+'/audit_fixes/lane5/clean_trained_sweep/clean_sweep/confusion_matrices/'
for s in (101,102,103,107):
    for dep in ('8.18','12.27','15.00'):
        rp=cm+'seed%d_depth%s_real_confmat.npy'%(s,dep); cp=cm+'seed%d_depth%s_controls_confmat.npy'%(s,dep)
        if not (os.path.exists(rp) and os.path.exists(cp)): continue
        r=np.load(rp).astype(float); c=np.load(cp).astype(float)
        share=c.sum(axis=1)
        if np.allclose(share,share[:,:1]): share=c.sum(axis=2)
        share=share/share.sum(axis=1,keepdims=True)
        rs=r.sum(axis=0)
        if np.allclose(rs,rs[0]): rs=r.sum(axis=1)
        rs=rs/rs.sum()
        near=int((share.max(axis=1)>=0.9).sum())
        print('seed %d depth %5s: real acc=%.3f pred-share=%.2f,%.2f,%.2f | controls near-constant (>=90%% one class): %d/200'%(s,dep,np.trace(r)/r.sum(),rs[0],rs[1],rs[2],near))
print('\nDONE')

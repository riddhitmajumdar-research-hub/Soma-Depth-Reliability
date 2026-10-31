import glob, re, os
import numpy as np
from scipy import stats
from pathlib import Path as _P; R=(str(_P(__file__).resolve().parents[4]) + "/audit_fixes/lane5/balanced_accuracy_retry/perclass_balacc/confusion_matrices/")
def bh(p,q=0.05):
    p=np.asarray(p); n=len(p); o=np.argsort(p); r=p[o]; ok=np.where(r<=q*np.arange(1,n+1)/n)[0]
    s=np.zeros(n,bool)
    if len(ok): s[o[:ok.max()+1]]=True
    return s
def first_fail(d,sig):
    for x,s in zip(d,sig):
        if not s: return x,False
    return d[-1],True
data={}
for f in glob.glob(R+'seed*_depth*_real_confmat.npy'):
    m=re.search(r'seed(\d+)_depth([\d.]+)_real',f); s=int(m.group(1)); d=float(m.group(2))
    cf=f.replace('_real_','_controls_')
    real=np.load(f).astype(float)
    ctrl=np.load(cf).astype(float) if os.path.exists(cf) else None
    data.setdefault(s,{})[d]=(real,ctrl)
seeds=sorted(data); print('seeds found:',len(seeds),'| example depths:',sorted(data[seeds[0]])[:6],'...')
r0=data[seeds[0]][sorted(data[seeds[0]])[0]][0]; print('confusion matrix sum (windows per test set):',r0.sum(),'| row sums:',r0.sum(axis=1).tolist())
print('assumed class order: cirrhosis, healthy, significant_fibrosis (alphabetical)')
names=['cirrhosis','healthy','fibrosis']
print('\n=== A. CERTIFIED-CNN REPLICATE: floor by three nulls (on the per-class-eval test sets) ===')
print('seed  perm_floor  binom_n  binom_n/4')
allf=[]
for s in seeds:
    ds=sorted(data[s]); acc=np.array([np.trace(data[s][d][0])/data[s][d][0].sum() for d in ds]); N=int(data[s][ds[0]][0].sum())
    pp=[]
    for d in ds:
        real,ctrl=data[s][d]
        if ctrl is None: pp.append(np.nan); continue
        ca=np.array([np.trace(c)/c.sum() for c in ctrl]); pp.append((1+(ca>=np.trace(real)/real.sum()).sum())/(1+len(ca)))
    pf=first_fail(ds,bh(pp))[0] if not np.isnan(pp).any() else np.nan
    bn=first_fail(ds,bh(stats.binom.sf(np.round(acc*N).astype(int)-1,N,1/3)))[0]
    b4=first_fail(ds,bh(stats.binom.sf(np.round(acc*(N//4)).astype(int)-1,N//4,1/3)))[0]
    allf.append((pf,bn,b4)); print('%4d  %9.2f  %7.2f  %9.2f'%(s,pf,bn,b4))
A=np.array(allf,float); print('median floors: perm=%.2f binom_n=%.2f binom_n/4=%.2f | mean perm=%.2f | seeds below 6.5: perm=%d binom_n=%d'%(np.nanmedian(A[:,0]),np.median(A[:,1]),np.median(A[:,2]),np.nanmean(A[:,0]),(A[:,0]<6.5).sum(),(A[:,1]<6.5).sum()))
print('\n=== B. PER-CLASS RECALL vs DEPTH across seeds (median [min,max]) ===')
alld=sorted(set(d for s in seeds for d in data[s]))
show=[d for d in alld if any(abs(d-x)<0.03 for x in (0,3.5,5.0,6.5,7.5,8.18,10.91,15.0))]
print('depth    '+'   '.join('%-24s'%n for n in names))
for d in show:
    row=[]
    for c in range(3):
        v=np.array([data[s][d][0][c,c]/data[s][d][0][c].sum() for s in seeds if d in data[s]])
        row.append('%.2f [%.2f,%.2f]'%(np.median(v),v.min(),v.max()))
    print('%6.2f   '%d+'   '.join('%-24s'%x for x in row))
print('\n=== C. FIRST DEPTH WHERE CLASS RECALL < 0.45 (across seeds) ===')
for c in range(3):
    cr=[]
    for s in seeds:
        ds=sorted(data[s]); cross=next((d for d in ds if data[s][d][0][c,c]/data[s][d][0][c].sum()<0.45),np.nan); cr.append(cross)
    cr=np.array(cr,float)
    print('%-10s median=%.2f  min=%.2f  seeds crossing before 6.50 cm: %d/%d  before 8.48 cm: %d/%d  never: %d'%(names[c],np.nanmedian(cr),np.nanmin(cr),(cr<6.5).sum(),len(cr),(cr<8.48).sum(),len(cr),np.isnan(cr).sum()))
print('\n=== D. WHERE WINDOWS GO AT DEPTH: share predicted healthy (all true classes pooled), median [min,max] ===')
for d in [x for x in alld if any(abs(x-y)<0.03 for x in [x] for y in (6.5,8.18,10.91,15.0))]:
    v=[]
    for s in seeds:
        if d in data[s]:
            r=data[s][d][0]; v.append(r[:,1].sum()/r.sum())
    v=np.array(v); print('depth %5.2f: predicted-healthy share median=%.2f [%.2f,%.2f]  (chance level 0.33)'%(d,np.median(v),v.min(),v.max()))
print('\nDONE')

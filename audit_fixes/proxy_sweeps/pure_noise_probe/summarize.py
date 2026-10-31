import csv,collections
rows=[r for k in ('rf','resnet') for r in csv.DictReader(open(f'raw_counts_{k}.csv'))]
with open('raw_counts_all.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
cls=['cirrhosis','healthy','fibrosis']
def sh(rs):
    c=[sum(int(r['pred_'+k]) for r in rs) for k in cls];n=sum(c);return c,[x/n for x in c],n
def line(lab,rs):
    c,s,n=sh(rs);flag='  <-- COLLAPSE (>=95% one class)' if max(s)>=.95 else ''
    return f'  {lab:<10} n={n:<5} cirr {s[0]*100:5.1f}%  healthy {s[1]*100:5.1f}%  fibr {s[2]*100:5.1f}%   [counts {c[0]}/{c[1]}/{c[2]}]{flag}'
out=[]
for k,nm in (('rf','RANDOM FOREST'),('resnet','RESNET1D')):
    R=[r for r in rows if r['classifier']==k]
    conds=list(dict.fromkeys(r['condition'] for r in R))
    out.append('='*100+f'\n{nm}  (seeds 101-108, real model only; shares of predictions)')
    for c in conds:
        out.append(f'\n[{c}]'+('  300 pulse-free noise windows per seed' if c=='pure_noise' else '  300 windows per true class per seed (900 per seed)'))
        for s in range(101,109): out.append(line(f'seed {s}',[r for r in R if r['condition']==c and int(r['seed'])==s]))
        out.append(line('POOLED',[r for r in R if r['condition']==c]))
        if c!='pure_noise':
            for t in ('healthy','cirrhosis','significant_fibrosis'):
                rr=[r for r in R if r['condition']==c and r['true_class']==t]
                c3,_,n=sh(rr); idx={'cirrhosis':0,'healthy':1,'significant_fibrosis':2}[t]
                out.append(f'     recall {t:<21} pooled {c3[idx]/n*100:5.1f}%   per seed: '+' '.join(f"{int(r['pred_'+{'cirrhosis':'cirrhosis','healthy':'healthy','significant_fibrosis':'fibrosis'}[t]])/int(r['n'])*100:3.0f}" for r in sorted(rr,key=lambda r:int(r['seed']))))
open('summary.txt','w').write('\n'.join(out)+'\n')
print('\n'.join(out))

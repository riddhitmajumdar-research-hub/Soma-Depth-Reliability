"""Builds the research map (two images) from map_spec.py. Usage: python3 build_map.py OUTDIR [README]"""
import sys,os,re,csv,textwrap
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from map_spec import NODES,REV
plt.rcParams['svg.fonttype']='none'; plt.rcParams['font.family']='DejaVu Sans'
out=sys.argv[1]; os.makedirs(out,exist_ok=True)
FILL={'A':'#cfe6ff','B':'#ffd9a8'}
EDGE={'step':('#475569',1.4,'-'),'check':('#047857',2.2,'-'),'correction':('#b91c1c',3.2,'-'),'deadend':('#78350f',2.2,(0,(4,3)))}
W,PAD,LINE=400,10,16
def wrap(label): return textwrap.wrap(label,52)
nodes=[]
for i,(log,cfg,kind,label,quote) in enumerate(NODES,1):
    nodes.append(dict(id=f'N{i:02d}',log=log,cfg=cfg,kind=kind,label=label,quote=quote,lines=wrap(label)))
def height(n): return PAD*2+LINE*(1+len(n['lines']))
def column_of(n):
    l=n['log']; return 0 if l<=11 else (1 if l<=25 else 2)
HEADS=['Logs 1 to 11: signal, models and tests','Logs 12 to 25: floor, calibration, stress tests, audits','Logs 26 to 38: records and checks written up afterwards']
def place(sel,cols,heads,top=70):
    pos={};ys=[top]*cols
    for n in sel:
        c=column_of(n) if cols==3 else 0
        n['col']=c; n['x']=30+c*(W+90); n['y']=ys[c]; n['h']=height(n); ys[c]+=n['h']+26; pos[n['id']]=n
    return pos,max(ys)
def legend(ax,x,y,show_b=True,show_seq=True):
    ax.text(x,y,'Legend',fontsize=11,fontweight='bold',va='top')
    items=[('A','step','Configuration A (earlier idealised simulator)'),('B','step','Configuration B (rebuilt simulator)'),
           ('A','step','Thin grey border: step or result'),('A','check','Green border: check or finding'),
           ('A','correction','Thick red border: correction recorded in the log'),('A','deadend','Dashed brown border: dead end recorded in the log')]
    yy=y+24
    for cfg,kind,t in items[:2]+items[2:]:
        if (not show_b) and 'Configuration B' in t: continue
        e=EDGE[kind]; fc=FILL[cfg] if t.startswith('Configuration') or True else '#fff'
        if t.startswith('Configuration'): fc=FILL[cfg]; e=EDGE['step']
        elif True: fc='#ffffff'
        ax.add_patch(FancyBboxPatch((x,yy),44,18,boxstyle='round,pad=0,rounding_size=4',fc=fc,ec=e[0],lw=e[1],ls=e[2]))
        ax.text(x+56,yy+9,t,fontsize=9.5,va='center'); yy+=26
    if show_seq:
        ax.add_patch(FancyArrowPatch((x,yy+9),(x+44,yy+9),arrowstyle='-|>',mutation_scale=12,color='#475569',lw=1.2))
        ax.text(x+56,yy+9,'Next step in the order the work was done (Logs 1 to 25)',fontsize=9.5,va='center'); yy+=26
    ax.add_patch(FancyArrowPatch((x,yy+9),(x+44,yy+9),arrowstyle='-|>',mutation_scale=12,color='#a21caf',lw=1.6,ls='--'))
    ax.text(x+56,yy+9,'A later log revises, explains or builds on an earlier one',fontsize=9.5,va='center')
    return yy+30
def draw(path,sel,cols,heads,title,extra_refs=()):
    pos,ymax=place(sel,cols,heads)
    width=30+cols*(W+90)+(520 if cols==1 else 70)
    leg_h=250
    if cols==3: fig_h=ymax+leg_h
    else: fig_h=max(ymax,leg_h+70)+40
    fig=plt.figure(figsize=(width/100,fig_h/100),dpi=100); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,width); ax.set_ylim(fig_h,0); ax.axis('off')
    ax.text(30,24,title,fontsize=14,fontweight='bold',va='center')
    for c in range(cols): ax.text(30+c*(W+90),56,heads[c],fontsize=10.5,fontweight='bold',va='center',color='#334155')
    for n in sel:
        e=EDGE[n['kind']]
        ax.add_patch(FancyBboxPatch((n['x'],n['y']),W,n['h'],boxstyle='round,pad=0,rounding_size=8',fc=FILL[n['cfg']],ec=e[0],lw=e[1],ls=e[2]))
        ax.text(n['x']+PAD,n['y']+PAD,f"Log {n['log']:02d} · configuration {n['cfg']}",fontsize=8.5,color='#334155',va='top',fontweight='bold')
        for k,l in enumerate(n['lines']): ax.text(n['x']+PAD,n['y']+PAD+LINE*(k+1),l,fontsize=9.5,va='top',color='#0f172a')
    edges=[]
    ids=[n['id'] for n in sel]
    gapcount=[0]
    def arrow(a,b,color,lw,ls,label=None,rad=0.0):
        A,B=pos[a],pos[b]
        if A['col']==B['col']:
            adjacent=abs(sel.index(A)-sel.index(B))==1
            if adjacent:
                p0=(A['x']+W/2,A['y']+A['h']) if B['y']>A['y'] else (A['x']+W/2,A['y']); p1=(B['x']+W/2,B['y']) if B['y']>A['y'] else (B['x']+W/2,B['y']+B['h'])
                ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle='-|>',mutation_scale=12,color=color,lw=lw,ls=ls,zorder=1))
            else:
                gx=A['x']+W+18+gapcount[0]*14; gapcount[0]+=1
                p0=(A['x']+W,A['y']+A['h']/2); p1=(B['x']+W,B['y']+B['h']/2)
                ax.plot([p0[0],gx,gx],[p0[1],p0[1],p1[1]],color=color,lw=lw,ls=ls,zorder=1)
                ax.add_patch(FancyArrowPatch((gx,p1[1]),p1,arrowstyle='-|>',mutation_scale=12,color=color,lw=lw,ls=ls,zorder=1))
                if label: ax.text(gx+4,(p0[1]+p1[1])/2,label,fontsize=8,color=color,rotation=90,va='center')
                return
        else:
            right=B['x']>A['x']
            p0=(A['x']+(W if right else 0),A['y']+A['h']/2); p1=(B['x']+(0 if right else W),B['y']+B['h']/2)
            gx=(A['x']+B['x']+W)/2 if not right else (A['x']+W+B['x'])/2
            gx=gx+(gapcount[0]-1.5)*14; gapcount[0]+=1
            ax.plot([p0[0],gx,gx],[p0[1],p0[1],p1[1]],color=color,lw=lw,ls=ls,zorder=1)
            ax.add_patch(FancyArrowPatch((gx,p1[1]),p1,arrowstyle='-|>',mutation_scale=12,color=color,lw=lw,ls=ls,zorder=1))
        if label and A['col']==B['col']: ax.text((p0[0]+p1[0])/2+8,(p0[1]+p1[1])/2,label,fontsize=8,color=color)
    # sequential arrows, Logs 1-25 only, within the first two columns
    seq=[n for n in sel if n['log']<=25]
    for a,b in zip(seq,seq[1:]):
        if a['col']==b['col'] and cols==3: arrow(a['id'],b['id'],'#475569',1.2,'-')
        elif a['col']==0 and b['col']==1 and cols==3: pass
    first={}; 
    for n in sel: first.setdefault(n['log'],n)
    last={}
    for n in sel: last[n['log']]=n
    for a_,b_,label,q in REVEDGES:
        if a_ in pos and b_ in pos: arrow(a_,b_,'#a21caf',1.6,(0,(4,3)),label)
    ly=ymax+20 if cols==3 else 70
    lx=30 if cols==3 else 30+W+90
    legend(ax,lx,ly,show_b=(cols==1),show_seq=(cols==3))
    fig.savefig(path+'.svg',metadata={'Date':None,'Creator':None}); fig.savefig(path+'.png',dpi=100,metadata={'Software':None}); plt.close(fig)
first_of={};last_of={}
for n in nodes: first_of.setdefault(n['log'],n); last_of[n['log']]=n
REVEDGES=[]
for later,earlier,label,q,tq in REV:
    tgt=next((n for n in nodes if n['log']==earlier and tq and n['quote']==tq),None) or (last_of[earlier] if label in ('builds on',) else first_of[earlier])
    REVEDGES.append((first_of[later]['id'],tgt['id'],label,q))
imgA=[n for n in nodes if n['log']<=38]; imgB=[n for n in nodes if n['log']>=39]
draw(os.path.join(out,'research_map_logs_1_38'),imgA,3,HEADS,'Research map, Logs 1 to 38 (configuration A)')
# image 2: add stub for Log 28 reference
draw(os.path.join(out,'research_map_logs_39_43'),imgB,1,['Logs 39 to 43: viscosity defect (A), rebuilt simulator (B)'],'Research map, Logs 39 to 43 (A: Log 39; B: Logs 40 to 43)')
with open(os.path.join(out,'map_nodes.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['node','log','configuration','kind','label','supporting_sentence_fragment'])
    for n in nodes: w.writerow([n['id'],n['log'],n['cfg'],n['kind'],n['label'],n['quote']])
with open(os.path.join(out,'map_edges.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['from','to','relation','supporting_sentence_fragment'])
    seq=[n for n in nodes if n['log']<=25]
    for a,b in zip(seq,seq[1:]): w.writerow([a['id'],b['id'],'next step in the order the work was done',''])
    for r in REVEDGES: w.writerow(r)
print(len(nodes),'nodes')

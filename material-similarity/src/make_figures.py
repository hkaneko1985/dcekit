"""English manuscript figures from the frozen material-similarity outputs."""
from pathlib import Path
import argparse
import json
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from scipy.spatial import ConvexHull, QhullError

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Draw Figures 2-7 from retained or newly reproduced tables.")
parser.add_argument('--results-dir', type=Path, default=ROOT/'results')
parser.add_argument('--upstream-dir', type=Path, default=ROOT/'data/upstream')
parser.add_argument('--out', type=Path, default=ROOT/'outputs/figures')
args = parser.parse_args()
U, R, OUT = args.upstream_dir.resolve(), args.results_dir.resolve(), args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
B, T, O, RED, GRAY = '#0072B2', '#009E73', '#E69F00', '#D55E00', '#67717B'
COLORS = [B, O, T, '#CC79A7', '#56B4E9', RED]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'axes.titlesize':9,
 'axes.labelsize':8.5, 'xtick.labelsize':8, 'ytick.labelsize':8,
 'legend.fontsize':8, 'axes.spines.top':False, 'axes.spines.right':False,
 'figure.dpi':150, 'savefig.dpi':350, 'svg.hashsalt':'jcim-material-similarity'})

def read(name, upstream=False):
    return pd.read_csv((U if upstream else R)/name)

def save(fig, name):
    buffer=io.BytesIO()
    fig.savefig(buffer, format='png', dpi=350, facecolor='white', bbox_inches='tight', pad_inches=0.08)
    (OUT/f'{name}.png').write_bytes(buffer.getvalue())
    fig.savefig(OUT/f'{name}.svg', facecolor='white', bbox_inches='tight', pad_inches=0.08, metadata={'Date':None})
    plt.close(fig)

def figure2():
    d=read('cross_dataset_diagnostics.csv',True).set_index('dataset')
    v=read('view_disagreement_summary.csv',True).set_index('dataset')
    fig,aa=plt.subplots(1,3,figsize=(7.4,3.2),gridspec_kw={'width_ratios':[1.4,1.1,1]})
    names=['Starrydata','HTEM','NanoMine','PNCExtract'];vals=d.loc[names,'context_gain_positive_fraction'].to_numpy()*100
    aa[0].barh(np.arange(4),vals,color=[B,B,T,GRAY]);aa[0].invert_yaxis()
    aa[0].set(yticks=np.arange(4),yticklabels=names,xlim=(0,116),xlabel='Configurations with positive change (%)',title='A  Within-resource sensitivity')
    for i,x in enumerate(vals):aa[0].text(x+2,i,f'{x:.1f}',va='center',fontsize=7.7)
    names2=['Starrydata','HTEM','NanoMine'];m=v.loc[names2,'median'].to_numpy();lo=v.loc[names2,'minimum'].to_numpy();hi=v.loc[names2,'maximum'].to_numpy()
    aa[1].errorbar(m,np.arange(3),xerr=[m-lo,hi-m],fmt='o',color=O,ecolor='#B58C36',capsize=3)
    aa[1].set(yticks=np.arange(3),yticklabels=names2,xlim=(0,1.02),xlabel='ARI between views',title='B  Different partitions');aa[1].invert_yaxis()
    aa[2].bar([0,1],[20,198],color=B,label='All candidates')
    aa[2].bar([1],[106],color=T,label='Extra context')
    aa[2].set(xticks=[0,1],xticklabels=['HTEM','NanoMine'],ylim=(0,250),ylabel='Cross-source pairs',title='C  Consensus candidates')
    aa[2].text(0,24,'20',ha='center',fontsize=8);aa[2].text(1,202,'198',ha='center',fontsize=8)
    aa[2].legend(loc='upper left',fontsize=6.8,frameon=False,handlelength=1)
    fig.tight_layout(w_pad=.9);save(fig,'figureS1_retrospective_context')

def figure3():
    d=read('cross_dataset_diagnostics.csv',True).set_index('dataset');q=read('objective_disagreement.csv',True)
    fig,aa=plt.subplots(1,3,figsize=(7,3.1))
    x=np.arange(2);names=['HTEM','NanoMine']
    aa[0].bar(x-.18,d.loc[names,'all_bounds_positive_fraction']*100,width=.36,color=T,label='Positive at all bounds')
    aa[0].bar(x+.18,d.loc[names,'bound_sign_change_fraction']*100,width=.36,color=RED,label='Sign changes')
    aa[0].set(xticks=x,xticklabels=names,ylim=(0,108),ylabel='Configurations (%)',title='A  Bound sensitivity')
    aa[0].legend(loc='upper left',fontsize=6.7,frameon=False,handlelength=1)
    h=q[q.dataset=='HTEM'];aa[1].scatter(h.global_gain,h.local_gain,s=9,alpha=.35,c=B,edgecolors='none')
    aa[1].set(xlabel='Global increment',ylabel='Local MRR increment',title='B  HTEM configurations')
    aa[1].text(.95,.04,'Both positive: 4.6%',transform=aa[1].transAxes,ha='right',fontsize=7.5,color=RED)
    for name,col,mark in [('NanoMine',T,'o'),('PNCExtract',O,'s')]:
        t=q[q.dataset==name];aa[2].scatter(t.global_gain,t.local_gain,s=24,c=col,marker=mark,label=name)
    aa[2].set(xlabel='Article-group ARI increment',ylabel='Local MRR increment',title='C  Task dependence');aa[2].legend(frameon=False,fontsize=7)
    for ax in aa[1:]:ax.axhline(0,color=GRAY,lw=.65);ax.axvline(0,color=GRAY,lw=.65)
    fig.tight_layout(w_pad=.9);save(fig,'figureS2_retrospective_uncertainty')

def figure4():
    q=read('weight_response.csv',True);fig,aa=plt.subplots(1,2,figsize=(6.8,3.0))
    for name,col,mark in [('HTEM',B,'o'),('NanoMine',T,'s')]:
        t=q[q.dataset==name];aa[0].plot(t.process_weight,t.global_gain_median,marker=mark,c=col,label=name,lw=1.5,ms=4)
    t=q[q.dataset=='HTEM'];aa[1].plot(t.process_weight,t.local_gain_median,marker='o',c=RED,lw=1.5,ms=4)
    for ax in aa:ax.axhline(0,color=GRAY,lw=.65);ax.set_xlabel(r'Process weight $\lambda$')
    aa[0].set(ylabel='Median global increment',title='A  Global contextual organization');aa[0].legend(frameon=False)
    aa[1].set(ylabel='Median MRR increment',title='B  Local retrieval in HTEM')
    fig.tight_layout(w_pad=1.7);save(fig,'figureS3_retrospective_weights')

def figure5():
    q=read('api_nanomine_validation.csv');q=q[q.bound=='reported'];p=read('representative_pair_views.csv')
    names={'material_identity':'Identity','material_identity_soft':'Identity (lexical)', 'synthesis_pathway':'Pathway', 'experimental_protocol':'Protocol','balanced_instance':'Balanced'}
    fig,aa=plt.subplots(1,2,figsize=(7,3.3),gridspec_kw={'width_ratios':[1.1,1]})
    x=np.arange(len(q));aa[0].bar(x-.18,q.paper_ari,width=.36,color=B,label='Article-group ARI');aa[0].bar(x+.18,q.paper_mrr,width=.36,color=T,label='Same-group MRR')
    aa[0].set(xticks=x,xticklabels=[names[v] for v in q.view],ylim=(0,1.05),title='A  Common NanoMine interface',ylabel='Contextual score')
    aa[0].tick_params(axis='x',rotation=30);aa[0].legend(frameon=False,fontsize=7)
    for tick in aa[0].get_xticklabels():tick.set_ha('right')
    y=np.arange(len(p));rr=p.reported.to_numpy();lo=p.optimistic.to_numpy();hi=p.pessimistic.to_numpy()
    aa[1].errorbar(rr,y,xerr=[rr-lo,hi-rr],fmt='o',color=O,ecolor='#BD9333',capsize=3,ms=4)
    aa[1].set(yticks=y,yticklabels=[names[v] for v in p.view],xlim=(-.02,.45),xlabel='Conditional distance',title='B  L157 S2 and L238 S2');aa[1].invert_yaxis()
    fig.tight_layout(w_pad=1.5);save(fig,'figure2_common_api')

def figure6():
    q=read('nanomine_material_system_embedding.csv');sys=read('nanomine_material_systems.csv');met=read('nanomine_material_system_separation.csv').set_index('view')
    fig,aa=plt.subplots(1,2,figsize=(7,4.1));markers=['o','s','^','D','P','v']
    for ax,view,ttl in zip(aa,['material_identity','balanced_instance'],['A  Constituent identity','B  Balanced instance']):
        t=q[q.view==view]
        for cid,grp in t.groupby('cluster'):
            xy=grp[['pcoa1','pcoa2']].to_numpy()
            if len(xy)>=3 and np.linalg.matrix_rank(xy-xy.mean(axis=0))==2:
                try:ax.add_patch(Polygon(xy[ConvexHull(xy).vertices],fill=False,ls='--',lw=.7,ec=GRAY,alpha=.5,zorder=1))
                except QhullError:pass
        for i,row in enumerate(sys.itertuples()):
            g=t[t.system_id==row.system_id]
            label=row.system_label.replace('SiO2','SiO$_2$').replace('Al2O3','Al$_2$O$_3$')
            ax.scatter(g.pcoa1,g.pcoa2,s=25,c=COLORS[i],marker=markers[i],edgecolors='white',linewidths=.4,label=label,zorder=3)
        ax.set(title=ttl,xlabel='PCoA 1',ylabel='PCoA 2');ax.axhline(0,c='#C8CED3',lw=.5);ax.axvline(0,c='#C8CED3',lw=.5)
        ax.set_aspect('equal',adjustable='datalim')
        m=met.loc[view];ax.text(.03,.98,f'ARI {m.ari_vs_exact_material_system:.3f}\nCross-group 5-NN {m.cross_paper_knn5_material_system_agreement*100:.1f}%',va='top',transform=ax.transAxes,fontsize=8,bbox={'facecolor':'white','edgecolor':'#D3D8DC','alpha':.92,'pad':3})
    h,l=aa[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=7.8,bbox_to_anchor=(.5,.005),columnspacing=1.1,handletextpad=.3)
    fig.tight_layout(rect=(0,.17,1,1),w_pad=1.5);save(fig,'figure3_material_systems')

def ellipse(ax,xy,color):
    ev,vec=np.linalg.eigh(np.cov(xy,rowvar=False));order=np.argsort(ev)[::-1];ev=ev[order];vec=vec[:,order]
    size=2*np.sqrt(-2*np.log(.2))*np.sqrt(np.clip(ev,0,None));ang=np.degrees(np.arctan2(vec[1,0],vec[0,0]))
    ax.add_patch(Ellipse(xy.mean(0),size[0],size[1],angle=ang,facecolor=color,edgecolor=color,alpha=.12,lw=1,zorder=1))

def figure7():
    q=read('nanomine_process_method_embedding.csv');met=read('nanomine_process_method_separation.csv').set_index('view')
    fig,aa=plt.subplots(1,2,figsize=(7,4.05));labels=['Solution processing','Melt mixing','In-situ polymerization'];cols=[B,O,T];marks=['o','s','^']
    for ax,view,ttl in zip(aa,['method_blind_process','method_aware_process'],['A  Method excluded','B  Method included']):
        t=q[q.view==view]
        for i,mid in enumerate(['M1','M2','M3']):
            g=t[t.method_id==mid];xy=g[['pcoa1','pcoa2']].to_numpy();ellipse(ax,xy,cols[i])
            ax.scatter(xy[:,0],xy[:,1],s=26,marker=marks[i],c=cols[i],edgecolors='white',linewidths=.35,alpha=.87,label=labels[i],zorder=3)
        ax.set(title=ttl,xlabel='PCoA 1',ylabel='PCoA 2');ax.set_aspect('equal',adjustable='datalim')
        ax.axhline(0,c='#CAD1D6',lw=.5);ax.axvline(0,c='#CAD1D6',lw=.5)
        m=met.loc[view];ax.text(.03,.98,f'Cross-group 5-NN\n{m.cross_paper_knn5_process_method_agreement*100:.1f}%',va='top',transform=ax.transAxes,fontsize=10,weight='bold',bbox={'facecolor':'white','edgecolor':T if view=='method_aware_process' else '#C9D0D5','alpha':.94,'pad':4},zorder=6)
    h,l=aa[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,.0),handletextpad=.3,columnspacing=1)
    fig.tight_layout(rect=(0,.105,1,1),w_pad=1.5);save(fig,'figure4_process_methods')

if __name__=='__main__':
    for f in [figure2,figure3,figure4,figure5,figure6,figure7]:f()
    print('Wrote Figures 2-4 and S1-S3 in PNG and SVG formats to',OUT)

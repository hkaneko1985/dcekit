#!/usr/bin/env python3
"""Precise scientific figures from the revision evaluation tables."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--results-dir',type=Path,default=ROOT/'results/revision');p.add_argument('--out',type=Path,default=ROOT/'outputs/figures');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'material-similarity-v02'})
names=['balanced_strict','equal_facets','gower_field_pool','sequence_as_bag','sequence_as_set'];labels=['Balanced facets','Equal facets','Gower-style pooling','Order removed','Order + repeats removed'];colors=['#0072B2','#009E73','#67717B','#E69F00','#CC79A7']
def save(fig,name):
 fig.savefig(a.out/(name+'.png'),dpi=350,bbox_inches='tight',facecolor='white')
 fig.savefig(a.out/(name+'.svg'),bbox_inches='tight',facecolor='white',metadata={'Date':None});plt.close(fig)
r=pd.read_csv(a.results_dir/'retrieval_summary.csv')
fig,axes=plt.subplots(1,2,figsize=(7.3,3.8))
for ax,task,metric,title in [(axes[0],'masked_record','mrr','A  Retrieve the original record'),(axes[1],'heldout_method','agreement_at_5','B  Hold out synthesis method')]:
 q=r[(r.task==task)&(r.metric==metric)].set_index('method').loc[names]
 for i,(name,row) in enumerate(q.iterrows()):ax.errorbar(row.article_macro_mean,i,xerr=[[row.article_macro_mean-row.ci_low],[row.ci_high-row.article_macro_mean]],fmt='o',color=colors[i],capsize=3)
 ax.set(yticks=np.arange(5),yticklabels=labels if ax is axes[0] else [],ylim=(4.6,-.6),title=title)
 ax.grid(axis='x',color='#E1E6EB',lw=.6);ax.set_axisbelow(True)
 if task=='masked_record':
  ax.set(xlim=(.90,1.01),xlabel='Article-macro MRR')
 else:
  chance=r[(r.task==task)&(r.method=='random_gallery')].article_macro_mean.iloc[0]
  ax.axvline(chance,ls='--',color='#555555',lw=1,label='Random gallery')
  ax.set(xlim=(.25,.55),xlabel='Cross-group 5-neighbor agreement')
fig.text(.54,.005,'Bars: 95% source-group bootstrap intervals; fixed gallery',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.055,1,1),w_pad=1.5);save(fig,'figure5_metadata_retrieval')
r=pd.read_csv(a.results_dir/'missingness_summary.csv');fig,axes=plt.subplots(1,2,figsize=(7,3.5))
scenarios=['unilateral','bilateral','article_block'];x=np.arange(3);w=.32
for name,label,c,offset in [('active','Pair-active schema','#E69F00',-w/2),('fixed_schema','Fixed reference schema','#0072B2',w/2)]:
 q=r[r.policy==name].set_index('mask').loc[scenarios]
 axes[0].bar(x+offset,q.containment_fraction,width=w,label=label,color=c)
 axes[1].bar(x+offset,q.mean_width,width=w,color=c)
for ax in axes:
 ax.set(xticks=x,xticklabels=['One side','Both sides','Source blocks'],ylim=(0,1.08));ax.grid(axis='y',color='#E1E6EB',lw=.5);ax.set_axisbelow(True)
axes[0].set(ylabel='Fraction retaining the original interval',title='A  Retention of original interval')
axes[1].set(ylabel='Mean settings interval width',title='B  Settings interval width')
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,.075),frameon=False,fontsize=8)
fig.text(.5,.005,'300 record pairs per condition; settings only; no imputation',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.20,1,1));save(fig,'figure6_missingness_sensitivity')
r=pd.read_csv(a.results_dir/'interval_utility_summary.csv')
fig,axes=plt.subplots(2,3,figsize=(7.5,5.1),sharex=True)
rules=['reported_point','coverage_0.5','coverage_0.75','interval_upper']
rule_labels=['Point distance','Point + coverage ≥ 0.50','Point + coverage ≥ 0.75','Upper distance']
rule_colors=['#67717B','#0072B2','#009E73','#CC79A7']
markers=['o','s','^','D']
for col,(view,title) in enumerate([('material_identity','Constituent view'),('experimental_protocol','Protocol view'),('balanced_instance','Balanced view')]):
 q=r[(r.view==view)&(r.mask_fraction==.5)]
 for rule,label,c,m in zip(rules,rule_labels,rule_colors,markers):
  rr=q[q.rule==rule].sort_values('distance_threshold')
  axes[0,col].plot(rr.distance_threshold,rr.retained_fraction*100,color=c,marker=m,ms=3.5,lw=1.2,label=label)
  axes[1,col].plot(rr.distance_threshold,rr.contradiction_fraction*100,color=c,marker=m,ms=3.5,lw=1.2)
 axes[0,col].set(title=title,ylim=(-.12,max(.8,q.retained_fraction.max()*110)))
 axes[1,col].set(xlabel='Distance threshold',ylim=(-2,102),xticks=[.1,.2,.3,.4,.5])
 for ax in axes[:,col]:
  ax.grid(axis='y',color='#E1E6EB',lw=.5);ax.set_axisbelow(True)
axes[0,0].set_ylabel('Accepted / eligible pairs (%)')
axes[1,0].set_ylabel('Contradictions / accepted (%)')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,bbox_to_anchor=(.5,.035),frameon=False,fontsize=8)
fig.text(.5,.005,'Half of query settings hidden; original excluded. Empty acceptance sets have undefined rates (gaps).',ha='center',fontsize=7.6)
fig.tight_layout(rect=(0,.16,1,1),w_pad=1.2,h_pad=1.4)
save(fig,'figure7_interval_interpretation')
print('Wrote Figures 5–7')

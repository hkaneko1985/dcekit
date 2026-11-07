from pathlib import Path
import csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run41 import eval_np
R=Path(__file__).resolve().parents[1]
read=lambda name:list(csv.DictReader((R/'results'/name).open(encoding='utf-8-sig')))
rows=read('summary.csv');summary={(r['source'],r['method']):r for r in rows}
order=['modulus','stress','strain','hardness','Tc'];labels=['Young modulus','Peak stress','Strain','Shore A hardness','Metal Tc']
# Scores compare well-defined zero nonadditivity prediction, not physical interaction recovery.
fig,ax=plt.subplots(figsize=(8.6,4.6));xx=np.arange(len(order))
for j,(method,label,mk)in enumerate([('NN','Regression NN','o'),('YAE','YAE','s'),('YVAE','YVAE','^'),('anchored_quadratic','Endpoint-anchored quadratic','D'),('local_linear','Local linear interpolation','x')]):
 y=[float(summary[(s,method)]['delta_skill'])for s in order]
 ax.plot(xx+(j-2)*.06,y,marker=mk,linestyle='none',markersize=6,label=label)
ax.axhline(0,linestyle='--',linewidth=1,label='Zero-departure reference');ax.set_xticks(xx,labels);ax.set_ylabel('Held-out departure skill: 1 - SSE / sum(observed departure squared)');ax.set_ylim(-.04,1.04);ax.set_title('Source-side validation of endpoint-linear departures');ax.legend(fontsize=8,loc='lower left');ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(R/f'figures/Fig41_1_departure_skill.{ext}',dpi=190)
plt.close(fig)
# Matched controls including shuffled labels, same stored test outcomes.
fig,ax=plt.subplots(figsize=(8.2,4.5))
for j,(method,label,mk) in enumerate([('NN','Trained NN','o'),('NN_shuffled','Within-pair shuffled NN','s'),('anchored_quadratic','Anchored quadratic','^'),('local_linear','Local linear interpolation','D')]):
 y=[float(summary[(s,method)]['rmse_y'])/float(summary[(s,'endpoint_linear')]['rmse_y'])for s in order]
 ax.plot(xx+(j-1.5)*.06,y,marker=mk,linestyle='none',label=label)
ax.axhline(1,linestyle='--',linewidth=1,label='Observed pure-endpoint blending');ax.set_xticks(xx,labels);ax.set_ylabel('Held-out property RMSE / endpoint-blending RMSE');ax.set_title('Property prediction: learning is not uniformly superior to interpolation');ax.legend(fontsize=8,loc='upper left');ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(R/f'figures/Fig41_2_controls.{ext}',dpi=190)
plt.close(fig)
# Pre-existing illustrative point, now left entirely out in training and inner tuning.
a=np.load(R/'data/source_stress.npz');X=a['X'];y=a['y'];idx=int(np.flatnonzero(np.max(abs(X-np.array([0,.5,.5])),axis=1)<1e-12)[0]);splits=json.loads((R/'data/splits.json').read_text());sp=next(s for s in splits if s['source']=='stress'and s['test']==[idx]);grid=np.linspace(0,1,201);gx=np.c_[np.zeros_like(grid),grid,1-grid];curves=[]
for seed in [101,202,303]:
 w=dict(np.load(R/f'models/stress/{sp["outer_id"]}/NN/seed_{seed}.npz'));p,_=eval_np(gx,w);vv,_=eval_np(np.eye(3),w);curves.append(p-grid*vv[1]-(1-grid)*vv[2])
curves=np.array(curves);pure={int(np.argmax(x)):float(v)for x,v in zip(X,y)if (x>1e-12).sum()==1}
ids=[i for i,x in enumerate(X)if x[0]==0 and x[1]>0 and x[2]>0];obs=y[ids]-X[ids,1]*pure[1]-X[ids,2]*pure[2]
fig,ax=plt.subplots(figsize=(7.4,4.4));ax.plot(grid,curves.mean(0),label='NN mean, trained without the 0.5 composition');ax.fill_between(grid,curves.min(0),curves.max(0),alpha=.18,label='Range across 3 initializations');trainids=[i for i in ids if i!=idx];ax.scatter(X[trainids,1],y[trainids]-X[trainids,1]*pure[1]-X[trainids,2]*pure[2],marker='o',label='Training binary observations');ax.scatter([.5],[y[idx]-.5*(pure[1]+pure[2])],marker='*',s=130,label='Held-out observed departure');ax.axhline(0,linestyle='--',linewidth=1);ax.set_xlabel('IBOA mass fraction within the IBOA/PEGDA pair');ax.set_ylabel('Departure from own endpoints [MPa]');ax.set_title('Previously illustrated composition, now held out');ax.legend(fontsize=8,loc='upper left');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(R/f'figures/Fig41_3_heldout_example.{ext}',dpi=190)
plt.close(fig)
# Single-source selection vs all sources under matched target CV.
tr=read('transfer_summary.csv');lookup={(r['target'],r['kind'],r['method'],r['source']):float(r['rmse'])for r in tr}
fig,ax=plt.subplots(figsize=(7.8,4.3));xx=np.arange(4)
for j,(m,label,mk)in enumerate([('single_inner_selected','Single source selected inside training','s'),('all_three','All three frozen source encoders','o')]):
 yy=[lookup[(s,'NN',m,'')]/lookup[(s,'baseline','XD','')]for s in order[:4]]
 ax.plot(xx+(j-.5)*.06,yy,marker=mk,linestyle='none',label=label)
ax.axhline(1,linestyle='--',linewidth=1,label='Composition + fixed features');ax.set_xticks(xx,labels[:4]);ax.set_ylabel('Target KRR RMSE / baseline RMSE');ax.set_title('Cross-property transfer: whole-composition latent features');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(R/f'figures/Fig41_4_single_multi.{ext}',dpi=190)
plt.close(fig)
print('Created four figure pairs')

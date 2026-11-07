from pathlib import Path
import csv,itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def read(name):return list(csv.DictReader((ROOT/'evidence'/name).open(encoding='utf-8-sig')))
rows=read('polymer_curves40.csv');obs=read('observed_source_departures40.csv')
# Three distinct plots. All pairs are shown; no performance-based pair selection.
for idx,pair in enumerate(['IDA--IBOA','IDA--PEGDA','IBOA--PEGDA'],1):
 r=[v for v in rows if v['source']=='stress' and v['encoder']=='NN' and v['pair']==pair]
 t=sorted(set(float(x['fraction_i']) for x in r));arr=np.array([[float(x['prediction_excess']) for x in r if int(x['seed'])==seed] for seed in [101,202,303]])
 fig,ax=plt.subplots(figsize=(7.0,3.9))
 ax.plot(t,arr.mean(axis=0),linewidth=2,label='Frozen NN: mean of 3 seeds')
 ax.fill_between(t,arr.min(axis=0),arr.max(axis=0),alpha=.18,label='Range across 3 seeds')
 o=sorted([x for x in obs if x['source']=='stress' and x['pair']==pair],key=lambda v:float(v['fraction_i']))
 ax.scatter([float(x['fraction_i']) for x in o],[float(x['observed_departure']) for x in o],marker='x',s=45,label='Observed source departures (in-sample)')
 ax.axhline(0,linestyle='--',linewidth=1,label='Linear blending reference')
 ax.set_xlabel(f'Mass fraction of {pair.split("--")[0]} within the pair')
 ax.set_ylabel('Deviation from endpoint blending [MPa]')
 ax.set_title(f'Peak-stress source model: {pair.replace("--", " / ")}')
 ax.legend(fontsize=8,loc='best');fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(ROOT/f'figures/FigS40_{idx}_source_stress.{ext}',dpi=190)
 plt.close(fig)
r=read('paired_comparison40.csv');r=[x for x in r if x['method']=='KRR']
order=['polymer_modulus','polymer_stress','polymer_strain','polymer_hardness'];r=sorted(r,key=lambda a:order.index(a['dataset']))
labels=['Young modulus','Peak stress','Strain','Shore A hardness'];views=[('source_linear_predictions','Classical linear: 3 outputs'),('source_quadratic_predictions','Classical quadratic: 3 outputs'),('NN_boundary_pred','NN: 3 boundary outputs'),('NN_boundary_latent','NN: 24 latent features')]
fig,ax=plt.subplots(figsize=(8.2,4.4));x=np.arange(4)
for j,(key,label) in enumerate(views):
 y=[float(v[key])/float(v['baseXD']) for v in r]
 ax.plot(x+(j-1.5)*.06,y,marker=['o','s','^','D'][j],linestyle='none',markersize=6,label=label)
ax.axhline(1,linestyle='--',linewidth=1,label='Common X + D baseline')
ax.set_xticks(x,labels);ax.set_ylabel('RMSE / common baseline RMSE');ax.set_title('Matched downstream KRR: source-derived mixture features')
ax.legend(fontsize=8,loc='upper left');ax.set_ylim(.65,1.65);ax.grid(axis='y',alpha=.2);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(ROOT/f'figures/Fig5_classical_controls40.{ext}',dpi=190)
plt.close(fig)
# tabular summaries consumed by documents/workbook
models=read('model_nonadditivity40.csv');summ=[]
for src in ['modulus','stress','strain','hardness','Tc']:
 for kind in ['NN','YAE','YVAE','random']:
  m=[v for v in models if v['source']==src and v['encoder']==kind]
  summ.append({'source':src,'encoder':kind,'n_source_models':len(m),'pairs_per_model':int(m[0]['n_pairs']), 'mean_of_pair_median_latent_rms_scaled':float(np.mean([float(v['median_latent_rms_scaled']) for v in m])),'mean_of_pair_median_pred_rms_scaled':float(np.mean([float(v['median_pred_rms_scaled']) for v in m])),'meaning':'non-additivity magnitude relative to chosen baseline, not physical interaction strength'})
with (ROOT/'evidence/nonadditivity_summary40.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(summ[0]));w.writeheader();w.writerows(summ)
print('Figures and summary written.')

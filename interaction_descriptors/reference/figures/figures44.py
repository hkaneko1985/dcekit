from pathlib import Path
import os,csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/mplconfig44')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((R/'results/summary.csv').open(encoding='utf-8-sig')))
V={(a['case'],int(a['budget']),a['kind'],a['method']):float(a['rmse_mean'])for a in rows}
cases=[('metals_volume','Metal atomic volume','RMSE (Å³/atom)'),('oxides_formation','Oxide formation energy','RMSE (eV/atom)')]
series=[('KRR',[('NN','XD','Composition + fixed descriptors','o'),('NN','single_source_fixed','Same-property single source','s'),('NN','all_frozen','All sources: regression NN','^'),('YAE','all_frozen','All sources: YAE','D'),('YVAE','all_frozen','All sources: YVAE','v'),('random','all_frozen','All sources: untrained control','x')]),('Adaptive',[('NN','base_head','Target NN without source descriptors','o'),('NN','frozen_concat_head','Frozen source descriptors','s'),('NN','gated_lastlayer_concat','Gate + source last-layer adaptation','^'),('random','frozen_concat_head','Frozen untrained descriptors','x')])]
for case,title,unit in cases:
 for group,seq in series:
  fig,ax=plt.subplots(figsize=(8,4.8))
  for k,m,label,marker in seq:ax.plot([16,32,64],[V[case,n,k,m]for n in [16,32,64]],marker=marker,label=label,linewidth=1.5)
  ax.set_xticks([16,32,64]);ax.set_xlabel('Target training labels per outer fit');ax.set_ylabel(unit);ax.set_title(title+(' | KRR'if group=='KRR'else' | Matched prediction head'));ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8);fig.tight_layout()
  for ext in ['png','svg']:fig.savefig(R/f'figures/Learning_{group}_{case}.{ext}',dpi=180)
  plt.close(fig)
print('Four figures regenerated')

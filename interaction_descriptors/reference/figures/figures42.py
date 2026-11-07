"""Re-create the four v42 comparison figures from the unrounded CSV."""
from pathlib import Path
import os,csv
R=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(R/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=list(csv.DictReader((R/'results/summary.csv').open(encoding='utf-8-sig')))
v={(r['case'],r['kind'],r['method']):float(r['rmse_mean'])for r in rows}
cases=[('metals_volume','Metal atomic volume','RMSE (Å³/atom)'),('oxides_formation','Oxide formation energy','RMSE (eV/atom)')]
specs=[('KRR',['single_source_fixed','single_source_nested','all_frozen'],['Same-property\nsingle source','Inner-CV-selected\nsingle source','All 24 sources\n(frozen)'],'XD','Composition + fixed descriptors'),('Adaptive',['frozen_concat_head','gated_frozen_concat','gated_lastlayer_concat','sum_frozen_head','gated_lastlayer_sum'],['Frozen\nconcat.','Gated\nfrozen','Gated + last\nlayer update','Frozen\nmean','Adaptive\nweighted sum'],'base_head','Target NN without source features')]
for case,title,unit in cases:
 for prefix,methods,labels,baseline,baseline_label in specs:
  fig,ax=plt.subplots(figsize=(8,4.4))
  for kind,marker in [('NN','o'),('YAE','s'),('YVAE','^')]:
   ax.plot(range(len(methods)),[v[case,kind,m]for m in methods],marker=marker,linewidth=1.4,label=kind)
  ax.axhline(v[case,'NN',baseline],linestyle='--',linewidth=1.2,label=baseline_label)
  ax.set_xticks(range(len(methods)),labels);ax.set_ylabel(unit);ax.set_title(title+(' | KRR'if prefix=='KRR'else' | Matched prediction head'))
  ax.legend(fontsize=8);ax.grid(axis='y',alpha=.25);fig.tight_layout()
  for extension in ('png','svg'):fig.savefig(R/f'figures/{prefix}_{case}.{extension}',dpi=180)
  plt.close(fig)
print('4 comparison figures regenerated')

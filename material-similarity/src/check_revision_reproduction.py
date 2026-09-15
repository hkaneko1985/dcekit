#!/usr/bin/env python3
"""Compare current evaluation outputs without overwriting frozen reference results."""
import argparse,csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=['retrieval_queries.csv','retrieval_summary.csv','retrieval_resolved_subset_sensitivity.csv','missingness_pairs.csv','missingness_summary.csv','sequence_counterexamples.csv','representation_sensitivity_pairs.csv','scale_block_bootstrap.csv','evaluation_summary.json']
FILES += ['mask_key_audit.csv','interval_utility_queries.csv','interval_utility_mask_audit.csv','interval_utility_summary.csv','interval_utility_audit.json','heldout_paired_comparisons.csv']
FILES += ['baseline_documentation_queries.csv','baseline_documentation_summary.csv','baseline_documentation_audit.json']
def main():
 p=argparse.ArgumentParser();p.add_argument('--actual',type=Path,default=ROOT/'outputs/revision');a=p.parse_args();errors=[]
 def eq(x,y):
  if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(eq(x[k],y[k]) for k in x)
  if isinstance(x,list):return isinstance(y,list) and len(x)==len(y) and all(eq(i,j) for i,j in zip(x,y))
  try:
   xx,yy=float(x),float(y);return (math.isnan(xx) and math.isnan(yy)) or math.isclose(xx,yy,rel_tol=1e-7,abs_tol=1e-8)
  except (ValueError,TypeError):return x==y
 for name in FILES:
  def read(path):return json.loads(path.read_text()) if path.suffix=='.json' else list(csv.DictReader(path.open()))
  actual=a.actual/name;reference=ROOT/'results/revision'/name
  ok=actual.exists() and eq(read(reference),read(actual));print(('PASS ' if ok else 'DIFFER ')+name)
  if not ok:errors.append(name)
 report={'files_checked':len(FILES),'passed':not errors,'differences':errors,'scope':'same-information retrieval and missingness/alignment/scale evaluations'}
 (ROOT/'outputs/revision_reproduction_check.json').write_text(json.dumps(report,indent=2)+'\n')
 if errors:raise SystemExit('Revision results differ: '+', '.join(errors))
if __name__=='__main__':main()

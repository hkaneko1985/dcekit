"""Read-only reconstruction of new sources and audit of current-cohort exclusion."""
from pathlib import Path
import csv,json,re,collections,hashlib,numpy as np
R=Path(__file__).resolve().parents[1]
def key(d,places=10):
 s=sum(d.values())
 if s<=0 or not np.isfinite(s):raise ValueError('Invalid elemental sum')
 return tuple(sorted((e,round(v/s,places))for e,v in d.items()if v>1e-12))
def formula(s):
 toks=re.findall(r'[A-Z][a-z]?|[0-9]*\.?[0-9]+|[()\[\]]',s)
 if ''.join(toks)!=s:raise ValueError(s)
 pos=0
 def parse():
  nonlocal pos
  c=collections.Counter()
  while pos<len(toks):
   a=toks[pos]
   if a in (')',']'):break
   pos+=1
   if a in ('(','['):
    v=parse()
    if pos>=len(toks)or toks[pos]not in(')',']'):raise ValueError(s)
    pos+=1
   elif re.fullmatch(r'[A-Z][a-z]?',a):v={a:1.}
   else:raise ValueError(s)
   fac=1.
   if pos<len(toks)and re.fullmatch(r'[0-9]*\.?[0-9]+',toks[pos]):fac=float(toks[pos]);pos+=1
   for e,n in v.items():c[e]+=fac*n
  return c
 c=parse()
 if pos!=len(toks):raise ValueError(s)
 return dict(c)
def main():
 excluded={tuple((e,float(v))for e,v in k)for k in json.loads((R/'data/source_excluded_jarvis_keys.json').read_text())};assert len(excluded)==865
 rows=[]
 for sid,filename,field in [('supercon_all_Tc','supercon_saved.csv','critical_temp'),('expt_gap','matbench_expt_gap.csv','gap expt')]:
  raw=list(csv.DictReader((R/'raw'/filename).open(encoding='utf-8-sig')));g=collections.defaultdict(list);nex=0
  for r in raw:
   d={e:float(v)for e,v in r.items()if e not in('',field)and float(v)>0}if sid.startswith('supercon')else formula(r['composition'])
   k=key(d);y=float(r[field]);assert np.isfinite(y)
   if k in excluded:nex+=1;continue
   g[k].append(y)
  a=np.load(R/f'sources/{sid}.npz');ks=sorted(g);X=np.array([[dict(k).get(e,0)for e in a['elements']]for k in ks]);y=np.array([np.median(g[k])for k in ks])
  dx=float(np.max(abs(X-a['X'])));dy=float(np.max(abs(y-a['y'])));assert dx<1e-12 and dy<1e-12
  rows.append(dict(source=sid,raw_rows=len(raw),source_compositions=len(y),excluded_rows=nex,max_X_difference=dx,max_y_difference=dy,raw_sha256=hashlib.sha256((R/'raw'/filename).read_bytes()).hexdigest()))
 # shared parent normalization to 8 decimal digits, as prior stored source keys
 ex8={tuple((e,round(v,8))for e,v in k)for k in excluded};union=set();obs=0
 for c in json.loads((R/'data/source_catalog.json').read_text()):
  a=np.load(R/f"sources/{c['source']}.npz");ks={key(dict(zip(a['elements'],x)),8)for x in a['X']};assert not ks&ex8
  union|=ks;obs+=len(a['X'])
 result=dict(new_sources=rows,old_and_new_sources_without_parent_overlap=True,parent_exclusions=865,total_source_property_composition_rows=obs,unique_source_compositions_round8=len(union),source_tasks=24)
 (R/'results/source_input_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

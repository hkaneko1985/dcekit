"""Prepare source/target datasets by chemistry and provenance, never target outcomes.
No component-number filter is imposed within the metallic / mixed oxide definitions.
Polymer branch is a separate repeat-unit descriptor extension, not blend transfer.
"""
from pathlib import Path
import csv, json, math, hashlib, collections, datetime, zipfile, sys
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupKFold
R=Path(__file__).resolve().parents[1]
NONMETALS=set('H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe At Rn'.split())
SEEDS=[101,202,303]
def read(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def write(p,rows):
 rows=list(rows)
 if not rows:return
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.tolist()if isinstance(v,np.ndarray)else v.item()))
def key(d):
 s=sum(d.values());return tuple(sorted((e,round(v/s,10)) for e,v in d.items() if v>1e-12))
def formula(s):
 import re
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
    v=parse();assert pos<len(toks) and toks[pos] in (')',']');pos+=1
   elif re.fullmatch(r'[A-Z][a-z]?',a):v={a:1.}
   else:raise ValueError((s,a))
   fac=1.
   if pos<len(toks)and re.fullmatch(r'[0-9]*\.?[0-9]+',toks[pos]):fac=float(toks[pos]);pos+=1
   for e,n in v.items():c[e]+=fac*n
  return c
 c=parse()
 if pos!=len(toks):raise ValueError(s)
 return dict(c)
def vector(keys,elements):return np.array([[dict(k).get(e,0)for e in elements]for k in keys])
def physical(X,ele):
 p=Chem.GetPeriodicTable();c=np.array([[p.GetAtomicNumber(e),p.GetAtomicWeight(e),p.GetRcovalent(e),p.GetNOuterElecs(e)]for e in ele]);out=[]
 for v in c.T:
  m=X@v;out.extend([m,np.sqrt(np.maximum((X*(v[None,:]-m[:,None])**2).sum(1),0)),np.where(X>0,v,np.inf).min(1),np.where(X>0,v,-np.inf).max(1)])
 return np.column_stack(out+[-(X*np.log(np.maximum(X,1e-300))).sum(1),(X**2).sum(1)])
def groupsplits(X,g,budgets):
 ans={}
 for budget in budgets:
  reps=[]
  for rep in range(2):
   folds=[]
   for fi,(pool,te) in enumerate(GroupKFold(3,shuffle=True,random_state=35000+rep).split(X,groups=g)):
    rng=np.random.default_rng(35100+rep*10+fi)
    # Nested label budgets: same shuffled training order at every budget.
    tr=np.sort(rng.permutation(pool)[:budget]);assert len(tr)==budget
    inn=[]
    for a,b in GroupKFold(3,shuffle=True,random_state=35200+rep*10+fi).split(X[tr],groups=g[tr]):
     aa=tr[a];bb=tr[b];assert not set(g[aa])&set(g[bb]);inn.append({'train':aa.tolist(),'valid':bb.tolist()})
    assert not set(g[tr])&set(g[te]);folds.append({'train':tr.tolist(),'test':te.tolist(),'inner':inn})
   reps.append(folds)
  ans[str(budget)]=reps
 return ans

def inorganic():
 target=collections.defaultdict(list)
 for r in read(R/'raw/jarvis_metadata.csv'):
  d={e:float(v)for e,v in (s.split(':')for s in r['composition'].split())};target[key(d)].append(r)
 sc=collections.defaultdict(list)
 for r in read(R/'raw/supercon_saved.csv'):
  d={e:float(v)for e,v in r.items()if e not in ('','critical_temp')and float(v)>0}
  if not set(d)&NONMETALS:sc[key(d)].append((float(r['critical_temp']),r['']))
 eps=collections.defaultdict(list)
 for r in read(R/'raw/dielectric_saved.csv'):
  if r['domain']=='organic' or not r['formula']:continue
  d=formula(r['formula']);es=set(d)
  if 'O'in es and not((es&NONMETALS)-{'O'})and len(es-{'O'})>=1:
   eps[key(d)].append(r)
 summary=[]
 for domain,raws in [('metals',sc),('mixed_oxides',eps)]:
  srcdict={k:v for k,v in raws.items()if k not in target} # remove exact target parents before target y inspection
  ele=sorted({e for k in srcdict for e,v in k});freq=collections.Counter(e for k in srcdict for e,v in k)
  targetkeys=[];exclude=[]
  for k,v in sorted(target.items()):
   es={e for e,_ in k};chem=(not es&NONMETALS and len(es)>=2)if domain=='metals'else('O'in es and not((es&NONMETALS)-{'O'})and len(es-{'O'})>=1)
   if not chem:continue
   reason='unknown_source_element'if not es<=set(ele)else''
   if reason:exclude.append({'id':str(k),'reason':reason})
   else:targetkeys.append(k)
  sk=sorted(srcdict);SX=vector(sk,ele);TX=vector(targetkeys,ele)
  if domain=='metals':
   SY=np.array([[np.mean([q[0]for q in srcdict[k]])]for k in sk]);props=['superconducting_Tc_K'];transform=['identity']
  else:
   SY=np.array([[np.mean([float(r[p])for r in srcdict[k]])for p in ['electronic_dielectric','static_dielectric_total']]for k in sk]);props=['electronic_epsilon','total_epsilon'];transform=['log1p','log1p']
  SG=np.array(['-'.join(e for e,v in k)for k in sk]);TG=np.array(['-'.join(e for e,v in k)for k in targetkeys])
  TY=np.array([np.median([float(r['formation_eV_atom'])for r in target[k]])for k in targetkeys]);TD=physical(TX,ele)
  ids=np.array([str(k)for k in targetkeys]);sids=np.array([str(k)for k in sk]);
  np.savez_compressed(R/f'data/source_{domain}.npz',X=SX,Y=SY,groups=SG,ids=sids,input_names=ele,properties=props,transforms=transform,reconstruct_simplex=True)
  np.savez_compressed(R/f'data/target_{domain}.npz',X=TX,D=TD,y=TY,groups=TG,ids=ids,input_names=ele)
  budgets=[32,64];assert len(TY)>96
  dump(R/f'data/splits_{domain}.json',groupsplits(TX,TG,budgets))
  write(R/f'data/source_{domain}.csv',[{'id':str(k),'n_elements':len(k),'group':g,**dict(zip(props,y)),**dict(zip(ele,x))}for k,g,x,y in zip(sk,SG,SX,SY)])
  write(R/f'data/target_{domain}.csv',[{'id':str(k),'group':g,'n_elements':len(k),'formation_median_eV_atom':y,'n_structures':len(target[k]),'jids':'|'.join(r['jid']for r in target[k]),**dict(zip(ele,x))}for k,g,x,y in zip(targetkeys,TG,TX,TY)])
  write(R/f'data/exclusions_{domain}.csv',exclude)
  summary.append({'domain':domain,'source_n':len(sk),'source_groups':len(set(SG)),'source_component_counts':dict(collections.Counter(map(len,sk))),'target_n':len(TY),'target_groups':len(set(TG)),'target_component_counts':dict(collections.Counter(map(len,targetkeys))),'source_exact_target_removed':len(raws)-len(srcdict),'n_features':len(ele),'source_properties':props,'targets':['formation_median_eV_atom'],'target_exclusions':len(exclude)})
 return summary

P_NAMES=['MolWt','MolLogP','MolMR','TPSA','NumValenceElectrons','HeavyAtomCount','NHOHCount','NOCount','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRotatableBonds','RingCount','NumAromaticRings','NumAliphaticRings','NumSaturatedRings','FractionCSP3','NumAmideBonds','NumBridgeheadAtoms','NumSpiroAtoms','BertzCT','BalabanJ','HallKierAlpha','Kappa1','Kappa2','Kappa3','Chi0v','Chi1v']
FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
def mol(s):
 m=Chem.MolFromSmiles(s)
 if m is None:raise ValueError(s)
 for a in m.GetAtoms():a.SetAtomMapNum(0)
 return m,Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)
def pdesc(m):return np.array([getattr(Descriptors,n)(m)for n in P_NAMES])
def pgroup(ms):
 fps=[FP.GetFingerprint(m)for m in ms];d=[]
 for i in range(1,len(fps)):d.extend([1-v for v in DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])])
 cl=Butina.ClusterData(d,len(fps),0.5,isDistData=True,reordering=True)
 g=np.empty(len(fps),dtype='U16')
 for i,c in enumerate(cl):g[list(c)]=f'cluster_{i:04d}'
 return g,fps

def polymers():
 src=collections.defaultdict(list);tm={};tar=collections.defaultdict(dict);dups=[]
 for r in read(R/'raw/dielectric_saved.csv'):
  if not r['source_dataset'].startswith('Polymer_Genome'):continue
  m,k=mol(r['repeat_unit_smiles']);src[k].append((m,float(r['static_dielectric_total']),r['source_id']))
 if not (R/'raw/polymer_heat_capacity_saved.csv').exists():
  with zipfile.ZipFile(R/'recovery/02_heat_capacity_300K_integrated_dataset.zip')as z:
   (R/'raw/polymer_heat_capacity_saved.csv').write_bytes(z.read('heat_capacity_300K_integrated_targets.csv'))
 for fn,prop,col in [('refractive_saved.csv','polymer_RI','refractive_index_n'),('polymer_heat_capacity_saved.csv','polymer_Cp','heat_capacity_native_value')]:
  for r in read(R/'raw'/fn):
   if r['source_dataset']!='RadonPy_PI1070':continue
   m,k=mol(r['repeat_unit_smiles']);tm[k]=m
   if prop in tar[k]:dups.append((k,prop))
   tar[k].setdefault(prop,[]).append(float(r[col]));tar[k].setdefault('source_ids',set()).add(r['source_id'])
 # Remove exact canonical repeats, and identical radius-2 fingerprints conservatively.
 tfps={FP.GetFingerprint(m).ToBitString()for m in tm.values()};sk=[];exclude=[]
 for k in sorted(src):
  reason='exact_target_canonical_repeat'if k in tar else('identical_target_Morgan_bit_vector'if FP.GetFingerprint(src[k][0][0]).ToBitString()in tfps else'')
  if reason:exclude.append({'canonical_repeat':k,'reason':reason})
  else:sk.append(k)
 SX=np.array([pdesc(src[k][0][0])for k in sk]);SY=np.array([[np.mean([v[1]for v in src[k]])]for k in sk]);SG,_=pgroup([src[k][0][0]for k in sk]);
 assert np.isfinite(SX).all() and np.isfinite(SY).all()
 np.savez_compressed(R/'data/source_polymer.npz',X=SX,Y=SY,groups=SG,ids=sk,input_names=P_NAMES,properties=['crystalline_static_epsilon'],transforms=['log1p'],reconstruct_simplex=False)
 write(R/'data/source_polymer.csv',[{'canonical_repeat':k,'group':g,'epsilon':y[0],**dict(zip(P_NAMES,x))}for k,g,x,y in zip(sk,SG,SX,SY)])
 write(R/'data/exclusions_polymer_source.csv',exclude)
 summary=[]
 for prop in ['polymer_RI','polymer_Cp']:
  tk=sorted(k for k in tar if prop in tar[k]);TX=np.array([pdesc(tm[k])for k in tk]);TY=np.array([np.mean(tar[k][prop])for k in tk]);TG,fps=pgroup([tm[k]for k in tk]);
  assert np.isfinite(TX).all()and np.isfinite(TY).all()
  # Strong fixed structural fingerprint is included in BOTH baseline and augmented model.
  D=np.array([list(map(int,rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=256).GetFingerprint(tm[k]).ToBitString()))for k in tk],dtype=np.uint8)
  np.savez_compressed(R/f'data/target_{prop}.npz',X=TX,D=D,y=TY,groups=TG,ids=tk,input_names=P_NAMES)
  dump(R/f'data/splits_{prop}.json',groupsplits(TX,TG,[64,128]))
  write(R/f'data/target_{prop}.csv',[{'canonical_repeat':k,'group':g,'y':y,'original_ids':'|'.join(sorted(tar[k]['source_ids'])),**dict(zip(P_NAMES,x))}for k,g,x,y in zip(tk,TG,TX,TY)])
  summary.append({'domain':prop,'source_n':len(sk),'source_groups':len(set(SG)),'source_initial_n':len(src),'source_exact_or_fp_target_removed':len(exclude),'target_n':len(tk),'target_groups':len(set(TG)),'largest_target_group':max(collections.Counter(TG).values()),'source_properties':['crystalline_static_epsilon'],'target_units':'dimensionless'if prop.endswith('RI')else'J/(kg K)','n_features':len(P_NAMES),'n_fixed_fingerprint_features':256,'mixture_proportions':False})
 return summary
if __name__=='__main__':
 summary=inorganic()+polymers();dump(R/'data/cohort_summary.json',summary)
 dump(R/'protocol.json',{'version':'0.35.0','locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_component_count_filter':True,'primary':'independent-source full-input frozen latent descriptors; NN/YAE/YVAE; 3 seeds; 6 regressors; 2 repeated 3-fold group CV, main target budget64; input-only and fixed-descriptor baselines; independent random encoder control','secondary':'prediction and both outputs only seed101 at budget64 all6 regressors; label learning curves32/64 inorganic,64/128 polymer for KRR and LightGBM only','new_raw_web_download':False,'source_sources':{'metals':'archived Hamidieh/UCI superconductivity, all metallic component counts','mixed_oxides':'archived Petousis source; >=1 metal and O; arbitrary component count','polymer':'archived Polymer Genome crystalline dielectric382 original rows'},'target_sources':{'inorganic':'JARVIS/ColabFit saved multi-structure subset; median formation energy over stored structures','polymer':'RadonPy PI1070 amorphous simulation 300K 1atm'},'cohorts':summary,'polymer_extension':'repeat-unit RDKit descriptors and Morgan fingerprint controls; NOT a polymer-blend interaction test','no_target_y_in_source_training':True,'groups':'element sets for inorganic; Morgan radius2 Butina distance0.5 clusters for polymers','negative_controls':'independently initialized random NN with same source preprocessing and feature dimensions','source_model':'input->32SiLU->16SiLU->8latent->16SiLU->1; AE reconstructs original source input; normalized squared reconstruction; YVAE KL1e-3; rec weight0.3','source_selection':'group3 CV choose epochs100/200/400 by source y-MSE only; reconstruction reported, not guaranteed','source_input_scaling':'inorganic fractions unscaled; polymer numerical descriptors standardized source-only; polymer AE scaled descriptor reconstruction','metrics':'RMSE per pooled repeat then mean seeds/repeats; no prediction ensemble; all adverse results retained','caveats':['targets in prior inorganic development dataset, not all new','polymer crystalline-to-amorphous physics/state difference','canonical repeat and fingerprint matching not full periodic polymer equivalence','no physical interaction identifiability claim','MPEA raw still not verified; FeCoNi not available']})
 print(json.dumps(summary,ensure_ascii=False,indent=2))

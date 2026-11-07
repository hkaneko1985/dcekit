"""Fixed-hyperparameter source-only training. Target labels are not loaded."""
import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from scipy.spatial.distance import cdist
from source_common import Network,fit_at,apply_np
R=Path(__file__).resolve().parents[1]
CAT=json.loads((R/'data/source_catalog.json').read_text());torch.set_num_threads(1)
EPOCHS=200;SEEDS=[101,202,303]
def work(job):
 si,kind=job;meta=CAT[si];sid=meta['source'];start=time.time();a=np.load(R/f'sources/{sid}.npz');X=a['X'];y=a['y'];g=a['groups']
 dest=R/f'models/{sid}/{kind}';dest.mkdir(parents=True,exist_ok=True)
 oof=np.zeros(len(y));folds=[];recs=[]
 for fi,(tr,te)in enumerate(GroupKFold(3,shuffle=True,random_state=36001).split(X,groups=g)):
  model,pre,res=fit_at(X,y,tr,te,kind,360100+si*101+fi,True,[EPOCHS]);oof[te]=res[EPOCHS]['pred'];recs.append(res[EPOCHS]['rec_skill'])
  folds.append(dict(fold=fi,train=tr.tolist(),test=te.tolist()))
 for seed in SEEDS:
  model,pre,_=fit_at(X,y,np.arange(len(y)),np.arange(min(7,len(y))),kind,seed+si*10000+360000,True,[EPOCHS])
  w={k:v.detach().numpy()for k,v in model.state_dict().items()};w.update(pre)
  z,p=apply_np(X,w);w.update(z_mean=z.mean(0),z_sd=np.maximum(z.std(0),1e-10),p_mean=float(p.mean()),p_sd=max(float(p.std()),1e-10))
  np.savez_compressed(dest/f'seed{seed}.npz',**w)
 np.savez_compressed(dest/'source_oof.npz',y=y,pred=oof)
 stats=dict(source=sid,kind=kind,n=len(y),p=X.shape[1],groups=len(set(g)),q2=1-float(np.sum((oof-y)**2)/np.sum((y-y.mean())**2)),rmse=float(np.sqrt(np.mean((oof-y)**2))),reconstruction_skill=None if kind=='NN'else float(np.mean(recs)),epochs=EPOCHS,elapsed_seconds=time.time()-start,folds=folds)
 (dest/'diagnostics.json').write_text(json.dumps(stats,indent=2));return{k:v for k,v in stats.items()if k!='folds'}
def randoms_features():
 checks=[]
 for si,meta in enumerate(CAT):
  sid=meta['source'];src=np.load(R/f'sources/{sid}.npz');X=src['X'];els=list(src['elements']);dest=R/f'models/{sid}/random';dest.mkdir(exist_ok=True)
  for seed in SEEDS:
   torch.manual_seed(seed+si*10000+760000);m=Network(X.shape[1],'NN').double()
   w={k:v.detach().numpy()for k,v in m.state_dict().items()};w.update(x_mean=np.zeros(X.shape[1]),x_sd=np.ones(X.shape[1]),y_mean=0.,y_sd=1.,simplex=True)
   z,p=apply_np(X,w);w.update(z_mean=z.mean(0),z_sd=np.maximum(z.std(0),1e-10),p_mean=float(p.mean()),p_sd=max(float(p.std()),1e-10));np.savez_compressed(dest/f'seed{seed}.npz',**w)
  for case in ['metals','mixed_oxides']:
   tar=np.load(R/f'data/target_{case}.npz');tel=list(tar['input_names']);T=tar['X'];allowed=set(els)
   ok=np.array([all(e in allowed for e,v in zip(tel,x)if v>0)for x in T]);mapped=np.zeros((len(T),len(els)))
   for j,e in enumerate(els):
    if e in tel:mapped[:,j]=T[:,tel.index(e)]
   assert np.allclose(mapped[ok].sum(1),1)
   un=sorted(set(els)|set(tel));SX=np.zeros((len(X),len(un)));TX=np.zeros((len(T),len(un)))
   for j,e in enumerate(els):SX[:,un.index(e)]=X[:,j]
   for j,e in enumerate(tel):TX[:,un.index(e)]=T[:,j]
   distance=cdist(TX,SX,'cityblock').min(1)
   for kind in ['NN','YAE','YVAE','random']:
    for seed in SEEDS:
     w=dict(np.load(R/f'models/{sid}/{kind}/seed{seed}.npz'));z=np.zeros((len(T),8));p=np.zeros((len(T),1));physical=np.full(len(T),np.nan)
     zz,pp=apply_np(mapped[ok],w);z[ok]=(zz-w['z_mean'])/w['z_sd'];p[ok,0]=(pp-w['p_mean'])/w['p_sd'];physical[ok]=pp
     out=R/f'features/{case}/{kind}/seed{seed}';out.mkdir(parents=True,exist_ok=True)
     np.savez_compressed(out/f'{sid}.npz',z=z,p=p,mask=ok.astype(float),distance=distance,physical_pred=physical)
     if seed==101:
      m=Network(X.shape[1],kind if kind!='random'else'NN').double();m.load_state_dict({k:torch.tensor(w[k])for k in m.state_dict()});m.eval()
      with torch.no_grad():pt,zt,_,_=m(torch.from_numpy(X[:7]),True,False)
      zn,pn=apply_np(X[:7],w);checks.append(dict(source=sid,case=case,kind=kind,max_z_diff=float(abs(zn-zt.numpy()).max()),max_p_diff=float(abs(pn-(pt.numpy()*w['y_sd']+w['y_mean'])).max())))
 with open(R/'results/neural_checks.csv','w',newline='')as f:w=csv.DictWriter(f,checks[0]);w.writeheader();w.writerows(checks)
 assert max(c['max_z_diff']for c in checks)<1e-10

def main():
 jobs=[(i,k)for i in range(len(CAT))for k in ['NN','YAE','YVAE']];rows=[]
 with ProcessPoolExecutor(max_workers=4)as ex:
  fut=[ex.submit(work,j)for j in jobs]
  for f in as_completed(fut):
   row=f.result();rows.append(row);print(row,flush=True)
 with open(R/'results/source_performance.csv','w',newline='')as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
 randoms_features();print('SOURCE_BANK_AND_FEATURES_DONE',flush=True)
if __name__=='__main__':main()

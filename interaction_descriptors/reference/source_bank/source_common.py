"""Source-only nested group validation and frozen feature generation.
No target y is loaded in this script; only target X is used for applying saved models.
"""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import json, csv, sys, time, hashlib
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import GroupKFold
from concurrent.futures import ProcessPoolExecutor,as_completed
R=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
EPOCHS=[100,200,400];SEEDS=[101,202,303]
class Network(nn.Module):
 def __init__(self,p,kind):
  super().__init__();self.kind=kind
  self.trunk=nn.Sequential(nn.Linear(p,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
  self.mu=nn.Linear(16,8);self.logvar=nn.Linear(16,8)
  self.head=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,1))
  self.decoder=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,32),nn.SiLU(),nn.Linear(32,p))
 def forward(self,x,simplex,sample=False):
  h=self.trunk(x);mu=self.mu(h);lv=self.logvar(h).clamp(-8,6)
  z=mu+torch.exp(.5*lv)*torch.randn_like(mu)if self.kind=='YVAE'and sample else mu
  rec=self.decoder(z)
  if simplex:rec=torch.softmax(rec,dim=1)
  return self.head(mu).ravel(),mu,lv,rec

def fit_at(X,y,train,test,kind,seed,simplex,epochs):
 torch.manual_seed(int(seed));model=Network(X.shape[1],kind).double()
 xm=np.zeros(X.shape[1])if simplex else X[train].mean(0);xs=np.ones(X.shape[1])if simplex else X[train].std(0);xs[xs<1e-10]=1.
 ym=y[train].mean();ys=max(y[train].std(),1e-10)
 a=torch.from_numpy((X[train]-xm)/xs);b=torch.from_numpy((X[test]-xm)/xs);yy=torch.from_numpy((y[train]-ym)/ys)
 norm=float(((a-a.mean(0))**2).mean());norm=max(norm,1e-12)
 opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-4);out={}
 for epoch in range(1,max(epochs)+1):
  model.train();opt.zero_grad();pred,mu,lv,rec=model(a,simplex,True)
  loss=((pred-yy)**2).mean()
  if kind!='NN':loss=loss+.3*((rec-a)**2).mean()/norm
  if kind=='YVAE':loss=loss+.001*min(1,epoch/100)*(-.5*(1+lv-mu**2-lv.exp())).mean()
  if not torch.isfinite(loss):raise FloatingPointError('nonfinite source loss')
  loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
  if epoch in epochs:
   model.eval()
   with torch.no_grad():p,z,lv,rr=model(b,simplex,False)
   den=float(((b-a.mean(0))**2).sum());num=float(((rr-b)**2).sum())
   out[epoch]={'pred':p.numpy()*ys+ym,'rec_skill':None if kind=='NN'else 1-num/max(den,1e-12)}
 return model,dict(x_mean=xm,x_sd=xs,y_mean=ym,y_sd=ys,simplex=bool(simplex)),out

def work(args):
 source,prop,kind=args;t=time.time();A=np.load(R/f'data/source_{source}.npz');X=A['X'].astype(float);Y=A['Y'][:,prop].astype(float);g=A['groups'];simplex=bool(A['reconstruct_simplex']);trans=str(A['transforms'][prop]);y=np.log1p(Y)if trans=='log1p'else Y
 name=str(A['properties'][prop]);root=R/f'models/{source}/{name}/{kind}';root.mkdir(parents=True,exist_ok=True)
 po=np.zeros(len(y));rs=[];foldmeta=[]
 for fi,(tr,te)in enumerate(GroupKFold(3,shuffle=True,random_state=35400).split(X,groups=g)):
  losses=np.zeros(len(EPOCHS));nv=0
  for ii,(a,b)in enumerate(GroupKFold(2,shuffle=True,random_state=35410+fi).split(X[tr],groups=g[tr])):
   _,_,res=fit_at(X,y,tr[a],tr[b],kind,350100+prop*10000+fi*10+ii,simplex,EPOCHS)
   for j,e in enumerate(EPOCHS):losses[j]+=((res[e]['pred']-y[tr[b]])**2).sum()
   nv+=len(b)
  ep=EPOCHS[int(np.argmin(losses))];_,_,res=fit_at(X,y,tr,te,kind,351100+prop*10000+fi,simplex,[ep]);po[te]=res[ep]['pred'];rs.append(res[ep]['rec_skill']);foldmeta.append({'fold':fi,'epoch':ep,'train':tr.tolist(),'test':te.tolist(),'inner_mse':(losses/nv).tolist()})
 losses=np.zeros(len(EPOCHS));nv=0
 for fi,(tr,te)in enumerate(GroupKFold(3,shuffle=True,random_state=35500).split(X,groups=g)):
  _,_,res=fit_at(X,y,tr,te,kind,352100+prop*10000+fi,simplex,EPOCHS)
  for j,e in enumerate(EPOCHS):losses[j]+=((res[e]['pred']-y[te])**2).sum()
  nv+=len(te)
 ep=EPOCHS[int(np.argmin(losses))]
 for seed in SEEDS:
  m,pre,_=fit_at(X,y,np.arange(len(y)),np.arange(min(4,len(y))),kind,seed+prop*10000+(70000 if source=='polymer'else 80000 if source=='metals'else 90000),simplex,[ep])
  np.savez_compressed(root/f'seed_{seed}.npz',**{k:v.detach().numpy()for k,v in m.state_dict().items()},**pre)
 stats={'source':source,'property':name,'kind':kind,'n':len(y),'p':X.shape[1],'groups':len(set(g)),'transform':trans,'source_nested_rmse':float(np.sqrt(np.mean((po-y)**2))),'source_nested_q2':float(1-np.sum((po-y)**2)/np.sum((y-y.mean())**2)),'reconstruction_skill_mean':None if kind=='NN'else float(np.mean(rs)),'final_epoch':ep,'final_source_cv_mse':(losses/nv).tolist(),'elapsed':time.time()-t,'folds':foldmeta}
 (root/'meta.json').write_text(json.dumps(stats,indent=2));np.savez_compressed(root/'source_oof.npz',true=y,pred=po)
 return stats

def apply_np(X,w):
 from scipy.special import expit
 a=(X-w['x_mean'])/w['x_sd'];a=a@w['trunk.0.weight'].T+w['trunk.0.bias'];a=a*expit(a);a=a@w['trunk.2.weight'].T+w['trunk.2.bias'];a=a*expit(a);z=a@w['mu.weight'].T+w['mu.bias'];h=z@w['head.0.weight'].T+w['head.0.bias'];h=h*expit(h);p=(h@w['head.2.weight'].T+w['head.2.bias']).ravel()*w['y_sd']+w['y_mean'];return z,p

def features():
 (R/'features').mkdir(exist_ok=True);checks=[]
 for source in ['metals','mixed_oxides','polymer']:
  a=np.load(R/f'data/source_{source}.npz');X=a['X'].astype(float);simplex=bool(a['reconstruct_simplex'])
  for i,p in enumerate(a['properties']):
   root=R/f'models/{source}/{p}/random';root.mkdir(parents=True,exist_ok=True)
   for seed in SEEDS:
    torch.manual_seed(seed+500000+i*10000+len(X));m=Network(X.shape[1],'NN').double();xm=np.zeros(X.shape[1])if simplex else X.mean(0);xs=np.ones(X.shape[1])if simplex else X.std(0);xs[xs<1e-10]=1
    np.savez_compressed(root/f'seed_{seed}.npz',**{k:v.detach().numpy()for k,v in m.state_dict().items()},x_mean=xm,x_sd=xs,y_mean=0.,y_sd=1.,simplex=simplex)
  cases=['polymer_RI','polymer_Cp']if source=='polymer'else[source]
  for case in cases:
   tx=np.load(R/f'data/target_{case}.npz')['X'].astype(float)
   for kind in ['NN','YAE','YVAE','random']:
    for seed in SEEDS:
     zs=[];ps=[]
     for p in a['properties']:
      w=dict(np.load(R/f'models/{source}/{p}/{kind}/seed_{seed}.npz'));z,pr=apply_np(tx,w);zs.append(z);ps.append(pr)
      model=Network(X.shape[1],kind if kind!='random'else'NN').double();model.load_state_dict({k:torch.tensor(w[k])for k in model.state_dict()});model.eval()
      with torch.no_grad(): pp,zz,_,_=model(torch.tensor((tx[:8]-w['x_mean'])/w['x_sd']),simplex,False)
      checks.append({'case':case,'kind':kind,'seed':seed,'property':str(p),'latent_diff':float(np.max(abs(z[:8]-zz.numpy()))),'pred_diff':float(np.max(abs(pr[:8]-(pp.numpy()*w['y_sd']+w['y_mean']))))})
     zz=np.column_stack(zs);pr=np.column_stack(ps);np.savez_compressed(R/f'features/{case}_{kind}_{seed}.npz',latent=zz,pred=pr,both=np.column_stack([zz,pr]))
 with (R/'results/neural_checks.csv').open('w',newline='')as f:w=csv.DictWriter(f,checks[0]);w.writeheader();w.writerows(checks)
 assert max(c['latent_diff']for c in checks)<1e-9

def main():
 tasks=[]
 for source in ['metals','mixed_oxides','polymer']:
  a=np.load(R/f'data/source_{source}.npz');tasks += [(source,i,kind)for i in range(a['Y'].shape[1])for kind in ['NN','YAE','YVAE']]
 rows=[]
 with ProcessPoolExecutor(max_workers=4)as ex:
  fut=[ex.submit(work,a)for a in tasks]
  for f in as_completed(fut):
   s=f.result();rows.append({k:v for k,v in s.items()if k not in ['folds','final_source_cv_mse']});print(rows[-1],flush=True)
 with (R/'results/source_performance.csv').open('w',newline='')as f:w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
 features();print('SOURCE AND FEATURES COMPLETE',flush=True)
if __name__=='__main__':main()

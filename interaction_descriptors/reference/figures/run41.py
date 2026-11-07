"""Strict held-out source ratio assessment. No historical model is reused or edited.
Run after prepare41.py. Outputs per-point predictions, independent model weights,
inner-validation losses, split audits and deterministic controls.
"""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,json,csv,hashlib,time,sys,itertools,traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import torch
from torch import nn
R=Path(__file__).resolve().parents[1]
EPOCHS=[100,200,400];SEEDS=[101,202,303];KINDS=['NN','YAE','YVAE'];SOURCES=['modulus','stress','strain','hardness','Tc']
torch.set_num_threads(1)
try:torch.set_num_interop_threads(1)
except RuntimeError:pass

def write_csv(p,rows):
 rows=list(rows)
 if not rows:return
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8-sig')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def group_binary(X,idx):
 d=defaultdict(list)
 for k in idx:
  p=tuple(map(int,np.flatnonzero(X[k]>1e-12)))
  if len(p)==2:d[p].append(int(k))
 return d

class Network(nn.Module):
 def __init__(self,p,kind):
  super().__init__();self.kind=kind
  self.trunk=nn.Sequential(nn.Linear(p,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
  self.mu=nn.Linear(16,8);self.logvar=nn.Linear(16,8)
  self.head=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,1))
  self.decoder=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,32),nn.SiLU(),nn.Linear(32,p))
 def forward(self,x,sample=False):
  h=self.trunk(x);mu=self.mu(h);lv=self.logvar(h).clamp(-8,6)
  z=mu+torch.exp(.5*lv)*torch.randn_like(mu)if self.kind=='YVAE'and sample else mu
  rec=torch.softmax(self.decoder(z),dim=1)
  return self.head(mu).ravel(),mu,lv,rec

def fit(X,y,train,valid,kind,seed,epochs,shuffle=False):
 torch.manual_seed(int(seed));m=Network(X.shape[1],kind).double()
 tr=np.array(train,int);va=np.array(valid,int);yt=y[tr].copy();permchanges=0
 if shuffle:
  rng=np.random.default_rng(seed+910000)
  for _,ids in sorted(group_binary(X,tr).items()):
   pos=np.array([int(np.flatnonzero(tr==i)[0]) for i in ids]);order=rng.permutation(len(pos))
   prev=yt[pos].copy();yt[pos]=prev[order];permchanges+=int(np.sum(abs(prev-yt[pos])>1e-12))
 ym=float(yt.mean());ys=max(float(yt.std()),1e-10)
 a=torch.tensor(X[tr],dtype=torch.float64);b=torch.tensor(X[va],dtype=torch.float64);yy=torch.tensor((yt-ym)/ys,dtype=torch.float64)
 norm=max(float(((a-a.mean(0))**2).mean()),1e-12)
 opt=torch.optim.Adam(m.parameters(),lr=.003,weight_decay=1e-4)
 hist=[];preds={};maxepoch=max(epochs)
 for ep in range(1,maxepoch+1):
  m.train();opt.zero_grad();pred,mu,lv,rec=m(a,True)
  yl=((pred-yy)**2).mean();rl=((rec-a)**2).mean()/norm
  kl=(-.5*(1+lv-mu**2-lv.exp())).mean()
  loss=yl
  if kind!='NN':loss=loss+.3*rl
  if kind=='YVAE':loss=loss+.001*min(1,ep/100)*kl
  if not torch.isfinite(loss):raise FloatingPointError('non-finite training loss')
  loss.backward();nn.utils.clip_grad_norm_(m.parameters(),10);opt.step()
  if ep in epochs:
   m.eval()
   with torch.no_grad():pp,z,_,rr=m(b);pt,_,_,_=m(a)
   preds[ep]=pp.numpy()*ys+ym
   hist.append({'epoch':ep,'train_y_mse_standardized':float(((pt-yy)**2).mean()),'reconstruction_loss':float(rl.detach()),'kl':float(kl.detach())})
 return m,{'y_mean':ym,'y_sd':ys,'train_indices':tr,'permuted_observation_count':permchanges},preds,hist

def eval_np(X,w):
 from scipy.special import expit
 a=np.asarray(X,float)
 for k in [0,2]:
  a=a@w[f'trunk.{k}.weight'].T+w[f'trunk.{k}.bias'];a=a*expit(a)
 z=a@w['mu.weight'].T+w['mu.bias'];a=z@w['head.0.weight'].T+w['head.0.bias'];a=a*expit(a)
 y=(a@w['head.2.weight'].T+w['head.2.bias']).ravel()*float(w['y_sd'])+float(w['y_mean'])
 return y,z

def make_rows(sp,X,y,ele,method,seed,pred,vertices,epoch,initseed,modelpath):
 pure={int(np.argmax(x)):i for i,x in enumerate(X)if (x>1e-12).sum()==1}
 tr=sp['train'];rows=[]
 for idx,p in zip(sp['test'],pred):
  active=np.flatnonzero(X[idx]>1e-12);i,j=map(int,active);t=float(X[idx,i]);lo=t*y[pure[i]]+(1-t)*y[pure[j]]
  lp=t*vertices[i]+(1-t)*vertices[j]
  row={'source':sp['source'],'outer_id':sp['outer_id'],'repeat':sp['repeat'],'method':method,'seed':seed,'init_seed':initseed,'epoch':epoch,'index':idx,'pair':str(ele[i])+'--'+str(ele[j]),'i':i,'j':j,'fraction_i':t,'n_train':len(tr),'n_test':len(sp['test']),'true_y':float(y[idx]),'pred_y':float(p),'observed_linear':float(lo),'predicted_linear':float(lp),'true_delta':float(y[idx]-lo),'pred_delta':float(p-lp),'pred_i':float(vertices[i]),'pred_j':float(vertices[j]),'observed_i':float(y[pure[i]]),'observed_j':float(y[pure[j]]),'model_file':modelpath}
  assert all(np.isfinite(float(row[k]))for k in ['pred_y','true_delta','pred_delta'])
  rows.append(row)
 return rows

def controls(sp):
 a=np.load(R/f'data/source_{sp["source"]}.npz');X=a['X'];y=a['y'];ele=a['elements'];tr=sp['train'];te=sp['test'];p=X.shape[1]
 pure={int(np.argmax(x)):k for k,x in enumerate(X)if (x>1e-12).sum()==1};v=np.zeros(p)
 for i,k in pure.items():v[i]=y[k]
 rows=[]
 pp=X[te]@v
 rows+=make_rows(sp,X,y,ele,'endpoint_linear',0,pp,v,0,0,'')
 coef=np.linalg.lstsq(X[tr],y[tr],rcond=1e-10)[0]
 rows+=make_rows(sp,X,y,ele,'global_linear',0,X[te]@coef,coef,0,0,'')
 if p==3:
  pairs=list(itertools.combinations(range(p),2))
  def phi(z):return np.c_[z,np.column_stack([z[:,i]*z[:,j]for i,j in pairs])]
  coef=np.linalg.lstsq(phi(X[tr]),y[tr],rcond=1e-10)[0]
  rows+=make_rows(sp,X,y,ele,'global_quadratic',0,phi(X[te])@coef,phi(np.eye(p))@coef,0,0,'')
 bg=group_binary(X,tr);predq=[];predloc=[]
 for k in te:
  i,j=map(int,np.flatnonzero(X[k]>1e-12));ids=bg[(i,j)];tt=X[ids,i];s=tt*(1-tt);d=y[ids]-tt*v[i]-(1-tt)*v[j]
  b=float(s@d/(s@s)) if s@s>1e-15 else 0
  t=X[k,i];predq.append(t*v[i]+(1-t)*v[j]+b*t*(1-t))
  points=sorted([(0.,v[j]),(1.,v[i])]+[(float(ti),float(yi))for ti,yi in zip(tt,y[ids])])
  px=np.array([v[0]for v in points]);py=np.array([v[1]for v in points]);predloc.append(float(np.interp(t,px,py)))
 rows+=make_rows(sp,X,y,ele,'anchored_quadratic',0,np.array(predq),v,0,0,'')
 rows+=make_rows(sp,X,y,ele,'local_linear',0,np.array(predloc),v,0,0,'')
 return rows

def worker(arg):
 sp,kind=arg;tic=time.time();src=sp['source'];a=np.load(R/f'data/source_{src}.npz');X=a['X'];y=a['y'];ele=a['elements'];p=X.shape[1]
 offset=100000*SOURCES.index(src)+sp['fold']*100
 scores=np.zeros(len(EPOCHS));nvalid=0;innerlog=[]
 for inn,fold in enumerate(sp['inner']):
  _,_,pr,hist=fit(X,y,fold['train'],fold['valid'],kind,4100000+offset+inn,EPOCHS)
  for n,e in enumerate(EPOCHS):
   se=float(((pr[e]-y[fold['valid']])**2).sum());scores[n]+=se
   innerlog.append({'source':src,'outer_id':sp['outer_id'],'kind':kind,'inner_fold':inn,'epoch':e,'n_valid':len(fold['valid']),'sse':se,'train_ids':';'.join(map(str,fold['train'])),'valid_ids':';'.join(map(str,fold['valid']))})
  nvalid+=len(fold['valid'])
 ep=EPOCHS[int(np.argmin(scores))]; rows=[];checks=[];trainlog=[]
 dest=R/f'models/{src}/{sp["outer_id"]}/{kind}';dest.mkdir(parents=True,exist_ok=True)
 for seed in SEEDS:
  init=4200000+offset+seed
  m,pre,_,hist=fit(X,y,sp['train'],sp['test'],kind,init,[ep])
  ww={k:v.detach().numpy()for k,v in m.state_dict().items()};ww.update(pre)
  path=dest/f'seed_{seed}.npz';np.savez_compressed(path,**ww)
  yp,z=eval_np(X[sp['test']],ww);vv,_=eval_np(np.eye(p),ww)
  with torch.no_grad():torchpred=m(torch.tensor(X[sp['test']]))[0].numpy()*pre['y_sd']+pre['y_mean']
  checks.append({'source':src,'outer_id':sp['outer_id'],'method':kind,'seed':seed,'max_numpy_torch_error':float(np.max(abs(yp-torchpred))),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  rows+=make_rows(sp,X,y,ele,kind,seed,yp,vv,ep,init,str(path.relative_to(R)))
  trainlog+=[{'source':src,'outer_id':sp['outer_id'],'method':kind,'seed':seed,**h}for h in hist]
  if kind=='NN':
   # Same initialization and training-label scale, no fitted weights.
   torch.manual_seed(init);random=Network(p,'NN').double();wr={k:v.detach().numpy()for k,v in random.state_dict().items()};wr.update({'y_mean':pre['y_mean'],'y_sd':pre['y_sd']})
   rp=dest/f'random_{seed}.npz';np.savez_compressed(rp,**wr)
   yp,_=eval_np(X[sp['test']],wr);vv,_=eval_np(np.eye(p),wr)
   rows+=make_rows(sp,X,y,ele,'random',seed,yp,vv,0,init,str(rp.relative_to(R)))
   sm,spr,_,shist=fit(X,y,sp['train'],sp['test'],'NN',init,[ep],shuffle=True)
   ws={k:v.detach().numpy()for k,v in sm.state_dict().items()};ws.update(spr)
   shp=dest/f'shuffled_{seed}.npz';np.savez_compressed(shp,**ws)
   yp,_=eval_np(X[sp['test']],ws);vv,_=eval_np(np.eye(p),ws)
   rows+=make_rows(sp,X,y,ele,'NN_shuffled',seed,yp,vv,ep,init,str(shp.relative_to(R)))
   trainlog+=[{'source':src,'outer_id':sp['outer_id'],'method':'NN_shuffled','seed':seed,**h}for h in shist]
 meta={'source':src,'outer_id':sp['outer_id'],'kind':kind,'chosen_epoch':ep,'inner_mse':(scores/nvalid).tolist(),'epoch_candidates':EPOCHS,'seconds':time.time()-tic}
 (dest/'selection.json').write_text(json.dumps(meta,indent=2))
 write_csv(dest/'outer_predictions.csv',rows);write_csv(dest/'inner_validation.csv',innerlog);write_csv(dest/'training.csv',trainlog);write_csv(dest/'independent_forward.csv',checks)
 return meta

def main():
 splits=json.loads((R/'data/splits.json').read_text());protocol=json.loads((R/'protocol.json').read_text());manifest=[]
 for name,h in protocol['data_hashes_sha256'].items():assert hashlib.sha256((R/'data'/name).read_bytes()).hexdigest()==h
 controls_rows=[]
 for sp in splits:controls_rows+=controls(sp)
 write_csv(R/'results/classical_predictions.csv',controls_rows)
 tasks=[(sp,k)for sp in splits for k in KINDS]
 workers=int(os.environ.get('WORKERS','4'));tic=time.time();done=0
 with ProcessPoolExecutor(max_workers=workers)as pool:
  futures=[pool.submit(worker,t)for t in tasks]
  for f in as_completed(futures):
   meta=f.result();manifest.append(meta);done+=1
   print(f'{done}/{len(tasks)} {meta["outer_id"]} {meta["kind"]} epoch={meta["chosen_epoch"]} seconds={meta["seconds"]:.1f} elapsed={time.time()-tic:.1f}',flush=True)
 write_csv(R/'results/run_manifest.csv',[{**x,'inner_mse':json.dumps(x['inner_mse']),'epoch_candidates':json.dumps(x['epoch_candidates'])}for x in manifest])
 print('COMPLETE',len(tasks),time.time()-tic,flush=True)
if __name__=='__main__':main()

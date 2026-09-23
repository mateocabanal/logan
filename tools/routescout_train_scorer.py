#!/usr/bin/env python3
import argparse, math, os, struct
import numpy as np
from routescout_analyze import parse
from routescout_lowrank import vec, fit_maps, truncate

def norm_nonneg(x):
    x=np.maximum(x,0).astype(np.float32,copy=False)
    s=float(x.sum())
    if s>0: x=x/s
    return x

def route_meta(ev, experts):
    route=ev[5]
    v=vec(route,experts)
    return v, float(ev[2])/math.log(experts), float(ev[3])*10.0, max((w for _,w in route),default=0.0)

def features_for(cy, prev, l, spatial_maps, priors, experts):
    prevv, pent, pmargin, ptop1 = route_meta(prev[l],experts)
    prior=priors[l]
    if l>0:
        srcv, sent, smargin, stop1 = route_meta(cy[l-1],experts)
        spatial=norm_nonneg(srcv @ spatial_maps[l])
    else:
        spatial=np.zeros(experts,np.float32)
        sent=smargin=stop1=0.0
    e=np.arange(experts,dtype=np.float32)
    prevflag=(prevv>0).astype(np.float32)
    X=np.empty((experts,16),np.float32)
    X[:,0]=prevv
    X[:,1]=prevflag
    X[:,2]=spatial
    X[:,3]=np.sqrt(spatial)
    X[:,4]=prior
    X[:,5]=sent
    X[:,6]=smargin
    X[:,7]=stop1
    X[:,8]=pent
    X[:,9]=pmargin
    X[:,10]=ptop1
    X[:,11]=l/max(1,len(cy)-1)
    X[:,12]=(l%4)/3.0
    X[:,13]=e/max(1,experts-1)
    X[:,14]=spatial*(1.0-prevflag)
    X[:,15]=1.0
    actual={ex for ex,_ in cy[l][5]}
    y=np.fromiter((1.0 if i in actual else 0.0 for i in range(experts)),dtype=np.float32,count=experts)
    cold=y*(1.0-prevflag)
    return X,y,cold,prevflag

def relu(x): return np.maximum(x,0)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))

class MLP:
    def __init__(self,seed=7):
        r=np.random.default_rng(seed)
        self.W0=(r.standard_normal((16,16))*math.sqrt(2/16)).astype(np.float32)
        self.W1=(r.standard_normal((8,16))*math.sqrt(2/16)).astype(np.float32)
        self.W2=(r.standard_normal((1,8))*math.sqrt(2/8)).astype(np.float32)
        self.params=[self.W0,self.W1,self.W2]
        self.m=[np.zeros_like(p) for p in self.params]
        self.v=[np.zeros_like(p) for p in self.params]
        self.step=0
    def forward(self,X):
        z0=X@self.W0.T; h0=relu(z0)
        z1=h0@self.W1.T; h1=relu(z1)
        z2=h1@self.W2.T
        return z2[:,0],(X,z0,h0,z1,h1)
    def train_batch(self,X,y,weights,lr=2e-3):
        logits,c=self.forward(X); p=sigmoid(logits)
        denom=max(1e-6,float(weights.sum()))
        dz=((p-y)*weights/denom)[:,None]
        X,z0,h0,z1,h1=c
        g2=dz.T@h1
        dh1=dz@self.W2
        dz1=dh1*(z1>0)
        g1=dz1.T@h0
        dh0=dz1@self.W1
        dz0=dh0*(z0>0)
        g0=dz0.T@X
        grads=[g0,g1,g2]
        self.step+=1
        b1,b2=.9,.999
        for i,(p0,g) in enumerate(zip(self.params,grads)):
            np.clip(g,-2,2,out=g)
            self.m[i]=b1*self.m[i]+(1-b1)*g
            self.v[i]=b2*self.v[i]+(1-b2)*(g*g)
            mh=self.m[i]/(1-b1**self.step); vh=self.v[i]/(1-b2**self.step)
            p0-=lr*mh/(np.sqrt(vh)+1e-8)
        eps=1e-7
        loss=-(weights*(y*np.log(p+eps)+(1-y)*np.log(1-p+eps))).sum()/denom
        return float(loss)
    def scores(self,X): return self.forward(X)[0]

def make_priors(train,layers,experts):
    out=[]
    for l in range(layers):
        p=np.zeros(experts,np.float32)
        for cy in train: p+=vec(cy[l][5],experts)
        out.append(norm_nonneg(p))
    return out

def build_train(cycles,split,maps,priors,layers,experts,neg_per=32,seed=3):
    rng=np.random.default_rng(seed); xs=[]; ys=[]; ws=[]
    for t in range(1,split):
        cy,prev=cycles[t],cycles[t-1]
        for l in range(layers):
            X,y,cold,prevflag=features_for(cy,prev,l,maps,priors,experts)
            pos=np.flatnonzero(y>0)
            neg=np.flatnonzero(y==0)
            choose=rng.choice(neg,size=min(neg_per,len(neg)),replace=False)
            ids=np.concatenate([pos,choose])
            xs.append(X[ids]); ys.append(y[ids])
            # Cold positives matter most; resident positives still teach demand.
            w=np.ones(len(ids),np.float32)
            w[:len(pos)]=np.where(cold[pos]>0,6.0,2.0)
            ws.append(w)
    return np.concatenate(xs),np.concatenate(ys),np.concatenate(ws)

def evaluate(model,cycles,split,maps,priors,layers,experts):
    totals={k:[0,0] for k in (8,16,24)}
    cold={k:[0,0,0] for k in (4,8,12,16,24)} # hit,total,issued
    for t in range(split,len(cycles)):
        cy,prev=cycles[t],cycles[t-1]
        for l in range(layers):
            X,y,cold_y,prevflag=features_for(cy,prev,l,maps,priors,experts)
            s=model.scores(X)
            actual=set(np.flatnonzero(y>0).tolist())
            for k in totals:
                ids=np.argpartition(s,-k)[-k:]
                totals[k][0]+=len(set(map(int,ids))&actual); totals[k][1]+=len(actual)
            if l>0:
                arrivals=set(np.flatnonzero(cold_y>0).tolist())
                sc=s.copy(); sc[prevflag>0]=-1e9
                for k in cold:
                    ids=np.argpartition(sc,-k)[-k:]
                    cold[k][0]+=len(set(map(int,ids))&arrivals); cold[k][1]+=len(arrivals); cold[k][2]+=k
    print("held-out learned scorer")
    for k,(h,d) in totals.items(): print(f"Recall@{k}: {h/max(1,d):.4f} ({h}/{d})")
    for k,(h,d,n) in cold.items(): print(f"Cold@{k}: recall={h/max(1,d):.4f} precision={h/max(1,n):.4f} ({h}/{d})")

def save_fp16(model,outdir):
    os.makedirs(outdir,exist_ok=True)
    for name,w in [("w0",model.W0),("w1",model.W1),("w2",model.W2)]:
        a=w.astype(np.float16)
        a.tofile(os.path.join(outdir,name+".f16"))
    with open(os.path.join(outdir,"meta.txt"),"w") as f:
        f.write("routescout-scorer-v1\n16 16 8 1\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--epochs",type=int,default=35)
    ap.add_argument("--train-frac",type=float,default=.7)
    ap.add_argument("--out",default=".perf_runs/routescout-scorer-v1")
    a=ap.parse_args()
    layers,experts,topk,cycles=parse(a.trace)
    split=max(4,min(len(cycles)-1,int(len(cycles)*a.train_frac)))
    train=cycles[:split]
    priors=make_priors(train,layers,experts)
    sm=fit_maps(train,layers,experts,"spatial")
    maps=[truncate(m,16) if l else m for l,m in enumerate(sm)]
    X,y,w=build_train(cycles,split,maps,priors,layers,experts)
    print(f"cycles={len(cycles)} split={split} train_rows={len(y)} positives={int(y.sum())}")
    model=MLP()
    rng=np.random.default_rng(11); batch=4096
    for ep in range(a.epochs):
        order=rng.permutation(len(y)); losses=[]
        for at in range(0,len(order),batch):
            ids=order[at:at+batch]
            losses.append(model.train_batch(X[ids],y[ids],w[ids]))
        if ep==0 or (ep+1)%5==0:
            print(f"epoch {ep+1:02d} loss={np.mean(losses):.5f}")
    evaluate(model,cycles,split,maps,priors,layers,experts)
    save_fp16(model,a.out)
    # Persist low-rank maps and priors for future runtime feature construction.
    np.save(os.path.join(a.out,"spatial_rank16.npy"),np.stack(maps))
    np.save(os.path.join(a.out,"priors.npy"),np.stack(priors))
    print("saved",a.out)

if __name__=="__main__": main()

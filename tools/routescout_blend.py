#!/usr/bin/env python3
import argparse,itertools
import numpy as np
from routescout_analyze import parse
from routescout_lowrank import vec,fit_maps,truncate

def norm(x):
    x=np.maximum(x,0)
    s=float(x.sum())
    return x/s if s>0 else x

def rec(score,actual,k):
    p=np.argpartition(score,-k)[-k:]
    a={e for e,_ in actual}
    return sum(int(int(e) in a) for e in p),len(a)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--rank",type=int,default=16)
    ap.add_argument("--train-frac",type=float,default=.7)
    a=ap.parse_args()
    layers,experts,topk,cycles=parse(a.trace)
    split=max(3,min(len(cycles)-1,int(len(cycles)*a.train_frac)))
    train,test=cycles[:split],cycles[split:]
    maps=fit_maps(train,layers,experts,"spatial")
    maps=[truncate(m,a.rank) if l else m for l,m in enumerate(maps)]
    priors=[]
    for l in range(layers):
        p=np.zeros(experts,np.float32)
        for cy in train: p+=vec(cy[l][5],experts)
        priors.append(norm(p))

    samples=[]
    for ti,cy in enumerate(test):
        gt=split+ti
        prev=cycles[gt-1]
        for l in range(layers):
            temporal=vec(prev[l][5],experts)
            spatial=np.zeros(experts,np.float32) if l==0 else norm(vec(cy[l-1][5],experts)@maps[l])
            samples.append((temporal,spatial,priors[l],cy[l][5]))

    grid=[0,.125,.25,.5,1,2,4,8]
    pg=[0,.125,.25,.5,1]
    for k in (8,16,24):
        best=None
        for at,asp,aprior in itertools.product(grid,grid,pg):
            if at==asp==aprior==0: continue
            hit=den=0
            for temporal,spatial,prior,actual in samples:
                score=at*temporal+asp*spatial+aprior*prior
                h,d=rec(score,actual,k); hit+=h; den+=d
            r=hit/max(1,den)
            if best is None or r>best[0]: best=(r,at,asp,aprior)
        print(f"best recall@{k}={best[0]:.4f} temporal={best[1]} spatial={best[2]} prior={best[3]}")
    # Fixed interpretable mixes for comparison.
    for combo in [(1,0,0),(0,1,0),(1,1,0),(2,1,.25),(1,2,.25)]:
        print("mix",combo,end="")
        for k in (8,16,24):
            hit=den=0
            for temporal,spatial,prior,actual in samples:
                score=combo[0]*temporal+combo[1]*spatial+combo[2]*prior
                h,d=rec(score,actual,k); hit+=h; den+=d
            print(f" R@{k}={hit/max(1,den):.4f}",end="")
        print()

if __name__=="__main__": main()

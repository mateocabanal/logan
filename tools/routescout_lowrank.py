#!/usr/bin/env python3
import argparse
import numpy as np
from routescout_analyze import parse

def vec(route,n):
    x=np.zeros(n,np.float32)
    for e,w in route:
        x[e]=w
    s=x.sum()
    if s>0: x/=s
    return x

def recall(scores,actual,k):
    if k>=len(scores):
        pred=np.arange(len(scores))
    else:
        pred=np.argpartition(scores,-k)[-k:]
    aset={e for e,_ in actual}
    return sum(int(int(e) in aset) for e in pred),len(aset)

def fit_maps(cycles,layers,experts,kind):
    mats=[np.zeros((experts,experts),np.float32) for _ in range(layers)]
    counts=[0]*layers
    if kind=="spatial":
        for cy in cycles:
            for l in range(1,layers):
                a=vec(cy[l-1][5],experts); b=vec(cy[l][5],experts)
                mats[l]+=np.outer(a,b); counts[l]+=1
    elif kind=="temporal":
        for t in range(1,len(cycles)):
            for l in range(layers):
                a=vec(cycles[t-1][l][5],experts); b=vec(cycles[t][l][5],experts)
                mats[l]+=np.outer(a,b); counts[l]+=1
    else: raise ValueError(kind)
    return mats

def truncate(m,r):
    if r<=0 or r>=min(m.shape): return m
    u,s,vt=np.linalg.svd(m,full_matrices=False)
    return (u[:,:r]*s[:r])@vt[:r]

def evaluate(train,test,allcycles,split,layers,experts,topk,ranks):
    for kind in ("temporal","spatial"):
        raw=fit_maps(train,layers,experts,kind)
        print("\n"+kind)
        for rank in ranks:
            maps=[truncate(m,rank) if (kind=="temporal" or l>0) else m for l,m in enumerate(raw)]
            for k in (8,16,24):
                hit=den=0
                for ti,cy in enumerate(test):
                    gt=split+ti
                    for l in range(layers):
                        if kind=="spatial":
                            if l==0: continue
                            src=cy[l-1][5]
                        else:
                            if gt==0: continue
                            src=allcycles[gt-1][l][5]
                        scores=vec(src,experts)@maps[l]
                        h,d=recall(scores,cy[l][5],k)
                        hit+=h; den+=d
                print(f"rank={rank:>3} recall@{k}={hit/max(1,den):.4f}",end="  ")
            print()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--train-frac",type=float,default=.7)
    ap.add_argument("--ranks",default="4,8,16,32,0")
    a=ap.parse_args()
    layers,experts,topk,cycles=parse(a.trace)
    if len(cycles)<4: raise SystemExit("need >=4 complete cycles")
    split=max(2,min(len(cycles)-1,int(len(cycles)*a.train_frac)))
    train=cycles[:split]; test=cycles[split:]
    ranks=[int(x) for x in a.ranks.split(",")]
    print(f"cycles={len(cycles)} train={len(train)} test={len(test)} layers={layers} experts={experts}")
    evaluate(train,test,cycles,split,layers,experts,topk,ranks)

if __name__=="__main__": main()

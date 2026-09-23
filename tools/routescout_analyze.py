#!/usr/bin/env python3
import argparse
from collections import defaultdict

def parse(path):
    events=[]
    layers=experts=topk=None
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            if line.startswith("# routescout-v1"):
                fields=line.split("\t")
                meta={}
                for x in fields[1:]:
                    k,v=x.split("=",1)
                    meta[k]=int(v)
                layers,experts,topk=meta["layers"],meta["experts"],meta["topk"]
                continue
            parts=line.split("\t")
            event=int(parts[0]); layer=int(parts[1])
            entropy=float(parts[2]); margin=float(parts[3]); wsum=float(parts[4])
            route=[]
            for x in parts[5:]:
                e,w=x.split(":",1)
                route.append((int(e),float(w)))
            events.append((event,layer,entropy,margin,wsum,route))
    if layers is None:
        raise SystemExit("missing routescout header")
    cycles=[]
    cur=[]
    for ev in events:
        if ev[1]==0 and cur:
            if len(cur)==layers:
                cycles.append(cur)
            cur=[]
        cur.append(ev)
    if len(cur)==layers:
        cycles.append(cur)
    return layers,experts,topk,cycles

def overlap(a,b):
    A={e for e,_ in a}; B={e for e,_ in b}
    return len(A&B)

def top_from_scores(scores,k,forbid=()):
    f=set(forbid)
    items=[(s,e) for e,s in enumerate(scores) if e not in f and s>0]
    items.sort(key=lambda x:(-x[0],x[1]))
    return [e for _,e in items[:k]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--train-frac",type=float,default=.7)
    args=ap.parse_args()
    layers,experts,topk,cycles=parse(args.trace)
    print(f"cycles={len(cycles)} layers={layers} experts={experts} topk={topk}")
    if not cycles:
        return

    # Heuristic baselines.
    temporal_num=temporal_den=0
    spatial_num=spatial_den=0
    arrival_total=0
    for t in range(1,len(cycles)):
        for l in range(layers):
            temporal_num += overlap(cycles[t-1][l][5],cycles[t][l][5])
            temporal_den += topk
            prev={e for e,_ in cycles[t-1][l][5]}
            cur={e for e,_ in cycles[t][l][5]}
            arrival_total += len(cur-prev)
        for l in range(1,layers):
            spatial_num += overlap(cycles[t][l-1][5],cycles[t][l][5])
            spatial_den += topk
    print(f"previous-token same-layer recall@{topk}: {temporal_num/max(1,temporal_den):.4f}")
    print(f"previous-layer same-token recall@{topk}: {spatial_num/max(1,spatial_den):.4f}")
    print(f"mean cold arrivals/layer-token: {arrival_total/max(1,(len(cycles)-1)*layers):.3f}")

    if len(cycles)<3:
        return
    split=max(1,min(len(cycles)-1,int(len(cycles)*args.train_frac)))
    train=cycles[:split]; test=cycles[split:]
    print(f"train_cycles={len(train)} test_cycles={len(test)}")

    # Layer-local conditional maps:
    # temporal: previous token same-layer expert -> current target expert
    temporal=[[[0]*experts for _ in range(experts)] for _ in range(layers)]
    # spatial h=1: current layer-1 expert -> target layer expert
    spatial=[[[0]*experts for _ in range(experts)] for _ in range(layers)]
    for t in range(1,len(train)):
        for l in range(layers):
            src=[e for e,_ in train[t-1][l][5]]
            dst=[e for e,_ in train[t][l][5]]
            for a in src:
                row=temporal[l][a]
                for b in dst: row[b]+=1
            if l>0:
                src2=[e for e,_ in train[t][l-1][5]]
                for a in src2:
                    row=spatial[l][a]
                    for b in dst: row[b]+=1

    for kpred in (8,16,24):
        hit_t=den=0
        hit_s=0
        hit_h=0
        arrival_hit=arrival_den=0
        for ti,cy in enumerate(test):
            global_t=split+ti
            if global_t==0: continue
            prev=cycles[global_t-1]
            for l in range(layers):
                actual=[e for e,_ in cy[l][5]]
                src=[e for e,_ in prev[l][5]]
                sc=[0]*experts
                for a in src:
                    row=temporal[l][a]
                    for e,c in enumerate(row): sc[e]+=c
                pred=top_from_scores(sc,kpred)
                hit_t+=len(set(pred)&set(actual))
                den+=len(actual)

                # hybrid adds same-token previous-layer evidence when available
                hs=sc[:]
                if l>0:
                    ssrc=[e for e,_ in cy[l-1][5]]
                    ss=[0]*experts
                    for a in ssrc:
                        row=spatial[l][a]
                        for e,c in enumerate(row): ss[e]+=c
                    sp=top_from_scores(ss,kpred)
                    hit_s+=len(set(sp)&set(actual))
                    for e,v in enumerate(ss): hs[e]+=v
                else:
                    hit_s+=0
                hp=top_from_scores(hs,kpred)
                hit_h+=len(set(hp)&set(actual))

                prevset=set(src); arrivals=set(actual)-prevset
                ap=top_from_scores(sc,kpred,forbid=prevset)
                arrival_hit+=len(set(ap)&arrivals); arrival_den+=len(arrivals)
        print(f"Recall@{kpred}: temporal={hit_t/max(1,den):.4f} spatial_h1={hit_s/max(1,den):.4f} hybrid={hit_h/max(1,den):.4f} cold_arrival={arrival_hit/max(1,arrival_den):.4f}")

if __name__=="__main__":
    main()

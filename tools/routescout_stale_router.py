#!/usr/bin/env python3
import argparse, json, os, struct
import numpy as np
from routescout_analyze import parse

MAGIC=b"RSCHID1\0"

def read_hidden(path):
    with open(path,"rb") as f:
        magic=f.read(8)
        if magic!=MAGIC:
            raise ValueError(f"bad hidden magic {magic!r}")
        layers,hidden=struct.unpack("<II",f.read(8))
        rec_bytes=10+4*hidden
        data=f.read()
    n=len(data)//rec_bytes
    if len(data)!=n*rec_bytes:
        raise ValueError(f"truncated hidden trace: {len(data)} not multiple of {rec_bytes}")
    events=[]
    at=0
    for _ in range(n):
        event=struct.unpack_from("<Q",data,at)[0]; at+=8
        layer=struct.unpack_from("<H",data,at)[0]; at+=2
        x=np.frombuffer(data,dtype="<f4",count=hidden,offset=at).copy(); at+=4*hidden
        events.append((event,layer,x))
    return layers,hidden,events

def load_router(root,layer,hidden,experts):
    index=json.load(open(os.path.join(root,"model.safetensors.index.json")))["weight_map"]
    name=f"language_model.model.layers.{layer}.mlp.gate.weight"
    shard=index[name]
    path=os.path.join(root,shard)
    with open(path,"rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        hdr=json.loads(f.read(n))
        spec=hdr[name]
        if spec["dtype"]!="F16" or spec["shape"]!=[experts,hidden]:
            raise ValueError((name,spec["dtype"],spec["shape"]))
        begin,end=spec["data_offsets"]
        f.seek(8+n+begin)
        raw=f.read(end-begin)
    return np.frombuffer(raw,dtype="<f2").astype(np.float32).reshape(experts,hidden)

def topk(v,k):
    ids=np.argpartition(v,-k)[-k:]
    return ids[np.argsort(v[ids])[::-1]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("route_trace")
    ap.add_argument("hidden_trace")
    ap.add_argument("model")
    ap.add_argument("--max-horizon",type=int,default=4)
    a=ap.parse_args()
    layers,experts,k,cycles=parse(a.route_trace)
    hlayers,hidden,events=read_hidden(a.hidden_trace)
    if layers!=hlayers: raise SystemExit((layers,hlayers))
    complete=len(cycles)*layers
    if len(events)<complete:
        raise SystemExit(f"hidden events {len(events)} < route events {complete}")
    events=events[:complete]
    xs=np.stack([e[2] for e in events]).reshape(len(cycles),layers,hidden)
    routers=[load_router(a.model,l,hidden,experts) for l in range(layers)]
    print(f"cycles={len(cycles)} layers={layers} hidden={hidden} experts={experts} topk={k}")
    for h in range(1,a.max_horizon+1):
        hit=den=0
        cold_hit={q:0 for q in (4,8,12,16,24)}
        cold_den=0
        issued={q:0 for q in cold_hit}
        cosine=[]
        for t,cy in enumerate(cycles):
            for l in range(layers-h):
                target=l+h
                score=routers[target]@xs[t,l]
                actual={e for e,_ in cy[target][5]}
                pred=topk(score,k)
                hit+=len(set(map(int,pred))&actual); den+=len(actual)
                # Cold relative to the target layer's previous-token route.
                if t>0:
                    prev={e for e,_ in cycles[t-1][target][5]}
                    arrivals=actual-prev
                    cold_den+=len(arrivals)
                    s=score.copy()
                    if prev: s[list(prev)]=-np.inf
                    for q in cold_hit:
                        ids=topk(s,q)
                        cold_hit[q]+=len(set(map(int,ids))&arrivals)
                        issued[q]+=q
        print(f"h={h} stale-router Recall@{k}={hit/max(1,den):.4f}")
        if cold_den:
            print("   "+" ".join(
                f"Cold@{q}=R{cold_hit[q]/cold_den:.4f}/P{cold_hit[q]/issued[q]:.4f}"
                for q in cold_hit
            ))

if __name__=="__main__":
    main()

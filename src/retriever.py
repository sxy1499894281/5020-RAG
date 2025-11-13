#!/usr/bin/env python3
import argparse
import os
from typing import List, Dict, Optional

import yaml

try:
    from .index_bm25 import search_bm25
    from .index_dense import search_dense
except ImportError:  # allow running as a script
    from index_bm25 import search_bm25
    from index_dense import search_dense

def _load_config(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def _norm(scores: List[float]) -> List[float]:
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def _to_map(items: List[Dict]) -> Dict[str, Dict]:
    m = {}
    for d in items:
        m[d.get("id")] = d
    return m

def retrieve(query: str, topk: int = 5, mode: str = "bm25", alpha: float = 0.5, config_path: str = "configs/config.yaml") -> List[Dict]:
    cfg = _load_config(config_path)
    bm25_path = ((cfg.get("bm25") or {}).get("index_path")) or "./data/bm25.idx"
    dense_db = ((cfg.get("dense") or {}).get("db")) or "./data/chroma"
    dense_model = ((cfg.get("dense") or {}).get("model")) or "all-MiniLM-L6-v2"
    collection = ((cfg.get("dense") or {}).get("collection")) or "arxiv"

    if mode == "bm25":
        return search_bm25(bm25_path, query, topk)
    if mode == "dense":
        return search_dense(dense_db, query, topk, dense_model, collection=collection)

    b = search_bm25(bm25_path, query, max(topk, 10))
    d = search_dense(dense_db, query, max(topk, 10), dense_model, collection=collection)
    mb = _to_map(b)
    md = _to_map(d)
    ids = set(mb.keys()) | set(md.keys())
    sb = {k: v.get("score", 0.0) for k, v in mb.items()}
    sd = {k: v.get("score", 0.0) for k, v in md.items()}
    nb = _norm(list(sb.values()))
    nd = _norm(list(sd.values()))
    bn = dict(zip(list(sb.keys()), nb))
    dn = dict(zip(list(sd.keys()), nd))
    merged = []
    for _id in ids:
        vb = bn.get(_id, 0.0)
        vd = dn.get(_id, 0.0)
        s = alpha * vb + (1.0 - alpha) * vd
        base = mb.get(_id) or md.get(_id)
        merged.append({
            "id": _id,
            "title": base.get("title"),
            "abstract": base.get("abstract"),
            "text": base.get("text"),
            "score": float(s)
        })
    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:topk]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query", type=str)
    p.add_argument("--mode", default="hybrid")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()
    res = retrieve(args.query, args.topk, args.mode, args.alpha)
    print(res)

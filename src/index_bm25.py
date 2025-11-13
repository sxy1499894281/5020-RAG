#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import re
from typing import List, Dict

from rank_bm25 import BM25Okapi

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())

def build_bm25_index(clean_path: str, index_path: str) -> int:
    docs: List[Dict] = []
    tokens: List[List[str]] = []
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            id_ = rec.get("id") or ""
            title = rec.get("title") or ""
            abstract = rec.get("abstract") or ""
            text = f"{title}\n{abstract}".strip()
            docs.append({"id": id_, "title": title, "abstract": abstract, "text": text})
            tokens.append(_tokenize(text))
    data = {"version": 1, "docs": docs, "tokens": tokens}
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    with open(index_path, "wb") as fw:
        pickle.dump(data, fw, protocol=pickle.HIGHEST_PROTOCOL)
    return len(docs)

def search_bm25(index_path: str, query: str, topk: int = 5) -> List[Dict]:
    with open(index_path, "rb") as fr:
        data = pickle.load(fr)
    docs = data["docs"]
    corpus_tokens = data["tokens"]
    bm25 = BM25Okapi(corpus_tokens)
    q_tokens = _tokenize(query or "")
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(topk, 0)]
    out = []
    for i in idxs:
        d = docs[i]
        out.append({
            "id": d.get("id"),
            "title": d.get("title"),
            "abstract": d.get("abstract"),
            "text": d.get("text"),
            "score": float(scores[i]),
        })
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--index", dest="index_path", required=True)
    args = p.parse_args()
    n = build_bm25_index(args.in_path, args.index_path)
    print(f"Built BM25 index with {n} docs → {args.index_path}")

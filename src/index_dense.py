#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

def _load_docs(clean_path: str) -> List[Dict]:
    docs = []
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
            docs.append({"id": id_, "title": title, "text": text})
    return docs

def embed_and_build_vector_db(clean_path: str, db_path: str, model_name: str, collection: str = "arxiv", batch_size: int = 64) -> int:
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection(name=collection)
    except Exception:
        pass
    col = client.create_collection(name=collection, metadata={"hnsw:space": "cosine"})
    docs = _load_docs(clean_path)
    model = SentenceTransformer(model_name)
    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        ids = [d["id"] for d in batch]
        texts = [d["text"] for d in batch]
        metas = [{"title": d["title"]} for d in batch]
        embs = model.encode(texts, normalize_embeddings=True).tolist()
        col.add(ids=ids, documents=texts, metadatas=metas, embeddings=embs)
        total += len(batch)
    return total

def search_dense(db_path: str, query: str, topk: int, model_name: str, collection: str = "arxiv") -> List[Dict]:
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(name=collection)
    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_emb, n_results=topk, include=["documents", "metadatas", "distances"])
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i, _id in enumerate(ids):
        title = (metas[i] or {}).get("title") if i < len(metas) else None
        text = docs[i] if i < len(docs) else None
        dist = float(dists[i]) if i < len(dists) else 0.0
        score = 1.0 - dist
        out.append({"id": _id, "title": title, "text": text, "score": score})
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--db", dest="db_path", required=True)
    p.add_argument("--model", dest="model", required=True)
    p.add_argument("--collection", dest="collection", default="arxiv")
    args = p.parse_args()
    n = embed_and_build_vector_db(args.in_path, args.db_path, args.model, collection=args.collection)
    print(f"Built dense DB with {n} docs → {args.db_path} [{args.collection}]")

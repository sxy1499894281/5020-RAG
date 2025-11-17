#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_dense.py

功能概述：
- 用 sentence-transformers + Chroma 构建稠密向量索引（持久化到目录）。
- 从 clean.jsonl 加载文档，将 title+abstract 拼为 text 并编码，写入向量库。
- 提供 `search_dense(db_path, query, topk, model_name, collection)` 查询接口。

实现思路：
- 构建阶段：
  - 读取 clean.jsonl，得到 [{id,title,categories,text}]。
  - 使用 SentenceTransformer(model_name) 做向量化，建议 normalize_embeddings=True 便于余弦相似。
  - 使用 chromadb.PersistentClient(path=db_path)，get_or_create_collection(collection) 并 add。
- 查询阶段：
  - get_collection -> 将 query 编码，使用 col.query(query_embeddings=..., n_results=topk, include=[...])。
  - 将返回的 ids/documents/metadatas/distances 组装为统一字典列表，score 可取 1 - distance。

主要函数：
- `_load_docs(clean_path) -> List[Dict]`：从 clean.jsonl 加载并拼接 text。
- `embed_and_build_vector_db(clean_path, db_path, model_name, collection, batch_size)`：写入向量库。
- `search_dense(db_path, query, topk, model_name, collection)`：查询接口。

测试/使用：
- 构建：
  python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-en-v1.5 --collection arxiv
- 查询（示例，建议通过 retriever.py 统一入口）：
  from src.index_dense import search_dense
  res = search_dense('./data/chroma', 'contrastive learning', 5, 'bge-small-en-v1.5', 'arxiv')

注意事项：
- 第一次构建后，重复 add 相同 id 会报错；如需重建请删除 db 目录或更换 collection 名称。
"""

import argparse
import json
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


def _load_docs(clean_path: str) -> List[Dict]:
    """从 clean.jsonl 加载文档，返回 [{"id","title","categories","text"}, ...]。"""
    docs: List[Dict] = []
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            _id = rec.get("id")
            title = rec.get("title") or ""
            abstract = rec.get("abstract") or ""
            categories = rec.get("categories") or ""
            if not _id or not title:
                continue
            text = (title + "\n" + abstract).strip()
            docs.append({"id": str(_id), "title": title, "categories": categories, "text": text})
    return docs


def embed_and_build_vector_db(
    clean_path: str,
    db_path: str,
    model_name: str,
    collection: str = "arxiv",
    batch_size: int = 64,
) -> int:
    """从 clean.jsonl 构建稠密向量库，返回写入的文档数量。"""
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_or_create_collection(collection_name=collection)

    docs = _load_docs(clean_path)
    if not docs:
        print("No docs loaded from clean file; nothing to index.")
        return 0

    model = SentenceTransformer(model_name)

    total = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        texts = [d["text"] for d in batch]
        emb = model.encode(texts, normalize_embeddings=True)
        ids = [d["id"] for d in batch]
        metas = [{"title": d["title"], "categories": d.get("categories", "")} for d in batch]
        try:
            col.add(ids=ids, documents=texts, metadatas=metas, embeddings=emb)
        except Exception as e:
            # 如果重复添加导致报错，可忽略已存在的 ids
            print(f"Warning: add to collection failed at [{i}:{i+len(batch)}]: {e}")
        total += len(batch)

    print(f"Built dense DB with {total} docs -> {db_path} [{collection}] using {model_name}")
    return total


def search_dense(
    db_path: str,
    query: str,
    topk: int,
    model_name: str,
    collection: str = "arxiv",
) -> List[Dict]:
    """用稠密向量检索 query，返回 Top-k 文档：[{id,title,text,categories,score}]。"""
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(collection_name=collection)

    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], normalize_embeddings=True)
    result = col.query(query_embeddings=q_emb, n_results=max(1, topk), include=["ids", "metadatas", "documents", "distances"])  # type: ignore

    docs: List[Dict] = []
    ids = result.get("ids", [[]])[0]
    docs_list = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    for _id, text, meta, dist in zip(ids, docs_list, metas, dists):
        title = (meta or {}).get("title", "")
        categories = (meta or {}).get("categories", "")
        score = float(1.0 - float(dist)) if dist is not None else 0.0
        docs.append({
            "id": str(_id),
            "title": title,
            "text": text,
            "categories": categories,
            "score": score,
        })
    return docs


def main():
    """命令行：
    python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-en-v1.5 --collection arxiv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--db", dest="db_path", required=True)
    parser.add_argument("--model", dest="model", required=True)
    parser.add_argument("--collection", dest="collection", default="arxiv")
    args = parser.parse_args()

    embed_and_build_vector_db(args.in_path, args.db_path, args.model, args.collection)


if __name__ == "__main__":
    main()

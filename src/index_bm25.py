#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
index_bm25.py

功能概述：
- 从 clean.jsonl 构建 BM25 索引（基于 rank-bm25 的 BM25Okapi）。
- 将语料的 tokens 与原始文档元信息序列化保存，便于后续快速重建 BM25 模型并进行查询。
- 提供 `search_bm25(index_path, query, topk)` 接口和命令行工具。

实现思路：
- 构建阶段：
  - 读取 clean.jsonl，每行包含 id/title/abstract/categories/created。
  - 将 title+abstract 拼接为 text，做简单英文分词，收集 tokens 列表。
  - 序列化保存：{"docs": [...], "tokens": [...]} 到 pickle 文件（index_path）。
- 查询阶段：
  - 从 index_path 读取 docs/tokens，用 tokens 重建 BM25Okapi 模型。
  - 对查询做相同分词，计算 scores，并返回 Top-k 文档（附带 score）。

主要函数：
- `_tokenize(text: str) -> List[str]`：简单英文分词，转小写，保留 [a-z0-9]。
- `build_bm25_index(clean_path: str, index_path: str) -> int`：构建并保存索引。
- `search_bm25(index_path: str, query: str, topk: int = 5) -> List[Dict]`：查询接口。

测试/使用：
- 构建：
  python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
- 查询（示例，建议通过 retriever.py 统一入口）：
  在 Python 中：
    from src.index_bm25 import search_bm25
    res = search_bm25('./data/bm25.idx', 'contrastive learning', 5)
    print(res[:2])

注意事项：
- 为简化实现，查询时每次重建 BM25Okapi（基于 tokens），在小数据/教学场景足够。
- 如果文档规模很大，可以考虑持久化 BM25 模型或引入更高效的检索库。
"""

import argparse
import json
import pickle
import re
from typing import List, Dict

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """简单英文分词：转小写，仅保留字母数字片段。"""
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


def build_bm25_index(clean_path: str, index_path: str) -> int:
    """从 clean.jsonl 构建 BM25 索引，并保存到 index_path。返回文档数量。"""
    docs: List[Dict] = []
    tokens: List[List[str]] = []

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
            docs.append({
                "id": str(_id),
                "title": title,
                "abstract": abstract,
                "categories": categories,
                "text": text,
            })
            tokens.append(_tokenize(text))

    with open(index_path, "wb") as f_out:
        pickle.dump({"docs": docs, "tokens": tokens}, f_out)

    print(f"Built BM25 index with {len(docs)} docs -> {index_path}")
    return len(docs)


def search_bm25(index_path: str, query: str, topk: int = 5) -> List[Dict]:
    """用 BM25 索引搜索 query，返回 Top-k 结果列表。每条包含：id/title/abstract/text/categories/score。"""
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    docs, tokens = data["docs"], data["tokens"]
    bm25 = BM25Okapi(tokens)
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    idxs = list(range(len(docs)))
    idxs_sorted = sorted(idxs, key=lambda i: scores[i], reverse=True)[: max(0, topk)]
    results: List[Dict] = []
    for i in idxs_sorted:
        d = dict(docs[i])
        d["score"] = float(scores[i])
        results.append(d)
    return results


def main():
    """命令行：
    python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--index", dest="index_path", required=True)
    args = parser.parse_args()

    build_bm25_index(args.in_path, args.index_path)


if __name__ == "__main__":
    main()

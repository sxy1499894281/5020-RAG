#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
snippets.py

功能概述：
- 将文档文本（title + abstract 或 text）分句；
- 对每句与 query 计算相关性分数（默认基于关键词重合度的简化 BM25 风格）；
- 为每篇文档选出若干 Top 句作为 evidence，供 RAG 构造上下文与展示引用。

实现思路：
- `sentence_split`：使用正则按中英标点切分，去除空句与多余空白。
- `score_sentences`：
  - 默认 method="bm25"：对 query 与句子做简单 token 化，使用共有词数量作为分数；
  - 预留 method="embedding" 的分支，后续可用向量相似度替换或增强。
- `select_evidence_for_docs`：
  - 遍历检索到的 doc 列表，为每个 doc 选出 `per_doc` 条高分句；
  - 汇总后按分数降序，取前 `max_total` 条返回。

主要函数：
- `sentence_split(text) -> List[str]`
- `score_sentences(query, doc, max_sentences=3, method='bm25') -> List[Dict]`
- `select_evidence_for_docs(query, docs, per_doc=2, max_total=10) -> List[Dict]`

测试/使用：
- 在交互式环境：
  from src.snippets import score_sentences, select_evidence_for_docs
  doc = {"id":"x","title":"A","text":"Sentence one. Sentence two about graph neural networks."}
  print(score_sentences("graph neural networks", doc))
"""

import re
from typing import List, Dict


def sentence_split(text: str) -> List[str]:
    """简单的分句函数：按常见标点切分，去除空句。"""
    text = text or ""
    parts = re.split(r"[。！？.!?]\s+|[。！？.!?]$", text)
    return [p.strip() for p in parts if p and p.strip()]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def score_sentences(
    query: str,
    doc: Dict,
    max_sentences: int = 3,
    method: str = "bm25",
) -> List[Dict]:
    """对单个文档中的句子进行打分，返回若干句子及其分数。"""
    text = doc.get("text") or ((doc.get("title", "") + " " + doc.get("abstract", "")).strip())
    sentences = sentence_split(text)
    if not sentences:
        return []

    scored: List[Dict] = []
    if method == "bm25":
        q_tokens = set(_tokenize(query))
        for s in sentences:
            s_tokens = set(_tokenize(s))
            overlap = len(q_tokens & s_tokens)
            if overlap > 0:
                scored.append({"sentence": s, "score": float(overlap)})
    elif method == "embedding":
        # 预留：可扩展为向量相似度打分
        q_tokens = set(_tokenize(query))
        for s in sentences:
            s_tokens = set(_tokenize(s))
            overlap = len(q_tokens & s_tokens)
            if overlap > 0:
                scored.append({"sentence": s, "score": float(overlap)})

    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored_sorted[: max(0, int(max_sentences))]


def select_evidence_for_docs(
    query: str,
    docs: List[Dict],
    per_doc: int = 2,
    max_total: int = 10,
) -> List[Dict]:
    """针对多个文档选取句级 evidence，返回统一结构列表。"""
    evidences: List[Dict] = []
    for d in docs or []:
        ss = score_sentences(query, d, max_sentences=int(per_doc))
        for item in ss:
            evidences.append({
                "id": d.get("id"),
                "title": d.get("title"),
                "sentence": item["sentence"],
                "score": float(item["score"]),
            })
    evidences_sorted = sorted(evidences, key=lambda x: x["score"], reverse=True)
    return evidences_sorted[: max(0, int(max_total))]

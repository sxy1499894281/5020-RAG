#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
expansion.py

功能概述：
- PRF（伪相关反馈）：对原始 query 做一次初排，从 Top-M 文档中统计高频关键词作为扩展词。
- LLM 改写：调用 LLMClient 将 query 改写为多个等价/更具体的问法。
- 综合扩展：`expand_query` 返回包含原始 query + PRF 扩展 + LLM 改写的变体集合。

实现思路：
- PRF：
  - 用当前检索后端（bm25/dense/hybrid）检索 Top-M；
  - 收集这些文档的 text，并做简单英文分词（正则 [a-z0-9]+，转小写）；
  - 统计词频，去掉停用词/过短词；
  - 返回前 top_terms 个词作为扩展候选。
- LLM 改写：
  - 准备 system+user 提示，要求每行一个改写；
  - 调用 `LLMClient.generate`，按行切分并清洗；
- 综合：
  - 初始 variants=[query]；
  - use_prf 时把 top-3 扩展词拼接为一个 query 追加；
  - use_llm 时追加若干 LLM 改写；
  - 去重保持顺序。

主要函数：
- `prf_terms(query, retriever_cfg, m_docs=5, top_terms=10) -> List[str]`
- `llm_expand(query, client, n_variants=3) -> List[str]`
- `expand_query(query, client=None, use_prf=True, use_llm=True, retriever_cfg=None, prf_cfg=None, llm_cfg=None) -> List[str]`

测试/使用：
- 在交互式环境：
  from src.expansion import expand_query
  print(expand_query('graph neural networks', client=None, use_prf=True, use_llm=False, retriever_cfg={'mode':'bm25','alpha':0.5,'config_path':'configs/config.yaml'}))
"""

import re
from collections import Counter
from typing import List, Dict, Optional

# 简单英文停用词表（可按需扩展）
_STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","by","we","our","this","that","is","are","be","as","at","from","it","its","into","can","may","using","use","used","based","via","such","also","these","those","their","have","has","had","not","no","yes","paper","study","research","method","approach","result","results","show","shows","showed","showing","provide","provides","provided","demonstrate","demonstrates","propose","proposes","proposed","present","presents","presented","model","models","task","tasks","dataset","datasets","data","analysis","recent","state","art","effect","effects","effectiveness","improve","improves","improved","improvement","improvements","towards","via","more","than","new","novel","framework","system","methodology","problem","problems","how","what","when","where","why","which","make","makes","made","making","learn","learning","neural","network","networks","deep","representation","representations","performance","performances","benchmark","benchmarks","paper","work","works","proposed","approach","technique","techniques","modeling","models","applications","application","results","experiments","experimental","evaluation","evaluations",
}


def _tokenize_en(text: str) -> List[str]:
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)


def prf_terms(
    query: str,
    retriever_cfg: Dict,
    m_docs: int = 5,
    top_terms: int = 10,
) -> List[str]:
    """从初排 Top-M 文档中统计高频关键词，返回扩展词列表。"""
    try:
        # 延迟导入，避免循环依赖
        from .retriever import retrieve  # type: ignore
    except Exception:
        return []

    mode = retriever_cfg.get("mode", "hybrid")
    alpha = float(retriever_cfg.get("alpha", 0.5))
    config_path = retriever_cfg.get("config_path", "configs/config.yaml")

    docs = retrieve(query, topk=max(1, int(m_docs)), mode=mode, alpha=alpha, config_path=config_path)
    bag = Counter()
    for d in docs:
        text = d.get("text") or (d.get("title", "") + " " + d.get("abstract", ""))
        tokens = [t for t in _tokenize_en(text) if len(t) >= 3 and t not in _STOPWORDS]
        bag.update(tokens)

    if not bag:
        return []
    return [w for w, _ in bag.most_common(int(top_terms))]


def llm_expand(
    query: str,
    client: "LLMClient",
    n_variants: int = 3,
) -> List[str]:
    """使用 LLM 把 query 改写为 n 个等价或更具体的问法。"""
    if client is None:
        return []
    system_prompt = "你是检索查询改写助手，请基于原始检索语句给出若干等价或更具体的提问，每行一个，语言保持英文。"
    user_prompt = f"Original query: {query}\nPlease provide {int(n_variants)} alternative queries, one per line."
    text = client.generate(system_prompt, user_prompt) or ""
    lines = [ln.strip("- •* ") for ln in text.splitlines() if ln.strip()]
    # 过滤掉与原始 query 完全相同的
    uniq: List[str] = []
    seen = set()
    for ln in lines:
        if ln.lower() == (query or "").lower():
            continue
        if ln not in seen:
            uniq.append(ln)
            seen.add(ln)
        if len(uniq) >= int(n_variants):
            break
    return uniq


def expand_query(
    query: str,
    client: Optional["LLMClient"],
    use_prf: bool = True,
    use_llm: bool = True,
    retriever_cfg: Optional[Dict] = None,
    prf_cfg: Optional[Dict] = None,
    llm_cfg: Optional[Dict] = None,
) -> List[str]:
    """综合 PRF 和 LLM 改写，返回若干 query 变体（至少包含原始 query）。"""
    variants: List[str] = [query]

    # PRF 扩展
    if use_prf and retriever_cfg is not None:
        terms = prf_terms(
            query,
            retriever_cfg=retriever_cfg,
            m_docs=int((prf_cfg or {}).get("m_docs", 5)),
            top_terms=int((prf_cfg or {}).get("top_terms", 10)),
        )
        if terms:
            expanded_q = (query + " " + " ".join(terms[:3])).strip()
            variants.append(expanded_q)

    # LLM 改写
    if use_llm and client is not None:
        n_vars = int((llm_cfg or {}).get("n_variants", 3))
        llm_qs = llm_expand(query, client, n_vars)
        variants.extend(llm_qs)

    # 去重保持顺序
    out: List[str] = []
    seen = set()
    for q in variants:
        if q and q not in seen:
            out.append(q)
            seen.add(q)
    return out

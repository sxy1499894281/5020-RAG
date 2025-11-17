#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
heuristics.py

功能概述：
- 对查询做轻量级启发式分析，输出查询类型与建议 alpha（用于 Hybrid 融合）。
- 可选：根据关键词粗略预测 query 的学科类别前缀，用于类别过滤/加权。
- 根据延迟预算（SLA）返回一组策略开关，用于在线快速决策。

实现思路：
- 通过简单的统计特征（长度、数字/符号占比、缩写/大写比例）将查询划分为 term/semantic/mixed。
- 使用词典/规则对 query 进行领域指示（如包含 NLP/GNN/Transformer → cs.*）。
- 使用分段规则（阈值）将 latency_budget_ms 映射到策略组合。

主要函数：
- `_basic_query_features(q: str) -> Dict`：提取基本特征。
- `classify_query(q: str) -> Dict`：输出 {type, alpha, features}。
- `predict_query_category(q: str) -> Optional[str]`：返回类似 "cs.CL" 的前缀或 None。
- `choose_strategy(latency_budget_ms: int) -> Dict`：根据 SLA 选择策略。

测试/使用：
- 在交互式环境：
  from src.heuristics import classify_query, predict_query_category, choose_strategy
  print(classify_query("GNN for text classification"))
  print(predict_query_category("Vision Transformer for CIFAR-10"))
  print(choose_strategy(800))
"""

from typing import Dict, Optional


def _basic_query_features(q: str) -> Dict:
    """提取 query 的简单特征。"""
    q = q or ""
    len_chars = len(q)
    tokens = q.split()
    len_tokens = len(tokens)
    num_digits = sum(ch.isdigit() for ch in q)
    punct_set = set(",.;:!?[](){}-_/\\|@#$%^&*+=~`<>")
    num_punct = sum(ch in punct_set for ch in q)
    num_upper = sum(ch.isupper() for ch in q)
    ratio_upper = (num_upper / len_chars) if len_chars else 0.0
    ratio_digits_punct = ((num_digits + num_punct) / len_chars) if len_chars else 0.0
    avg_token_len = (sum(len(t) for t in tokens) / len_tokens) if len_tokens else 0.0
    return {
        "len_chars": len_chars,
        "len_tokens": len_tokens,
        "num_digits": num_digits,
        "num_punct": num_punct,
        "ratio_upper": ratio_upper,
        "ratio_digits_punct": ratio_digits_punct,
        "avg_token_len": avg_token_len,
    }


def classify_query(q: str) -> Dict:
    """将 query 粗分为 'term' / 'semantic' / 'mixed' 三类，并给出建议 alpha。"""
    feat = _basic_query_features(q)
    t = "mixed"
    alpha = 0.5

    # 简单启发：短而符号/数字密集 → term；长句 → semantic；其他 → mixed
    if feat["len_tokens"] <= 4 and feat["ratio_digits_punct"] >= 0.10:
        t, alpha = "term", 0.2
    elif feat["len_tokens"] >= 10 or (feat["len_tokens"] >= 7 and feat["avg_token_len"] >= 5.0):
        t, alpha = "semantic", 0.8
    else:
        t, alpha = "mixed", 0.5

    return {"type": t, "alpha": alpha, "features": feat}


def predict_query_category(q: str) -> Optional[str]:
    """根据关键词粗略预测学科类别前缀。返回如 'cs.CL' 或 None。"""
    ql = (q or "").lower()
    # 极简规则集合，可按需扩展
    rules = [
        ("nlp", "cs.CL"),
        ("transformer", "cs.CL"),
        ("bert", "cs.CL"),
        ("gpt", "cs.CL"),
        ("vision", "cs.CV"),
        ("image", "cs.CV"),
        ("graph", "cs.LG"),
        ("gnn", "cs.LG"),
        ("reinforcement", "cs.LG"),
        ("retrieval", "cs.IR"),
        ("quantum", "quant-ph"),
        ("statistical", "stat.ML"),
        ("bayesian", "stat.ML"),
        ("mathematical", "math"),
        ("economics", "econ"),
    ]
    for kw, cat in rules:
        if kw in ql:
            return cat
    return None


def choose_strategy(latency_budget_ms: int) -> Dict:
    """根据延迟预算选择策略组合。"""
    if latency_budget_ms < 500:
        return {"mode": "bm25", "enable_rerank": False, "enable_expansion": False}
    elif latency_budget_ms < 1200:
        return {"mode": "hybrid", "enable_rerank": False, "enable_expansion": True}
    else:
        return {"mode": "hybrid", "enable_rerank": True, "enable_expansion": True}

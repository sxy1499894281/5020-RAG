#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retriever.py

功能概述：
- 基础检索接口 `retrieve`：支持 bm25 / dense / hybrid 三种模式。
- 增强检索接口 `retrieve_enhanced`：在基础上整合动态 alpha、查询扩展（PRF+LLM）、类别感知、SLA 策略与 Cross-Encoder 重排。

实现思路：
- 读取 configs/config.yaml，集中管理路径与参数。
- bm25：调用 `index_bm25.search_bm25`；dense：调用 `index_dense.search_dense`。
- hybrid：对两端分数做 min-max 归一化后融合：final = alpha * dense + (1-alpha) * bm25。
- 类别逻辑：可按 `category` 配置过滤或对匹配类别做加权。
- 增强入口：
  - 若 `use_dynamic_alpha=True` 且 `alpha=None`，调用 `heuristics.classify_query` 估计 alpha；
  - 若启用 expansion：调用 `expansion.expand_query` 生成多个变体并合并检索结果；
  - 若启用 rerank：对前 N 个候选调用 `reranker.rerank` 精排。

主要函数：
- `_load_config(path) -> Dict`：读取 YAML 配置。
- `_norm(scores) -> List[float]`：min-max 归一化。
- `_apply_category_logic(docs, query_cat, cat_cfg)`：类别过滤/加权。
- `retrieve(query, topk, mode, alpha, config_path)`：基础检索。
- `retrieve_enhanced(...)`：增强检索。

命令行使用（快速测试）：
- 基础：
  python src/retriever.py "contrastive learning" --mode hybrid --topk 5
- 增强：
  python src/retriever.py "graph neural networks" --mode hybrid --topk 5 --enhanced
"""

import argparse
from typing import List, Dict, Optional

import yaml
import os
import sys

try:
    from .index_bm25 import search_bm25
    from .index_dense import search_dense
    from .heuristics import classify_query, predict_query_category, choose_strategy
    from .expansion import expand_query
    from .reranker import rerank as rerank_docs
except Exception:
    # 作为脚本运行时的回退：将项目根目录加入 sys.path
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.index_bm25 import search_bm25  # type: ignore
    from src.index_dense import search_dense  # type: ignore
    from src.heuristics import classify_query, predict_query_category, choose_strategy  # type: ignore
    from src.expansion import expand_query  # type: ignore
    from src.reranker import rerank as rerank_docs  # type: ignore


def _load_config(path: str) -> Dict:
    """从 YAML 文件加载配置。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _norm(scores: List[float]) -> List[float]:
    """min-max 归一化到 0~1。空或常数序列返回 0。"""
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [0.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _apply_category_logic(docs: List[Dict], query_cat: Optional[str], cat_cfg: Dict) -> List[Dict]:
    """根据类别配置进行过滤/加权。"""
    def primary_cat(d: Dict) -> str:
        cats = d.get("categories") or ""
        return (cats.split() or [""])[0]

    # 过滤
    if cat_cfg.get("enable_filter"):
        prefixes = cat_cfg.get("allowed_prefixes") or []
        docs = [d for d in docs if any(primary_cat(d).startswith(p) for p in prefixes)]

    # 加权
    if cat_cfg.get("enable_boost") and query_cat:
        bf = float(cat_cfg.get("boost_factor", 0.0))
        qpref = query_cat.split(".")[0] if "." in query_cat else query_cat
        for d in docs:
            if primary_cat(d).startswith(qpref):
                d["score"] = float(d.get("score", 0.0)) * (1.0 + bf)
    return docs


def _as_id_map(docs: List[Dict]) -> Dict[str, Dict]:
    m: Dict[str, Dict] = {}
    for d in docs:
        if not d:
            continue
        m[str(d.get("id"))] = d
    return m


def retrieve(
    query: str,
    topk: int = 5,
    mode: str = "bm25",
    alpha: float = 0.5,
    config_path: str = "configs/config.yaml",
) -> List[Dict]:
    """基础检索入口。
    - mode: "bm25" / "dense" / "hybrid"
    - hybrid: final_score = alpha * dense_score + (1-alpha) * bm25_score
    返回：[{id,title,text,categories,score}]
    """
    cfg = _load_config(config_path)
    bm25_path = cfg["bm25"]["index_path"]
    dcfg = cfg["dense"]

    if mode == "bm25":
        return search_bm25(bm25_path, query, topk)
    if mode == "dense":
        return search_dense(dcfg["db"], query, topk, dcfg["model"], dcfg["collection"])

    if mode != "hybrid":
        raise ValueError(f"Unknown mode: {mode}")

    # hybrid：分别检索，再融合
    bm25_res = search_bm25(bm25_path, query, max(topk, 20))
    dense_res = search_dense(dcfg["db"], query, max(topk, 20), dcfg["model"], dcfg["collection"])

    bm25_scores = _norm([d.get("score", 0.0) for d in bm25_res])
    dense_scores = _norm([d.get("score", 0.0) for d in dense_res])
    bm25_map = _as_id_map(bm25_res)
    dense_map = _as_id_map(dense_res)

    # 合并 id 集合
    ids = set(bm25_map.keys()) | set(dense_map.keys())
    merged: List[Dict] = []
    for _id in ids:
        d_b = bm25_map.get(_id)
        d_d = dense_map.get(_id)
        # 提供稳定 text/title
        any_d = d_d or d_b or {}
        s_b = bm25_scores[bm25_res.index(d_b)] if d_b in bm25_res else 0.0
        s_d = dense_scores[dense_res.index(d_d)] if d_d in dense_res else 0.0
        score = float(alpha) * s_d + float(1.0 - float(alpha)) * s_b
        merged.append({
            "id": any_d.get("id"),
            "title": any_d.get("title", ""),
            "text": any_d.get("text", any_d.get("abstract", "")),
            "categories": any_d.get("categories", ""),
            "score": score,
        })

    merged_sorted = sorted(merged, key=lambda x: x["score"], reverse=True)
    return merged_sorted[: topk]


def retrieve_enhanced(
    query: str,
    topk: int = 5,
    mode: str = "hybrid",
    alpha: Optional[float] = None,
    config_path: str = "configs/config.yaml",
    use_dynamic_alpha: bool = True,
    enable_expansion: bool = False,
    enable_rerank: bool = False,
    latency_budget_ms: Optional[int] = None,
) -> List[Dict]:
    """增强版检索入口。返回最终 Top-k 文档。"""
    cfg = _load_config(config_path)

    # SLA 策略
    if latency_budget_ms is not None:
        strat = choose_strategy(int(latency_budget_ms))
        mode = strat["mode"]
        enable_rerank = strat["enable_rerank"]
        enable_expansion = strat["enable_expansion"]

    # 动态 alpha（仅 hybrid 有意义）
    if use_dynamic_alpha and alpha is None and mode == "hybrid":
        info = classify_query(query)
        alpha = float(info.get("alpha", cfg["retrieval"].get("alpha", 0.5)))
    if alpha is None:
        alpha = float(cfg["retrieval"].get("alpha", 0.5))

    # 生成查询变体
    queries = [query]
    exp_cfg = cfg["retrieval"].get("expansion", {})
    if enable_expansion and exp_cfg.get("enable", False):
        from .rag import LLMClient  # 延迟导入，避免循环
        gencfg = cfg.get("generation", {})
        client: Optional[LLMClient] = None
        if gencfg.get("provider", "mock") != "mock":
            # 非 mock 才实例化真实 client
            from dotenv import load_dotenv
            import os
            load_dotenv()
            base_url = os.getenv("OPENAI_BASE_URL")
            api_key = os.getenv("OPENAI_API_KEY")
            client = LLMClient(gencfg.get("provider", "openai"), gencfg.get("model", "gpt-4.1-mini"), base_url=base_url, api_key=api_key, max_tokens=int(gencfg.get("max_tokens", 512)))
        queries = expand_query(
            query,
            client=client,
            use_prf=exp_cfg.get("prf", {}).get("enable", True),
            use_llm=exp_cfg.get("llm", {}).get("enable", False),
            retriever_cfg={"mode": mode, "alpha": alpha, "config_path": config_path},
            prf_cfg={"m_docs": exp_cfg.get("prf", {}).get("m_docs", 5), "top_terms": exp_cfg.get("prf", {}).get("top_terms", 10)},
            llm_cfg={"n_variants": exp_cfg.get("llm", {}).get("n_variants", 3)},
        )

    # 对每个变体检索并合并分数（取最大）
    merged_map: Dict[str, Dict] = {}
    for q in queries:
        if mode == "hybrid":
            partial = retrieve(q, topk=max(topk, 50), mode=mode, alpha=alpha, config_path=config_path)
        else:
            partial = retrieve(q, topk=max(topk, 50), mode=mode, alpha=alpha, config_path=config_path)
        for d in partial:
            _id = str(d.get("id"))
            if _id not in merged_map or d.get("score", 0.0) > merged_map[_id].get("score", 0.0):
                merged_map[_id] = d

    docs_all = list(merged_map.values())
    # 类别逻辑
    query_cat = predict_query_category(query)
    docs_all = _apply_category_logic(docs_all, query_cat, cfg.get("category", {}))

    # 重排
    if enable_rerank and cfg.get("rerank", {}).get("enable", False):
        topN = int(cfg["rerank"].get("topn", 50))
        model_name = cfg["rerank"].get("model", "bge-reranker-base")
        batch_size = int(cfg["rerank"].get("batch_size", 16))
        docs_all_sorted = sorted(docs_all, key=lambda x: x.get("score", 0.0), reverse=True)[: topN]
        docs_reranked = rerank_docs(query, docs_all_sorted, model_name, topk=topk, batch_size=batch_size)
        return docs_reranked

    # 否则直接按分数排序
    docs_sorted = sorted(docs_all, key=lambda x: x.get("score", 0.0), reverse=True)
    return docs_sorted[: topk]


def main():
    """命令行简单测试：
    python src/retriever.py "contrastive learning" --mode hybrid --topk 5 --enhanced
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--enhanced", action="store_true")
    args = parser.parse_args()

    if args.enhanced:
        res = retrieve_enhanced(args.query, args.topk, mode=args.mode)
    else:
        res = retrieve(args.query, args.topk, mode=args.mode)
    for i, d in enumerate(res, 1):
        print(f"[{i}] {d.get('id')} | {d.get('title')} | score={d.get('score'):.4f}")


if __name__ == "__main__":
    main()

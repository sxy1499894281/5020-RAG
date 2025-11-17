#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval.py

功能概述：
- 在 QA 集上评测不同检索模式或策略组合（基础/增强）。
- 计算 Recall@k、MRR，以及检索时间、（可选）生成时间和端到端时间。
- 将结果写入 CSV，便于对比与画图。

实现思路：
- 读取 JSONL QA：每行形如 {"q": str, "gold_ids": [str, ...]}。
- 对每个模式循环：
  - 遍历样本，计时进行检索（retrieve 或 retrieve_enhanced）；
  - 计算单样本 recall@k 与 mrr@k；
  - 若 include_gen=True，则调用 answer/enhanced_answer 计时；
  - 汇总均值写入 CSV。

主要函数：
- `_read_qa(path) -> List[Dict]`
- `_recall_at_k(retrieved_ids, gold_ids) -> float`
- `_mrr_at_k(retrieved_ids, gold_ids) -> float`
- `evaluate(qa_path, modes, out_csv, k=5, include_gen=False, use_enhanced=False)`

测试/使用：
- 基础检索评测：
  python src/eval.py --qa ./data/dev_qa.jsonl --modes bm25 dense hybrid --out ./logs/metrics_baseline.csv --k 5
- 增强检索 + 端到端：
  python src/eval.py --qa ./data/dev_qa.jsonl --modes hybrid --out ./logs/metrics_enhanced.csv --k 5 --include_gen --use_enhanced

注意事项：
- 运行前应已构建 BM25 与稠密索引；
- 若 generation.provider=mock，则生成时间表示 mock 的占位时延；
- 大数据评测时建议先用较小 QA 子集验证流程是否通畅。
"""

import argparse
import csv
import json
import os
import sys
import time
from statistics import mean
from typing import List, Dict

try:
    from .retriever import retrieve, retrieve_enhanced
    from .rag import answer, enhanced_answer
except Exception:
    # 脚本运行回退导入
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.retriever import retrieve, retrieve_enhanced  # type: ignore
    from src.rag import answer, enhanced_answer  # type: ignore


def _read_qa(path: str) -> List[Dict]:
    """读取 QA JSONL 文件，返回列表。"""
    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def _recall_at_k(retrieved: List[str], gold: List[str]) -> float:
    """Recall@k 的单样本版本：命中则为 1，否则为 0。"""
    set_r = set([str(x) for x in retrieved])
    set_g = set([str(x) for x in gold])
    return 1.0 if (set_r & set_g) else 0.0


def _mrr_at_k(retrieved: List[str], gold: List[str]) -> float:
    """MRR 的单样本版本：第一个命中的倒数排名。"""
    gold_set = set([str(x) for x in gold])
    for i, rid in enumerate(retrieved):
        if str(rid) in gold_set:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(
    qa_path: str,
    modes: List[str],
    out_csv: str,
    k: int = 5,
    include_gen: bool = False,
    use_enhanced: bool = False,
) -> None:
    """核心评测函数。将结果写入 out_csv。"""
    qa_items = _read_qa(qa_path)
    rows = []
    for mode in modes:
        recalls, mrrs, search_times, gen_times, end2end_times = [], [], [], [], []
        for item in qa_items:
            q = item.get("q") or ""
            gold = item.get("gold_ids") or []
            t0 = time.perf_counter()
            if use_enhanced:
                docs = retrieve_enhanced(q, k, mode)
            else:
                docs = retrieve(q, k, mode)
            t1 = time.perf_counter()
            ids = [d.get("id") for d in docs]
            recalls.append(_recall_at_k(ids, gold))
            mrrs.append(_mrr_at_k(ids, gold))
            search_ms = (t1 - t0) * 1000.0
            search_times.append(search_ms)

            gen_ms = 0.0
            if include_gen:
                g0 = time.perf_counter()
                if use_enhanced:
                    _ = enhanced_answer(q, mode, k)
                else:
                    _ = answer(q, mode, k)
                g1 = time.perf_counter()
                gen_ms = (g1 - g0) * 1000.0
            gen_times.append(gen_ms)
            end2end_times.append(search_ms + gen_ms)

        row = {
            "mode": mode,
            "k": k,
            "recall": round(mean(recalls), 4) if recalls else 0.0,
            "mrr": round(mean(mrrs), 4) if mrrs else 0.0,
            "search_ms": round(mean(search_times), 2) if search_times else 0.0,
            "gen_ms": round(mean(gen_times), 2) if gen_times else 0.0,
            "end2end_ms": round(mean(end2end_times), 2) if end2end_times else 0.0,
        }
        rows.append(row)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", "k", "recall", "mrr", "search_ms", "gen_ms", "end2end_ms"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Metrics written to {out_csv}")


def main():
    """命令行：
    python src/eval.py --qa ./data/synth_qa.jsonl --modes bm25 dense hybrid --out ./logs/metrics.csv --k 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", dest="qa", required=True)
    parser.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--out", dest="out_csv", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--include_gen", action="store_true")
    parser.add_argument("--use_enhanced", action="store_true")
    args = parser.parse_args()

    evaluate(args.qa, args.modes, args.out_csv, args.k, include_gen=args.include_gen, use_enhanced=args.use_enhanced)


if __name__ == "__main__":
    main()

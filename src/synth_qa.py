#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synth_qa.py

功能概述：
- 从 clean.jsonl 抽样论文，使用 LLM（或 mock 模式）为每篇生成若干问题，形成合成 QA 集合。
- 输出 JSONL：每行 {"q": 问题, "gold_ids": [论文id], "source": "synthetic_llm", "category": 主类别}。

实现思路：
- 逐行读取 clean.jsonl，必要时按类别过滤；
- 随机打乱后取样前 N 篇；
- 为每篇调用 `generate_questions_for_doc`（基于 LLMClient 或 mock 模式）生成 questions_per_doc 个问题；
- 写入 out_path。

主要函数：
- `_iter_clean_docs(clean_path)`：流式 yield 文档字典。
- `_primary_category(categories)`：返回第一个类别标签或 None。
- `generate_questions_for_doc(title, abstract, n_q, client)`：生成 n_q 个问题（mock/LLM）。
- `generate_synthetic_qa(clean_path, out_path, sample_size, questions_per_doc, category_filter, config_path)`：核心逻辑。

测试/使用：
- 生成 50 条（mock）：
  python src/synth_qa.py --in ./data/clean.jsonl --out ./data/synth_qa.jsonl --sample_size 50 --questions_per_doc 1
"""

import argparse
import json
import random
from typing import List, Dict, Optional, Iterator

from .rag import LLMClient, _load_config


def _iter_clean_docs(clean_path: str) -> Iterator[Dict]:
    """逐行读取 clean.jsonl，yield 单篇文档字典。"""
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _primary_category(categories: str) -> Optional[str]:
    """从 categories 字符串中取出第一个标签，比如 "cs.CL stat.ML" -> "cs.CL"。"""
    if not categories:
        return None
    parts = categories.split()
    return parts[0] if parts else None


def generate_questions_for_doc(
    title: str,
    abstract: str,
    n_q: int,
    client: "LLMClient",
) -> List[str]:
    """输入单篇论文的 title + abstract，生成 n_q 个问题。mock/LLM 均可。"""
    title = title or ""
    abstract = abstract or ""
    if client is None or client.provider == "mock":
        # 简单 mock 模式
        base = title if title else (abstract[:48] + ("..." if len(abstract) > 48 else ""))
        return [f"What is the main contribution of: {base}?" for _ in range(max(1, int(n_q)))]

    system_prompt = (
        "你是一个研究助教，请根据论文标题和摘要生成若干可被摘要回答的问题，"
        "尽量覆盖论文的核心对象、方法、贡献与实验场景。每行一个英文问题。"
    )
    user_prompt = f"Title: {title}\nAbstract: {abstract}\nPlease write {int(n_q)} questions, one per line."
    text = client.generate(system_prompt, user_prompt) or ""
    qs = [ln.strip("- •* ") for ln in text.splitlines() if ln.strip()]
    return qs[: max(1, int(n_q))]


def generate_synthetic_qa(
    clean_path: str,
    out_path: str,
    sample_size: int = 1000,
    questions_per_doc: int = 2,
    category_filter: Optional[str] = None,
    config_path: str = "configs/config.yaml",
) -> int:
    """抽样 sample_size 篇论文，对每篇生成 questions_per_doc 个问题，并写入 out_path。返回生成的 QA 条数。"""
    cfg = _load_config(config_path)
    gen_cfg = cfg.get("generation", {})
    client = LLMClient(gen_cfg.get("provider", "mock"), gen_cfg.get("model", "gpt-4.1-mini"), max_tokens=int(gen_cfg.get("max_tokens", 512)))

    docs = list(_iter_clean_docs(clean_path))
    if category_filter:
        docs = [d for d in docs if _primary_category(d.get("categories", "")) == category_filter]

    if not docs:
        print("No documents available for synthetic QA generation.")
        return 0

    random.shuffle(docs)
    sampled = docs[: max(0, int(sample_size))]
    count = 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for rec in sampled:
            title = rec.get("title", "")
            abstract = rec.get("abstract", "")
            doc_id = rec.get("id")
            cat = _primary_category(rec.get("categories", ""))
            qs = generate_questions_for_doc(title, abstract, int(questions_per_doc), client)
            for q in qs:
                item = {
                    "q": q,
                    "gold_ids": [doc_id] if doc_id else [],
                    "source": "synthetic_llm" if client.provider != "mock" else "synthetic_mock",
                    "category": cat,
                }
                json.dump(item, f_out, ensure_ascii=False)
                f_out.write("\n")
                count += 1
    print(f"Generated {count} QA pairs -> {out_path}")
    return count


def main():
    """命令行：
    python src/synth_qa.py --in ./data/clean.jsonl --out ./data/synth_qa.jsonl --sample_size 500 --questions_per_doc 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--questions_per_doc", type=int, default=2)
    parser.add_argument("--category", dest="category_filter", default=None)
    args = parser.parse_args()

    generate_synthetic_qa(
        args.in_path,
        args.out_path,
        sample_size=args.sample_size,
        questions_per_doc=args.questions_per_doc,
        category_filter=args.category_filter,
    )


if __name__ == "__main__":
    main()

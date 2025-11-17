#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest.py

功能概述：
- 从原始 arXiv JSONL（每行一个 JSON 对象）中流式抽取核心字段，生成更小的 clean.jsonl。
- 支持超大文件的逐行处理，避免一次性加载 4.6GB 到内存。

实现思路：
- 逐行读取原始文件，strip 空行与异常行，使用 json.loads 解析。
- 抽取字段：id/title/abstract/categories/versions -> created（从 versions[0].created 解析）。
- 对 abstract 做轻量清洗：去除换行与简单 LaTeX 片段（$...$、\\(...\\) 等）。
- 写入目标文件（UTF-8，一行一个 JSON）。

主要函数：
- `_parse_created(versions) -> Optional[str]`：从 versions 列表解析 created。
- `_clean_abstract(raw: str) -> str`：轻量清洗摘要。
- `stream_clean_arxiv(input_path, output_path, max_rows=None)`：核心流式清洗并写出。

输入/输出：
- 输入：原始 JSONL 文件路径（可能后缀 .json，但格式为 JSON Lines）。
- 输出：clean.jsonl（每行包含 id/title/abstract/categories/created）。

命令行使用（测试）：
- 基于小样本调试：
  python src/ingest.py --in ./data/sample_raw.jsonl --out ./data/clean.jsonl --max_rows 1000
- 基于完整版数据：
  python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl

注意事项：
- 遇到坏行或缺失字段会跳过但不中断处理。
- 请确保目标目录可写。
"""

import argparse
import json
import re
from typing import Optional


def _parse_created(versions) -> Optional[str]:
    """从 versions 字段里解析 created 时间。
    - versions 通常是一个列表，如 [{"version": "v1", "created": "..."}, ...]
    - 若解析失败返回 None
    """
    try:
        if not versions:
            return None
        if isinstance(versions, list) and versions:
            first = versions[0]
            if isinstance(first, dict):
                return first.get("created")
    except Exception:
        return None
    return None


def _clean_abstract(raw: str) -> str:
    """对摘要做非常轻量的清洗：
    - 去掉多余的换行
    - 简单移除 LaTeX 公式片段（$...$、\\(...\\)、\\[...\\]）
    - 去两端空白
    """
    text = (raw or "").replace("\n", " ")
    # 去除 $...$ 与 \(...\) 和 \[...\]
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\\([^)]*\\\)", " ", text)
    text = re.sub(r"\\\[[^]]*\\\]", " ", text)
    return text.strip()


def stream_clean_arxiv(input_path: str, output_path: str, max_rows: Optional[int] = None) -> None:
    """核心函数：流式读取 input_path，写出精简字段到 output_path。
    - max_rows: 调试用，只处理前 N 行。
    """
    n_in, n_out = 0, 0
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            n_in += 1
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            try:
                _id = rec.get("id")
                title = rec.get("title") or ""
                abstract = _clean_abstract(rec.get("abstract") or "")
                categories = rec.get("categories") or ""
                created = _parse_created(rec.get("versions"))
                if not _id or not title:
                    continue
                new_rec = {
                    "id": str(_id),
                    "title": title.strip(),
                    "abstract": abstract,
                    "categories": categories,
                    "created": created,
                }
                json.dump(new_rec, fout, ensure_ascii=False)
                fout.write("\n")
                n_out += 1
            except Exception:
                continue
    # 简单输出处理统计
    print(f"Processed {n_in} lines, wrote {n_out} records to {output_path}")


def main():
    """命令行入口
    示例：
      python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl
    可选：
      python src/ingest.py --in ./data/sample_raw.jsonl --out ./data/clean.jsonl --max_rows 5000
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    stream_clean_arxiv(args.in_path, args.out_path, max_rows=args.max_rows)


if __name__ == "__main__":
    main()

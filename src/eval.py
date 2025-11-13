#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from statistics import mean
from typing import List, Dict

try:
    from .retriever import retrieve
    from .rag import answer as rag_answer
except ImportError:  # allow running as a script
    from retriever import retrieve
    from rag import answer as rag_answer


def _read_qa(path: str) -> List[Dict]:
    items = []
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
    s = set(gold or [])
    return 1.0 if any(x in s for x in retrieved) else 0.0


def _mrr_at_k(retrieved: List[str], gold: List[str]) -> float:
    s = set(gold or [])
    for i, x in enumerate(retrieved):
        if x in s:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(qa_path: str, modes: List[str], out_csv: str, k: int = 5, include_gen: bool = False) -> None:
    data = _read_qa(qa_path)
    rows = []
    for mode in modes:
        recalls = []
        mrrs = []
        search_times = []
        gen_times = []
        total_times = []
        for item in data:
            q = item.get("q") or ""
            gold = item.get("gold_ids") or []
            t0 = time.perf_counter()
            res = retrieve(q, k, mode)
            t1 = time.perf_counter()
            ids = [r.get("id") for r in res]
            recalls.append(_recall_at_k(ids, gold))
            mrrs.append(_mrr_at_k(ids, gold))
            search_ms = (t1 - t0) * 1000.0
            search_times.append(search_ms)
            gen_ms = 0.0
            if include_gen:
                g0 = time.perf_counter()
                _ = rag_answer(q, mode, k)
                g1 = time.perf_counter()
                gen_ms = (g1 - g0) * 1000.0
            gen_times.append(gen_ms)
            total_times.append(search_ms + gen_ms)
        row = {
            "mode": mode,
            "k": k,
            "recall": round(mean(recalls), 4) if recalls else 0.0,
            "mrr": round(mean(mrrs), 4) if mrrs else 0.0,
            "search_ms": round(mean(search_times), 2) if search_times else 0.0,
            "gen_ms": round(mean(gen_times), 2) if gen_times else 0.0,
            "end2end_ms": round(mean(total_times), 2) if total_times else 0.0,
        }
        rows.append(row)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "k", "recall", "mrr", "search_ms", "gen_ms", "end2end_ms"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qa", dest="qa", required=True)
    p.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    p.add_argument("--out", dest="out_csv", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--include_gen", action="store_true")
    args = p.parse_args()
    evaluate(args.qa, args.modes, args.out_csv, k=args.k, include_gen=args.include_gen)
    print(f"Metrics written to {args.out_csv}")

import os
from pathlib import Path

from src.ingest import stream_clean_arxiv
from src.index_bm25 import build_bm25_index
from src.retriever import retrieve
from src.rag import answer

DATA_DIR = Path("data")


def test_ingest_sample(tmp_path):
    in_path = DATA_DIR / "sample_raw.jsonl"
    out_path = tmp_path / "clean.jsonl"
    n = stream_clean_arxiv(str(in_path), str(out_path))
    assert n >= 3


def test_bm25_index_and_search(tmp_path):
    clean_path = DATA_DIR / "clean-sample.jsonl"
    index_path = DATA_DIR / "bm25.idx"  # write to default path used by config
    n = build_bm25_index(str(clean_path), str(index_path))
    assert n >= 3
    res = retrieve("graph", 2, "bm25")
    assert isinstance(res, list) and len(res) >= 1


def test_rag_answer_mock():
    # ensure bm25 index exists
    index_path = DATA_DIR / "bm25.idx"
    if not index_path.exists():
        build_bm25_index(str(DATA_DIR / "clean-sample.jsonl"), str(index_path))
    res = answer("What is contrastive learning?", "bm25", 2)
    assert "answer" in res and "citations" in res

#!/usr/bin/env bash
set -euo pipefail

RAW=${RAW:-./data/arxiv-metadata-oai-snapshot.json}
CLEAN=${CLEAN:-./data/clean.jsonl}
BM25_IDX=${BM25_IDX:-./data/bm25.idx}
DB=${DB:-./data/chroma}
MODEL=${MODEL:-bge-small-en-v1.5}
COLLECTION=${COLLECTION:-arxiv}
MAX_ROWS=${MAX_ROWS:-}
BUILD_DENSE=${BUILD_DENSE:-0}
EVAL_QA=${EVAL_QA:-./data/dev_qa.jsonl}
K=${K:-5}

mkdir -p ./logs

echo "[1/6] Ingest -> $CLEAN"
python src/ingest.py --in "$RAW" --out "$CLEAN" ${MAX_ROWS:+--max_rows "$MAX_ROWS"}

echo "[2/6] BM25 Index -> $BM25_IDX"
python src/index_bm25.py --in "$CLEAN" --index "$BM25_IDX"

if [ "$BUILD_DENSE" = "1" ]; then
  echo "[3/6] Dense Index -> $DB"
  python src/index_dense.py --in "$CLEAN" --db "$DB" --model "$MODEL" --collection "$COLLECTION"
else
  echo "[3/6] Dense Index skipped (set BUILD_DENSE=1 to enable)"
fi

echo "[4/6] Retrieve (hybrid)"
python -c "from src.retriever import retrieve; print(retrieve('contrastive learning', ${K}, 'hybrid', 0.5))"

echo "[5/6] RAG Answer"
python -c "from src.rag import answer; print(answer('What is contrastive learning?', 'hybrid', ${K}))"

echo "[6/6] Evaluate -> ./logs/metrics.csv"
python src/eval.py --qa "$EVAL_QA" --modes bm25 dense hybrid --out ./logs/metrics.csv || true

echo "Done."

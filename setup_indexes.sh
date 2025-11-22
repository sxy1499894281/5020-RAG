#!/bin/bash
# setup_indexes.sh - 构建所有索引

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Step 1/3: Cleaning raw data..."
echo "=========================================="
python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl

echo ""
echo "=========================================="
echo "Step 2/3: Building BM25 index..."
echo "=========================================="
python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx

echo ""
echo "=========================================="
echo "Step 3/3: Building dense vector DB..."
echo "=========================================="
python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model BAAI/bge-small-en-v1.5 --collection arxiv

echo ""
echo "=========================================="
echo "All indexes built successfully!"
echo "=========================================="
echo "You can now run your RAG queries."

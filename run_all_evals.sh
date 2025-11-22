#!/bin/bash
# run_all_evals.sh - 运行所有评估配置

set -e
cd ~/rag-project

# 确保有测试集
if [ ! -f ./data/dev_qa.jsonl ]; then
    echo "Generating test QA set..."
    python src/synth_qa.py --in ./data/clean-sample.jsonl --out ./data/dev_qa.jsonl --sample_size 50 --questions_per_doc 1
fi

echo "=========================================="
echo "1. Baseline: hybrid (no enhancements)"
echo "=========================================="
# 备份配置
cp configs/config.yaml configs/config.yaml.backup

# 关闭所有增强功能
sed -i 's/dynamic_alpha: true/dynamic_alpha: false/' configs/config.yaml
sed -i 's/enable: true/enable: false/g' configs/config.yaml
python src/eval.py --qa ./data/dev_qa.jsonl --modes hybrid --out ./logs/metric_baseline.csv --k 5

echo ""
echo "=========================================="
echo "2. + dynamic_alpha"
echo "=========================================="
# 恢复配置
cp configs/config.yaml.backup configs/config.yaml
sed -i 's/dynamic_alpha: true/dynamic_alpha: true/' configs/config.yaml
sed -i 's/rerank:/rerank:\n  enable: false/' configs/config.yaml
python src/eval.py --qa ./data/dev_qa.jsonl --modes hybrid --out ./logs/metric_dynalpha.csv --k 5

echo ""
echo "=========================================="
echo "3. + rerank"
echo "=========================================="
cp configs/config.yaml.backup configs/config.yaml
sed -i 's/enable: true/enable: true/' configs/config.yaml
python src/eval.py --qa ./data/dev_qa.jsonl --modes hybrid --out ./logs/metric_rerank.csv --k 5 --use_enhanced

echo ""
echo "=========================================="
echo "4. + QE (Query Expansion - PRF)"
echo "=========================================="
# QE已在配置中启用
python src/eval.py --qa ./data/dev_qa.jsonl --modes hybrid --out ./logs/metric_qe.csv --k 10 --use_enhanced

echo ""
echo "=========================================="
echo "5. + evidence snippets"
echo "=========================================="
python src/eval.py --qa ./data/dev_qa.jsonl --modes hybrid --out ./logs/metric_evidence.csv --k 5 --use_enhanced --include_gen

echo ""
echo "=========================================="
echo "6. + category filter + SLA"
echo "=========================================="
cp configs/config.yaml.backup configs/config.yaml
sed -i 's/enable_filter: true/enable_filter: true/' configs/config.yaml
sed -i 's/enable_boost: true/enable_boost: true/' configs/config.yaml
python src/eval.py --qa ./data/dev_qa.jsonl --modes hybrid --out ./logs/metric_cat_sla.csv --k 5 --use_enhanced

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
echo "Results saved in ./logs/metric_*.csv"
echo ""
echo "To view results:"
echo "  cat ./logs/metric_baseline.csv"
echo "  cat ./logs/metric_dynalpha.csv"
echo "  cat ./logs/metric_rerank.csv"
echo "  cat ./logs/metric_qe.csv"
echo "  cat ./logs/metric_evidence.csv"
echo "  cat ./logs/metric_cat_sla.csv"

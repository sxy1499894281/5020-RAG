"""
src 包初始化

本包包含多策略增强 RAG 学术问答系统的所有模块：
- 数据预处理 ingest
- BM25 与稠密索引 index_bm25 / index_dense
- 检索器 retriever（含增强 retrieve_enhanced）
- 生成与上下文 rag
- 启发式 heuristics、重排 reranker、扩展 expansion、证据 snippets
- 评测 eval、合成问答 synth_qa

用法请参考各模块文件头部注释中的 CLI 示例。
"""

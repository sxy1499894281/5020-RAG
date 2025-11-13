# arXiv 摘要 RAG - BM25 vs 稠密检索（可混合）

基于 arXiv 元数据摘要构建面向学术领域的检索增强生成（RAG）系统。支持 BM25 倒排检索、稠密向量检索与混合检索；对比准确率、召回率与延迟；输出答案与引用。

---

## 🎯 核心特性

- ✅ 双通路检索：BM25 与稠密向量（Chroma/FAISS），可配置、可持久化
- 🔀 混合融合：分数归一化 + 加权融合（`alpha` 可调）
- 🧰 轻量预处理：面向 4.6GB JSONL 的流式抽取与清洗
- 🧠 可插拔 LLM：本地 Ollama / OpenAI 兼容 API / 本地权重
- 📊 可评估：Recall@k / MRR / Latency，输出 CSV 便于对比
- 🧪 可测试：冒烟与单元测试覆盖 ingest/index/retrieve 关键路径

> 💡 设计理念：以最小工程面、可复现流程，验证 BM25 与稠密检索（含混合）在学术摘要场景的效果与性能权衡。

---

## 📖 工作流程

```
START
  ↓
ingest (数据预处理)
  ├─ 流式读取 JSONL (arXiv 元数据)
  └─ 抽取清洗: id / title / abstract / categories / created → clean.jsonl
  ↓
index (索引构建)
  ├─ BM25: build_bm25_index(clean.jsonl) → bm25.idx
  └─ 稠密: embed_and_build_vector_db(clean.jsonl) → chroma/ 或 faiss 索引
  ↓
retrieve (检索)
  ├─ mode = "bm25" | "dense" | "hybrid" (alpha 加权)
  └─ 输出 Top-k 命中文档 (id/title/score/...)
  ↓
rag (上下文 + 生成)
  ├─ build_context(标题+摘要, 长度裁剪)
  └─ answer(query, mode, topk) → 答案 + 引用
  ↓
eval (离线评估)
  └─ Recall@k / MRR / Latency → metrics.csv
  ↓
END
```

---

## 🚀 快速开始

### 1) 环境准备

建议 Python 版本 >= 3.10

```bash
# 创建虚拟环境（可选）
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -U -r requirements.txt
```

### 2) 数据准备

- 下载 arXiv 元数据快照为 `./data/arxiv-metadata-oai-snapshot.json`（约 4.6GB，JSON Lines）
- 文件编码 UTF-8，每行一个 JSON 对象

可选：先抽样前 N 万行做快速迭代（开发期更高效）

### 3) 最小可运行流程（命令速览）

```bash
# 阶段1：预处理
python src/ingest.py \
  --in ./data/arxiv-metadata-oai-snapshot.json \
  --out ./data/clean.jsonl

# 阶段2：索引构建（BM25）
python src/index_bm25.py \
  --in ./data/clean.jsonl \
  --index ./data/bm25.idx

# 阶段2：索引构建（稠密向量库 - 以 Chroma 为例）
python src/index_dense.py \
  --in ./data/clean.jsonl \
  --db ./data/chroma \
  --model bge-small-en-v1.5

# 阶段3：检索（混合）
python -c "from src.retriever import retrieve;print(retrieve('contrastive learning', 5, 'hybrid', 0.5))"

# 阶段4：RAG 生成
python -c "from src.rag import answer; print(answer('What is contrastive learning?', 'hybrid', 5))"

# 阶段5：离线评估
python src/eval.py \
  --qa ./data/dev_qa.jsonl \
  --modes bm25 dense hybrid \
  --out ./logs/metrics.csv
```

---

## ⚙️ 配置

在 `configs/config.yaml` 配置数据路径、索引与检索、LLM、运行参数。示例：

```yaml
data:
  raw: ./data/arxiv-metadata-oai-snapshot.json
  clean: ./data/clean.jsonl

bm25:
  index_path: ./data/bm25.idx
  language: en   # en/zh
  ngram: 1

dense:
  db: ./data/chroma
  provider: chroma  # chroma/faiss
  model: bge-small-en-v1.5

retrieval:
  mode: hybrid     # bm25/dense/hybrid
  topk: 5
  alpha: 0.5       # hybrid 加权

generation:
  provider: openai # openai/ollama/local
  model: gpt-4o-mini
  max_tokens: 512

runtime:
  max_context_chars: 6000
  seed: 42
```

环境变量（示例）：

```bash
# OpenAI 兼容后端
export OPENAI_API_KEY=your-key
export OPENAI_BASE_URL=https://api.openai.com/v1

# 本地 Ollama（可选）
export OLLAMA_HOST=http://localhost:11434
```

> 模型建议：英文摘要默认 `bge-small-en-v1.5`；多语/中英混用可选 `bge-m3` 或 `bge-small-zh-v1.5`。

---

## 🧩 模块与脚本

- `src/ingest.py`
  - `stream_clean_arxiv(in_path, out_path)`: 流式读取原始 JSONL，抽取 `id/title/abstract/categories/created`，清洗后写入 `clean.jsonl`

- `src/index_bm25.py`
  - `build_bm25_index(clean_path, index_path)`: 基于 `title+abstract` 构建倒排索引
  - `search_bm25(index_path, query, topk)`: 关键词检索，返回 `[Doc{id,title,score,...}]`

- `src/index_dense.py`
  - `embed_and_build_vector_db(clean_path, db_path, model)`: 生成向量并持久化至 Chroma/FAISS
  - `search_dense(db_path, query, topk, model)`: 语义检索，返回 `[Doc{...}]`

- `src/retriever.py`
  - `retrieve(query, topk, mode="bm25"|"dense"|"hybrid", alpha=0.5)`: 两路检索与加权融合

- `src/rag.py`
  - `build_context(docs, max_chars)`: 拼接 `Title + Abstract`，分隔符与长度裁剪
  - `answer(query, mode, topk)`: 调用 LLM 生成答案，附 `citations=[{id,title}...]`

- `src/eval.py`
  - `evaluate(qa_path, modes, out_csv)`: 评估 Recall@k/MRR/Latency 并导出 CSV

---

## 🧪 测试与冒烟

建议至少包含：

- `ingest`：随机抽 100 行验证字段完整性与清洗效果
- `index_bm25/index_dense`：小样本索引 + 查询返回非空
- `retriever`：`bm25/dense/hybrid` 一致性（同 `topk`）
- `rag.answer`：能返回答案与引用字段

示例（Python 一行冒烟）：

```bash
python -c "from src.ingest import stream_clean_arxiv; print('OK')"
python -c "from src.index_bm25 import search_bm25; print('OK')"
python -c "from src.index_dense import search_dense; print('OK')"
python -c "from src.retriever import retrieve; print(retrieve('graph neural networks', 3, 'bm25'))"
python -c "from src.rag import answer; print(answer('What is contrastive learning?', 'hybrid', 5))"
```

---

## 📊 数据说明

- 文件：`./data/arxiv-metadata-oai-snapshot.json`（JSON Lines，UTF-8）
- 重要字段：`id/title/abstract/categories/versions[0].created`
- 派生字段：
  - `primary_category = categories.split()[0]`
  - `year` 由 `created` 解析

读取示例：

```python
import json
path = "./data/arxiv-metadata-oai-snapshot.json"
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3: break
        rec = json.loads(line)
        print({
            "id": rec.get("id"),
            "title": rec.get("title"),
            "categories": rec.get("categories"),
            "created": (rec.get("versions") or [{}])[0].get("created")
        })
```

---

## 📈 评估与指标

- 评测集格式：`data/dev_qa.jsonl`

```json
{"q":"What is contrastive learning?","gold_ids":["0704.0001","2101.12345"]}
```

- 指标：
  - Recall@k、MRR（检索有效性）
  - Latency（检索/生成/端到端）

- 运行：

```bash
python src/eval.py --qa ./data/dev_qa.jsonl --modes bm25 dense hybrid --out ./logs/metrics.csv
```

- 输出：`logs/metrics.csv`（示例字段）

```
mode, k, recall, mrr, search_ms, gen_ms, end2end_ms
bm25,5,0.62,0.41,45,380,460
dense,5,0.66,0.44,70,375,495
hybrid,5,0.71,0.48,85,378,520
```

> 调参与建议：对 `alpha ∈ [0.3, 0.7]` 网格搜索；k ∈ {5, 10}；记录并对比三种 Latency。

---

## 🧱 项目结构

```
.
├── configs/
│   └── config.yaml
├── data/
│   ├── arxiv-metadata-oai-snapshot.json
│   ├── clean.jsonl
│   ├── bm25.idx
│   └── chroma/
├── logs/
│   └── metrics.csv
├── src/
│   ├── ingest.py            # stream_clean_arxiv
│   ├── index_bm25.py        # build_bm25_index / search_bm25
│   ├── index_dense.py       # embed_and_build_vector_db / search_dense
│   ├── retriever.py         # retrieve(mode="bm25"/"dense"/"hybrid")
│   ├── rag.py               # build_context / answer
│   └── eval.py              # evaluate
└── requirements.txt
```

---

## 🧾 依赖（建议）

```txt
ijson
tqdm
pyyaml
rank-bm25
nltk
chromadb
faiss-cpu; sys_platform == 'darwin'
sentence-transformers
# 或 FlagEmbedding（若使用 bge-* 官方实现）
# FlagEmbedding

numpy
pandas
```

> 分词：英文用 `nltk`；如摘要包含中文，可增补 `jieba`。

---

## 🖥️ 示例输出

```
================================================================================
[Q] What is contrastive learning?

🔎 检索阶段 - Retrieval (hybrid, topk=5, alpha=0.5)
  ✓ BM25 命中 5 篇，Dense 命中 5 篇
  ✓ 分数归一化 + 加权融合完成
  Top-3: [0704.0001] [2101.12345] [1806.01234]

📚 上下文构建 - Build Context
  ✓ 拼接 Title + Abstract (截断至 6000 chars)

🧠 生成阶段 - Generation
  ✓ provider=openai, model=gpt-4o-mini, max_tokens=512
  → 答案: Contrastive learning aims to ...

🔗 引用 - Citations
  - 0704.0001: An Example Paper Title
  - 2101.12345: Another Example Title
```

---

## ❓ FAQ

- 数据很大内存不够？
  - 使用流式读取，仅抽字段；先在小样本上构建 `clean.jsonl` 做验证。

- 选 BM25 还是稠密？
  - 术语明确/关键词强 → BM25；语义泛化/变体多 → 稠密；混合通常更稳。

- 中文/多语支持？
  - 稠密模型换为多语（`bge-m3`）；BM25 切换中文分词器（如 `jieba`）。

- LLM 超时或不可用？
  - 检查 `OPENAI_BASE_URL`、代理与 Key；或切换至本地 Ollama。

---

## ✅ 验收清单

- 已生成 `data/clean.jsonl`
- 已构建 `data/bm25.idx` 与 `data/chroma/`
- `retrieve` 在 bm25/dense/hybrid 下均能返回 Top-k
- `rag.answer` 返回答案与 `citations`
- `logs/metrics.csv` 产出 Recall@k/MRR/Latency

---

## 🧠 创新思路与课堂 Presentation 方案
 
 ### 创新方向（问题 → 方法 → 指标目标）
 
 - **[动态混合权重：自适应 alpha]**
   - 问题：固定 `alpha` 难以同时适配术语型（偏关键词）与语义型（偏语义）查询，易出现 query-style mismatch。
   - 方法：基于查询特征（长度、标点/数字占比、OOV 率、BM25/Dense 预分差等）做轻量分类，按类型动态设定 `alpha`。
   - 指标目标：在不明显增加开销的前提下，提升 Recall@k/MRR 的稳定性与均衡性。
 
 - **[跨编码器重排（Rerank）]**
   - 问题：初排 Top-k 相关性不足，语义相近项排序不稳，影响最终引用质量。
   - 方法：对 Top-N 使用 Cross-Encoder（如 `bge-reranker-base` 或 `cross-encoder/ms-marco-MiniLM-L-6-v2`）进行精排，输出更稳健的 Top-k。
   - 指标目标：显著提升 MRR/NDCG 与引用准确性；注意会引入额外延迟，可通过减小 N 或缓存控制。
 
 - **[查询扩展（QE：PRF + LLM 改写）]**
   - 问题：同义表达与长尾词导致召回不足，Recall@k 受限。
   - 方法：PRF 从初次检索的 Top-M 中提取高权词（RM3 风格）；并行生成 2–3 个等价问法，多查询检索后归一化融合与去重。
   - 指标目标：提升 Recall@10/覆盖率；检索并发带来额外开销，需要合并去重以控制延迟。
 
 - **[证据片段高亮（句级）]**
   - 问题：上下文冗长、答案可解释性弱，难以定位依据。
   - 方法：对命中文档做句级打分，抽取 Top-句拼接上下文；在答案中高亮证据句并给出 id/title 来源。
   - 指标目标：提升可解释性与可读性；缩短上下文有望降低生成延迟与噪声。
 
 - **[类别感知检索]**
   - 问题：跨学科噪声引入 off-topic 命中，影响准确性与精确率。
   - 方法：基于 arXiv `categories` 与查询类别先验进行筛选或加权融合。
   - 指标目标：提升准确性/精确率，减少无关文档带来的干扰。
 
 - **[SLA 驱动模式选择]**
   - 问题：不同场景对延迟与效果的权衡不同，固定检索策略难以满足 SLA。
   - 方法：根据 latency 预算自动选择 `bm25/dense/hybrid(+rerank)` 的组合与参数（如 topN、是否启用 rerank）。
   - 指标目标：在给定预算下实现最优效果–延迟权衡，稳定满足 SLA。

### 对应实现流程（分阶段可落地）

1) 动态混合权重（轻量）
  - 问题：固定 alpha 难以兼顾术语型/语义型查询，存在 query-style mismatch。
  - 方法：提取查询特征（长度、符号/数字占比、OOV 率、BM25 vs Dense 初分差等），规则+LogReg 分类，按类型自适应 alpha。
  - 实现步骤：
    - 新增 `src/heuristics.py: classify_query(q)->{"type","alpha","features"}`
    - 修改 `src/retriever.py: retrieve(..., alpha=None)` 时调用分类器返回动态 alpha
    - 配置开关：`configs/config.yaml -> retrieval.dynamic_alpha: true`
  - 评估与目标：Recall@k/MRR 在术语型/语义型两类上更均衡；延迟≈不变。

2) Rerank 精排（中等工作量）
  - 问题：初排 Top-k 相关性不足，影响引用质量。
  - 方法：对 Top-N 使用 Cross-Encoder 精排（如 `bge-reranker-base`）。
  - 实现步骤：
    - 新增 `src/reranker.py: rerank(query, docs, model)->scored_docs`
    - `retrieve(..., rerank_topn=50, enable_rerank=true)`：初排取 N，再 rerank 取 k
    - 依赖：`sentence-transformers` 的 CrossEncoder（requirements 已覆盖）
  - 评估与目标：MRR/NDCG↑，引用准确率↑；延迟↑但受 N 控制，可缓存减负。

3) 查询扩展（PRF + LLM 改写）（中等）
  - 问题：同义表达、长尾词导致召回不足。
  - 方法：PRF 提取高权词（RM3 风格）+ 生成 2–3 个等价问法并行检索，归一化融合与去重。
  - 实现步骤：
    - 新增 `src/expansion.py: prf_terms(top_docs)->tokens; llm_expand(q)->variants`
    - 检索聚合：对 [q + PRF + LLM variants] 并行检索，合并去重并归一化分数
    - 配置：`retrieval.expansion: {prf: true, llm: true, n_variants: 3}`
  - 评估与目标：Recall@10↑，新增命中占比↑；端到端延迟↑（可通过并发/限流控制）。

4) 证据片段高亮（可选）
  - 问题：上下文冗长、可解释性弱。
  - 方法：句级打分抽取 Top-句拼接上下文，答案中高亮证据并附来源。
  - 实现步骤：
    - 新增 `src/snippets.py: sentence_split(text); score_sentences(query, doc)`
    - `rag.build_context` 支持句级选择；`rag.answer` 返回 `evidence=[{"id","title","sentence"}]`
  - 评估与目标：可解释性↑，上下文长度↓，生成延迟↓。

5) 类别感知与 SLA 策略（扩展）
  - 问题：跨学科噪声与延迟预算不一致。
  - 方法：类别先验加权/过滤；按 `latency_budget_ms` 自动选择检索/重排组合。
  - 实现步骤：
    - `src/heuristics.py: predict_category(q)`；配置 `retrieval.allowed_categories`/`category_boost`
    - 脚本与代码增加 `--latency_budget_ms`，在检索阶段根据预算启/停 rerank、调节 topN
  - 评估与目标：精确率↑；在预算内稳定满足 end2end_ms。

### Presentation 结构（12–15 分钟）

- 1min 目标与约束：任务定义、评估指标（Recall@k/MRR/NDCG/Latency）与 SLA 预算
- 2min 问题与数据：基线现状与主要痛点（术语型 vs 语义型、召回不足、排序不稳、证据不可见）
- 2min 系统概览：流程图（ingest→index→retrieve→rag→eval），配置与脚手架
- 4min 方法与实现（逐项“问题→方法→实现→指标目标”）：
  - 动态 alpha：特征、分类器、集成点；期望 Recall/MRR 更均衡
  - Rerank：Top-N 精排与 N 的权衡；期望 MRR/NDCG 提升
  - QE：PRF+LLM 并行与融合；期望 Recall@10 提升
  - 证据高亮：句级抽取与可解释性改进
- 3min 实验与消融：
  - 表格/曲线：Baseline → +alpha → +rerank → +QE → All 的指标变化与延迟对比
  - 分组统计：术语型/语义型；展示典型成功/失败案例
- 1min Demo：`scripts/run_pipeline.sh`（离线 mock 兜底），展示答案+引用（+证据）
- 1min 成本与权衡：索引大小、内存/显存、构建与查询时延；SLA 策略
- 1min 结论与展望：多语、结构化答案、在线学习

### Demo 清单与风险规避

- 预先构建：`data/bm25.idx` 和 `data/chroma/`，避免现场构建耗时
- 首次运行下载模型耗时：提前 `pip install -r requirements.txt` 并预拉取 S-BERT 与 reranker
- 无网/代理问题：将 `generation.provider` 设为 `mock`，演示检索与流程；或使用本地 Ollama
- 兜底：准备截图与 metrics.csv，以防现场波动

---


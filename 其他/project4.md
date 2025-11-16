我来直接把 [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0) 改写成一个**独立项目说明**：不再出现“在 v3 基础上”之类表述，而是从零完整描述这个多策略 RAG 系统，同时保留你现在文档里的结构、创新点和技术细节。我会先读取当前 [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0) 的完整内容，然后用编辑工具在原文件上替换开头和相关表述。

```markdown
# 项目说明文档（加强版多策略 RAG 学术问答系统）

> 本项目构建了一个**基于 arXiv 论文摘要的学术问答系统**，采用检索增强生成（RAG）架构，结合传统 BM25、稠密向量检索和多种策略增强模块（动态权重、重排、查询扩展、证据高亮、类别感知、SLA 控制），在复杂查询场景下兼顾效果、延迟和可解释性。

---

## 0. 项目总览

- **项目目标**  
  利用 arXiv 论文摘要数据构建一个“懂论文的问答助手”：
  - 用户输入一个学术问题；
  - 系统在论文摘要库中检索相关论文；
  - 基于检索结果构造上下文并调用大语言模型生成答案；
  - 同时给出引用的论文 id/title 以及支撑答案的关键句。

- **系统特性**
  - 支持 **BM25 / 稠密向量 / Hybrid** 三种检索模式；
  - 支持 **动态混合权重、自适应检索策略**；
  - 支持 **Cross-Encoder 重排** 提升排序质量；
  - 支持 **PRF + LLM 查询扩展** 提升召回；
  - 支持 **句级证据抽取与高亮** 提升可解释性；
  - 支持 **类别感知检索** 减少跨学科噪声；
  - 支持 **SLA（延迟预算）驱动的策略选择**，在效果与延迟之间自动权衡。

- **工程定位**
  - 从 4.6GB 原始 JSONL 数据到可运行的 RAG 系统的完整工程路线；
  - 所有策略模块均配置化，可单独开启/关闭，方便消融实验与课堂展示。

---

## 1. 课程任务与项目目标

### 1.1 课程题目与要求

- 课程主题：**面向特定领域问答的检索增强生成（RAG）**
- 核心要求：
  - 选择一个特定领域的语料库（本项目使用 **arXiv 论文摘要**）；
  - 使用向量数据库（本项目选用 **Chroma**）构建稠密检索；
  - 将检索流程与开源大语言模型集成（如 Qwen、LLaMA 或任一 OpenAI 兼容模型）；
  - 从准确率、召回率、延迟等维度，对比稠密检索与传统 BM25。

### 1.2 本项目的具体目标

- **功能目标**
  1. 构建一个基于 arXiv 摘要的 RAG 学术问答系统；
  2. 在检索层面支持 BM25、Dense、Hybrid 三种模式，并实现多策略增强：
     - 动态混合权重（自适应 α）；
     - Cross-Encoder 重排；
     - PRF + LLM 查询扩展；
     - 句级证据片段高亮；
     - 类别感知检索；
     - SLA 驱动的模式选择；
  3. 基于合成问答数据集和（可选）真实标注问答，在不同策略组合下评估 Recall@k、MRR 和延迟。

- **范围与非目标**
  - 仅使用 arXiv 摘要和基础元数据，不解析 PDF 全文；
  - 不微调大语言模型，仅调用现成的开源或云端 LLM；
  - UI 采用命令行/脚本形式，不做复杂 Web 前端；
  - 不涉及用户系统、多租户、鉴权等工程化运维问题。

---

## 2. 数据与基础系统

### 2.1 数据来源与格式

- **数据源**：`./data/arxiv-metadata-oai-snapshot.json`  
  - 大小约 4.6GB；
  - 格式：JSON Lines（JSONL），即每行一个完整论文元数据对象。
- **单条数据示例**（解析后）：

```json
{
  "id": "0704.0001",
  "submitter": "John Doe",
  "authors": "John Doe, Jane Smith",
  "title": "An Example Paper Title",
  "comments": "12 pages, 3 figures",
  "journal-ref": "J. Example 12 (2007) 34-56",
  "doi": "10.1000/example.doi",
  "categories": "cs.CL stat.ML",
  "abstract": "We propose an approach to ...",
  "versions": [
    { "version": "v1", "created": "Mon, 2 Apr 2007 12:00:00 GMT", "source": "arXiv" }
  ],
  "update_date": "2007-04-10"
}
```

- **预处理后保留字段**：
  - `id`：论文唯一标识；
  - `title`：论文标题；
  - `abstract`：摘要文本（可能含 LaTeX）；
  - `categories`：空格分隔的学科标签；
  - `created`：从 `versions[0].created` 解析出的时间字符串。

### 2.2 预处理流程（ingest.py）

- 使用 `ingest.py` 对原始 JSONL 做流式预处理：
  - 逐行读取原始文件，避免一次性加载 4.6GB；
  - 解析 JSON，抽取 `id/title/abstract/categories/created`；
  - 对摘要进行轻量清洗（去换行、简单去除 LaTeX）；
  - 输出到 `data/clean.jsonl`，一行一篇论文。

### 2.3 基础模块与项目结构

```text
5020-RAG/
├── data/
│   ├── arxiv-metadata-oai-snapshot.json      # 原始 JSONL (4.6GB)
│   ├── clean.jsonl                           # [阶段1] 精简字段
│   ├── bm25.idx                              # [阶段2] BM25 索引
│   ├── chroma/                               # [阶段2] 稠密向量库
│   └── synth_qa.jsonl                        # 合成问答评测集
├── src/
│   ├── ingest.py                             # 数据预处理
│   ├── index_bm25.py                         # 构建/查询 BM25 索引
│   ├── index_dense.py                        # 构建/查询 稠密索引 (Chroma)
│   ├── retriever.py                          # 统一检索接口 (bm25/dense/hybrid + enhanced)
│   ├── rag.py                                # RAG 上下文构建 + LLM 调用
│   ├── eval.py                               # 离线评测 (检索+生成)
│   ├── synth_qa.py                           # 合成问答数据生成
│   ├── heuristics.py                         # 查询/类别/策略 轻量启发式
│   ├── reranker.py                           # Cross-Encoder 重排
│   ├── expansion.py                          # 查询扩展 (PRF + LLM)
│   └── snippets.py                           # 句级证据抽取与高亮
├── configs/
│   └── config.yaml                           # 数据/模型/检索/策略 配置
├── logs/
│   ├── metrics_*.csv                         # 各模式评估结果
│   └── traces/                               # [可选] SLA & 延迟日志
└── requirements.txt
```

基础模块实现了一个标准的 RAG 流程；后续的创新模块作为可选组件插在这条流程的关键节点。

---

## 3. 系统总流水线（含增强模块）

### 3.1 离线流水线

```text
原始 JSONL (4.6GB)
   ↓ ingest.py
clean.jsonl  (精简字段)
   ↓ index_bm25.py
bm25.idx     (BM25 索引)
   ↓ index_dense.py
chroma/      (稠密向量库)
   ↓ synth_qa.py (可选)
synth_qa.jsonl (合成 QA 数据集)
   ↓ eval.py
metrics_*.csv (不同检索/策略组合的评估结果)
```

说明：

- 离线阶段主要负责处理数据、构建索引、生成合成 QA，并在需要时对不同策略组合进行批量评测。

### 3.2 在线查询阶段（增强版流程）

```text
用户 query
  ↓
heuristics.classify_query(q)
  # 轻量分析查询特征 → 查询类型 + 动态 alpha 等
  ↓
expansion.expand_query(q)
  # 可选：PRF + LLM 生成多个等价/扩展问法
  ↓
retriever.retrieve_enhanced(
    base_queries=[q, q1, q2, ...],
    mode=bm25/dense/hybrid,
    dynamic_alpha=on/off,
    category_filter/boost,
    latency_budget_ms
) → 初排候选 Top-N
  ↓
reranker.rerank(query, docs)
  # Cross-Encoder 对 Top-N 精排，取 Top-k
  ↓
snippets.select_evidence(query, docs)
  # 对命中文档做句级打分，选出 evidences
  ↓
rag.enhanced_answer(query, docs, evidences)
  # 基于 evidence 或完整摘要构造上下文 + 调用 LLM
  → answer + citations + evidence
```

说明：

- Online pipeline 将多个策略模块串联起来，每个模块都可通过配置开关控制；
- `retrieve_enhanced` 和 `enhanced_answer` 是对基础 `retrieve` 和 `answer` 的扩展，支持更多策略组合。

---

## 4. 创新模块 1：动态混合权重（自适应 alpha）

### 4.1 问题与直觉

- 混合检索（Hybrid）需要一个权重 `alpha` 用来融合 BM25 与稠密向量的分数：
  - `score_final = alpha * score_dense + (1 - alpha) * score_bm25`；
- 如果 `alpha` 固定，会存在 **query-style mismatch**：
  - 术语密集、缩写较多的问题（“术语型查询”）更需要依赖 BM25；
  - 自然语言长句、表述模糊的问题（“语义型查询”）更需要依赖稠密向量；
  - 固定权重难以匹配两类查询的不同需求。

### 4.2 方法概述

- 为每个 query 提取一组简单特征：
  - 字符长度 / token 数；
  - 数字/符号占比；
  - 大写字母或缩写比例等；
- 使用简单规则将查询划分为三类：
  - `term-heavy`（术语型）；
  - `semantic`（语义型）；
  - `mixed`（混合型）；
- 根据类型设定 `alpha`：
  - 术语型：`alpha_dense` 偏小，如 0.2（更信 BM25）；
  - 语义型：`alpha_dense` 偏大，如 0.8（更信 Dense）；
  - 混合型：`alpha_dense` 约 0.5（折中）。

### 4.3 模块与接口设计

- 文件：`src/heuristics.py`
  - 核心函数：`classify_query(q: str) -> Dict`
    - 输出示例：

```python
{
  "type": "term" | "semantic" | "mixed",
  "alpha": 0.2,
  "features": {...}
}
```

- 文件：`src/retriever.py`
  - 在增强版入口 `retrieve_enhanced` 中，当 `use_dynamic_alpha=True 且 alpha=None` 时：
    - 调用 `classify_query(q)` 获取建议 alpha；
    - 用该 alpha 做 hybrid 融合。

- 配置项（`configs/config.yaml`）：

```yaml
retrieval:
  mode: hybrid
  topk: 5
  alpha: 0.5           # 默认 alpha
  dynamic_alpha: true  # 启用自适应 alpha
```

### 4.4 评估与指标

- 可以在 QA 集中人为区分：
  - 术语型查询子集 / 语义型查询子集 / 混合型查询子集；
- 指标：
  - 子集上的 Recall@k 和 MRR；
  - 与固定 alpha 情况下的差异；
  - 延迟变化（理论上几乎不变，因为 `classify_query` 非常轻量）。

---

## 5. 创新模块 2：跨编码器重排（Rerank）

### 5.1 问题与直觉

- 初排 Top-k 由 BM25/Dense/Hybrid 生成，速度快但排序质量有限；
- 对于 RAG，**前几篇文档的排序质量直接决定答案和引用质量**；
- 希望在初排 Top-N 的基础上，请更强的模型进行二次精排。

### 5.2 方法概述

- 使用 Cross-Encoder 模型（如：
  - `bge-reranker-base`；
  - 或 `cross-encoder/ms-marco-MiniLM-L-6-v2` 等）；
- 对 `query` 与每个候选 `doc.text` 组成一个输入对，让模型直接输出相关性分数；
- 对 Top-N 候选按该分数排序，取前 k 个作为最终结果。

### 5.3 模块与接口设计

- 文件：`src/reranker.py`

```python
def load_reranker(model_name: str):
    """初始化或缓存 CrossEncoder 模型。"""

def rerank(
    query: str,
    docs: List[Dict],
    model_name: str,
    topk: int,
    batch_size: int = 16,
) -> List[Dict]:
    """
    输入：query + 初排 docs 列表
    输出：按 Cross-Encoder 打分后的 Top-k docs（增加 'rerank_score' 字段）
    """
```

- 文件：`src/retriever.py` / `src/rag.py`
  - 在 `retrieve_enhanced` 中集成：
    - 初排取 Top-N（例如 50）；
    - 若 `enable_rerank=True`：
      - 调用 `rerank(query, topN_docs, model_name, topk)`；
    - 否则直接返回初排 Top-k。

- 配置项：

```yaml
rerank:
  enable: false
  model: bge-reranker-base
  topn: 50       # 初排取前 N 再重排
  batch_size: 16
```

### 5.4 评估与指标

- 对比：
  - 不使用 rerank 与使用 rerank 的差异；
- 指标：
  - MRR / NDCG（排序质量）；  
  - 正确引用的文献比例；  
  - 检索延迟（随 topN 增大而增加，需要选择合适的 N）。

---

## 6. 创新模块 3：查询扩展（PRF + LLM 改写）

### 6.1 问题与直觉

- 用户 query 可能过短或表达模糊：
  - 信息不足；
  - 使用长尾词或非主流说法；
- 直接检索时容易出现召回不足，Recall@k 上限偏低；
- 希望自动“帮用户多想几种问法”，扩大检索覆盖面。

### 6.2 方法概述

- **PRF（Pseudo Relevance Feedback）**：
  - 用原始 query 检索一次，取初排 Top-M 文档；
  - 从这些文档中统计高频、代表性的关键词（类似传统 RM3）；
  - 将这些词作为扩展词拼回 query。
- **LLM 改写**：
  - 使用 LLM 读取原始 query；
  - 生成若干个语义等价或更具体的问法。
- 综合起来：
  - 形成一个 `query variants` 集合：`[q_original, q_prf, q_llm1, q_llm2, ...]`；
  - 对每个变体分别检索；
  - 将结果合并，对相同 `id` 的分数进行归一化和融合，并去重。

### 6.3 模块与接口设计

- 文件：`src/expansion.py`
  - PRF 接口：

```python
def prf_terms(
    query: str,
    retriever_config: Dict,
    m_docs: int = 5,
    top_terms: int = 10,
) -> List[str]:
    """使用当前检索后端，对 query 做一次初排，从 top-m 文档中统计关键词。"""
```

  - LLM 改写接口（可复用 `rag.LLMClient`）：

```python
def llm_expand(
    query: str,
    client: LLMClient,
    n_variants: int = 3,
) -> List[str]:
    """使用 LLM 生成 n 个等价或更明确的问法。"""
```

  - 综合扩展：

```python
def expand_query(
    query: str,
    client: Optional[LLMClient],
    use_prf: bool = True,
    use_llm: bool = True,
) -> List[str]:
    """综合 PRF 和 LLM 改写，返回若干 query 变体。"""
```

- 文件：`src/retriever.py`
  - 在 `retrieve_enhanced` 中，当 `enable_expansion=True`：
    - 调用 `expand_query` 获取 query 变体；
    - 对每个变体进行检索；
    - 合并结果并去重。

- 配置项：

```yaml
expansion:
  enable: false
  prf:
    enable: true
    m_docs: 5
    top_terms: 10
  llm:
    enable: true
    n_variants: 3
```

### 6.4 评估与指标

- 指标关注点：
  - Recall@10 / Recall@20 提升幅度；
  - 通过变体检索才命中的样本比例；
  - 检索延迟增量（可通过变体数量与并行策略控制）。

---

## 7. 创新模块 4：证据片段高亮（句级）

### 7.1 问题与直觉

- 标准 RAG 通常将整个摘要拼接成上下文：
  - 上下文可能很长，包含大量无关信息；
  - 用户看到答案时，很难定位“答案依据是哪一句话”；
- 希望：
  - 从摘要中抽取“最有可能回答问题的句子”，构造更紧凑、聚焦的上下文；
  - 在答案中明确给出这些句子和对应论文，提升可解释性。

### 7.2 方法概述

- 对每篇文档的 `text`（title + abstract）做句子切分；
- 对每个句子与 query 的相关性进行打分：
  - 简单实现：基于关键词重合度（共有词数量）；
  - 进阶实现：基于嵌入相似度（query 与句子向量余弦相似度）；
- 选出每篇文档中得分最高的若干句作为该文档的 evidence；
- 将多个文档的 evidence 汇总，构建上下文并附带 evidence 元信息。

### 7.3 模块与接口设计

- 文件：`src/snippets.py`

```python
def sentence_split(text: str) -> List[str]:
    """按标点将文本切分为句子。"""

def score_sentences(
    query: str,
    doc: Dict,
    max_sentences: int = 3,
    method: str = "bm25"
) -> List[Dict]:
    """对单个文档中的句子进行打分，返回若干 {"sentence", "score"}。"""

def select_evidence_for_docs(
    query: str,
    docs: List[Dict],
    per_doc: int = 2,
    max_total: int = 10,
) -> List[Dict]:
    """
    针对多个文档选取句级 evidence。
    返回 [{"id": doc_id, "title": title, "sentence": s, "score": ...}, ...]
    """
```

- 文件：`src/rag.py`
  - 增强版上下文构造：

```python
def build_context_with_evidence(
    evidences: List[Dict],
    max_chars: int = 4000,
) -> str:
    """使用句级 evidence 构造上下文，而不是完整摘要。"""
```

  - 增强版回答接口：

```python
def enhanced_answer(
    query: str,
    mode: str = "hybrid",
    topk: int = 5,
    ...
) -> Dict:
    """
    返回：
    {
      "answer": ...,
      "citations": [...],
      "evidence": [{"id","title","sentence"}, ...]
    }
    """
```

- 配置项：

```yaml
rag:
  use_evidence_snippets: true
  evidence:
    per_doc: 2
    max_total: 10
    method: bm25
```

### 7.4 评估与指标

- 定性：
  - 人工检查 evidence 的合理性；
  - 答案引用部分是否更容易理解、可解释性是否增强。
- 定量：
  - 上下文总长度（字符/Token 数）是否下降；
  - 生成延迟是否减少；
  - 在某些情况下，生成质量也可能受益于更干净的上下文。

---

## 8. 创新模块 5：类别感知检索

### 8.1 问题与直觉

- arXiv 涉及多个学科：
  - 用户询问“图神经网络用于文本分类”，不希望命中物理、生物等无关领域；
- 希望利用 arXiv 的 `categories` 信息，加强领域相关性，减少 off-topic 文档。

### 8.2 方法概述

- 利用 query 和候选文档的类别信息，进行：
  - **过滤**：只保留指定类别或前缀（如仅 `cs.*`）；
  - **加权**：对类别匹配的文档增加一定分数。

### 8.3 模块与接口设计

- 文件：`src/heuristics.py`
  - 新增类别预测函数：

```python
def predict_query_category(query: str) -> Optional[str]:
    """
    通过简单规则或轻量模型，预测 query 的主类别前缀（如 "cs.CL"）。
    """
```

- 文件：`src/retriever.py`
  - 在初排/融合后，对文档列表应用类别逻辑：
    - 如果启用过滤：只保留 `primary_cat(doc)` 以允许前缀开头的文档；
    - 如果启用加权：对于类别匹配的文档，按配置对 score 做乘法加成；

示例伪逻辑：

```python
if category_cfg["enable_filter"]:
    docs = [d for d in docs if primary_cat(d) in allowed]
if category_cfg["enable_boost"] and query_cat:
    for d in docs:
        if primary_cat(d).startswith(query_cat_prefix):
            d["score"] *= (1.0 + boost_factor)
```

- 配置项：

```yaml
category:
  enable_filter: false
  allowed_prefixes: ["cs."]
  enable_boost: true
  boost_factor: 0.1
```

### 8.4 评估与指标

- 指标：
  - Precision@k（前 k 条结果中真正相关论文的比例）；
  - off-topic 文档比例；
- 通常需要人工抽样若干 query，检查命中文档是否属于期望学科。

---

## 9. 创新模块 6：SLA 驱动模式选择

### 9.1 问题与直觉

- 不同使用场景对“响应速度 vs 效果”的要求不同：
  - 演示/离线分析场景：可以接受较高延迟，换取更好效果；
  - 在线服务场景：端到端延迟需要控制在特定范围内（如 <1s）；
- 希望系统根据预设的延迟预算自动选择合适的检索策略组合，而非固定策略。

### 9.2 方法概述

- 在配置或函数调用中引入延迟预算参数：
  - `latency_budget_ms`：例如 300ms、800ms、1500ms；
- 在 `heuristics.choose_strategy` 中，根据预算返回策略组合：
  - 紧张预算：仅 BM25，不开启 QE 和 Rerank；
  - 中等预算：Hybrid + 部分扩展；
  - 宽松预算：Hybrid + Rerank + 证据抽取等。

### 9.3 模块与接口设计

- 文件：`src/heuristics.py`

```python
def choose_strategy(latency_budget_ms: int) -> Dict:
    """
    根据预算返回策略开关：
    {
      "mode": "bm25" | "dense" | "hybrid",
      "enable_rerank": bool,
      "enable_expansion": bool,
      ...
    }
    """
```

- 文件：`src/retriever.py` / `src/rag.py`
  - 在入口处读取 `latency_budget_ms`（来自配置或函数参数）；
  - 调用 `choose_strategy` 决定：
    - 检索模式（bm25/dense/hybrid）；
    - 是否启用 dynamic_alpha、expansion、rerank 等；
  - 后续模块根据该策略执行。

- 配置项：

```yaml
runtime:
  latency_budget_ms: 1000
```

### 9.4 评估与指标

- 使用 `eval.py` 扩展统计：
  - `end2end_ms`：整条 pipeline 的端到端平均延迟；
- 在不同预算配置下对比：
  - 是否满足延迟约束；
  - Recall/MRR 在约束下的变化；
- 可用三档预算（如 300 / 800 / 1500ms）做演示，展示系统如何自动在速度和效果间做 trade-off。

---

## 10. 总结：系统贡献与教学价值

- **从“单一 RAG 系统”到“多策略可调的实验平台”**
  - 在完整 RAG 流程上增加多个策略层模块，形成可配置、可组合、可消融的实验平台；
  - 动态权重和查询扩展展示了如何根据 query 特征和反馈机制改进混合检索；
  - Cross-Encoder 重排展示了“效果与延迟”的典型权衡实践；
  - 证据高亮和类别感知检索强化了学术问答的可解释性和领域相关性；
  - SLA 驱动模式选择则体现了工程系统中将“业务约束”显式引入算法策略的思路。

- **教学与实验价值**
  - 对学生：提供了一个完整的、从数据到模型、从检索到生成、从策略到评测的 RAG 项目模板；
  - 对实验：所有增强模块都有清晰的“问题→方法→模块→指标”的闭环，便于做系统性的消融与对比；
  - 对工程实践：展示了如何通过配置和启发式策略，让一个 RAG 系统在不同需求场景下灵活适配。

---

## 11. 面向小白的专业术语解释（小白词典）

> 说明：  
> 下面这些术语，是整个项目文档里经常出现的关键词。  
> 假设读者完全不懂计算机/机器学习，也可以把这一节当成“概念挂件”来查。

```
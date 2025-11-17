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

### 2.4 模型 API 与部署配置（支持 OpenAI 兼容接口 + .env 管理）

- **调用方式**  
  本项目在 `src/rag.py` 中通过一个 `LLMClient` 类来封装大模型调用逻辑，内部支持三种 provider：
  - `mock`：不发真实请求，只返回伪造答案，方便在没有 API Key 时调试整个 RAG 流程；
  - `openai`：使用官方 OpenAI API；
  - `ollama` 或其他 OpenAI 兼容服务：只要提供与 OpenAI 相同的 HTTP 接口（`/v1/chat/completions`），即可复用同一套代码。

- **配置位置（非敏感参数）**  
  模型相关的“非敏感配置”（如 provider / model / max_tokens）统一写在 `configs/config.yaml` 的 `generation` 小节，例如：

  ```yaml
  generation:
    provider: mock           # mock | openai | ollama（或其他兼容服务）
    model: gpt-4.1-mini      # OpenAI 模型名，或第三方兼容模型名
    max_tokens: 512
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
> 建议：先大概浏览一遍标题，有不懂的再随时翻到这里查。

---

### 11.1 整体概念相关

- **检索增强生成（RAG, Retrieval-Augmented Generation）**  
  RAG 的意思是：**先查资料，再让大模型根据资料回答问题**。  
  可以类比：你写作业时，会先去翻教材、论文（检索），再根据这些内容整理自己的答案（生成）。  
  在本项目里：系统先从 arXiv 论文库里找出相关摘要，再把这些摘要提供给大语言模型，让它“有依据地”回答，而不是凭空想象。

- **学术问答系统**  
  指专门针对“学术类问题”的问答系统，比如“最近图神经网络在文本分类上有哪些进展？”。  
  在本项目里：它的知识来源是 arXiv 论文摘要，而不是维基百科、网页等综合资料。

- **流水线 / Pipeline**  
  指一连串按顺序执行的步骤，好比生产线：原材料进来，经过加工、包装，最后变成成品。  
  在本项目里：从“原始 JSONL 数据 → 清洗 → 建索引 → 评测 → 在线问答”就是一条完整流水线。

- **策略 / 策略模块**  
  策略就是“系统在不同情况下怎么做选择”的规则，比如用哪种检索方式、要不要做重排、要不要扩展查询。  
  在本项目里：动态混合权重、重排、查询扩展、类别感知、SLA 控制都属于“策略模块”。

- **消融实验**  
  意思是“关掉/打开某个模块，看效果差多少”，从而判断这个模块是不是有用。  
  在本项目里：可以单独关闭重排、关闭查询扩展等，比较前后指标变化。

---

### 11.2 数据与文件格式

- **arXiv**  
  一个收录全球大量预印本论文的网站，主要是数学、计算机、物理等学科。  
  在本项目里：我们使用的是 arXiv 提供的论文元数据快照，其中包含每篇论文的标题、摘要、类别等。

- **摘要（Abstract）**  
  论文开头的一段“浓缩版说明书”，快速告诉你这篇论文做了什么。  
  在本项目里：系统只使用每篇论文的摘要，而不解析 PDF 全文。

- **元数据（Metadata）**  
  “描述数据的数据”，比如论文的 `id`、作者、标题、类别、提交时间等。  
  在本项目里：元数据主要用于标记和筛选论文（例如按类别过滤）。

- **JSON / JSONL（JSON Lines）**  
  - JSON：常见的数据格式，看起来有点像 Python 的字典。  
  - JSONL：一行一个 JSON 对象的文件格式，方便流式读取大文件。  
  在本项目里：原始 arXiv 数据是一个 4.6GB 的 JSONL 文件，每一行是一篇论文的元数据。

- **数据预处理 / 清洗**  
  把原始数据整理成更干净、更紧凑、方便后续使用的格式，比如只保留 `id/title/abstract/categories/created`，去掉多余字段、简单清理 LaTeX 符号。  
  在本项目里：由 `ingest.py` 完成，输出 `data/clean.jsonl`。

---

### 11.3 检索与索引相关

- **检索（Retrieval）**  
  从一个大文档库里“搜”出和用户问题最相关的若干篇文档。  
  在本项目里：检索的目标是从大量论文摘要中找出和问题最相关的几篇论文。

- **索引（Index）**  
  为了更快地查找而提前构建的“目录结构”或“搜索加速结构”。  
  类比：给书建好分门别类的目录和页码，以后就可以快速定位内容。  
  在本项目里：`bm25.idx` 是 BM25 索引，`chroma/` 是稠密向量索引。

- **BM25（传统关键词检索）**  
  一种经典的“关键词匹配”打分方法，是很多搜索引擎的基础算法。  
  简单理解：用户问题里的词，和文档的词重合越多、越“重要”，文档得分就越高。  
  在本项目里：`index_bm25.py` 构建并查询 BM25 索引，适合“术语型”问题。

- **稠密向量 / 稠密检索（Dense Retrieval）**  
  把一句话编码成一个长长的数字向量（比如几百维），通过比较向量之间的距离或相似度来判断语义相近程度。  
  类比：把每句话都放在一个高维坐标系里，距离越近表示含义越接近。  
  在本项目里：使用向量数据库（如 Chroma）存储和搜索这些向量。

- **Hybrid 检索（混合检索）**  
  同时利用 BM25（关键词匹配）和稠密检索（语义相似），再用一个权重 `alpha` 把两个分数融合。  
  简单理解：BM25 像“关键字搜索”，稠密检索像“意思相近搜索”，Hybrid 是把两者结合，兼顾精确术语和语义理解。

- **向量数据库（Vector DB，例如 Chroma）**  
  专门用来存储和检索“向量”的数据库，支持高效的相似度搜索。  
  在本项目里：`chroma/` 目录就是 Chroma 向量库的存储位置。

- **Top-k / Top-N**  
  - Top-k：取得分最高的前 k 条结果，例如 Top-5 表示取最相关的 5 篇论文。  
  - Top-N：常用于“先取前 N 再做二次处理”（如重排）。  
  在本项目里：初排可以取 Top-50，再交给重排模块挑出最终 Top-5。

- **查询类型：术语型 / 语义型 / 混合型**  
  - 术语型：包含很多缩写、符号和专业名词的问题，比如“GNN for text classification”。  
  - 语义型：更像自然语言长句，表达比较口语化。  
  - 混合型：两种成分都有。  
  在本项目里：`heuristics.classify_query` 会对 query 做简单判别，用于选择合适的权重 `alpha` 等策略。

---

### 11.4 大语言模型与生成相关

- **大语言模型（LLM, Large Language Model）**  
  类似 ChatGPT 这一类能理解和生成自然语言的大模型。  
  在本项目里：LLM 用来“读”检索到的论文摘要（或证据句），再生成自然语言答案。

- **LLM 查询扩展 / 改写**  
  让大语言模型帮用户“想出更多问法”，比如把“图神经网络应用”改写成几种更具体、更标准的提问。  
  在本项目里：`expansion.llm_expand` 会生成多个等价或更详细的 query 变体，提升召回率。

- **RAG 上下文构建**  
  指把检索到的文档内容整理成一个合理的“上下文文本”，再提供给大模型。  
  在本项目里：`rag.py` 负责把摘要或句级证据拼成提示词，让 LLM 在这些材料基础上回答问题。

- **证据（Evidence）**  
  支撑答案的原文片段，通常是具体的句子或段落。  
  在本项目里：系统会从论文摘要中选出若干句子作为证据，并和答案一起展示，提升可解释性。

- **引用（Citations）**  
  指答案中提到的论文 ID / 标题等信息，好比论文里的参考文献列表。  
  在本项目里：RAG 输出格式里会附带使用到的论文 id/title。

---

### 11.5 创新模块与策略

- **动态混合权重（自适应 alpha）**  
  在 Hybrid 检索中，`alpha` 控制“更信稠密向量”还是“更信 BM25”。  
  - 术语型问题：更依赖 BM25 → `alpha` 偏小  
  - 语义型问题：更依赖稠密检索 → `alpha` 偏大  
  在本项目里：`heuristics.classify_query` 会根据 query 特征给出一个合适的 `alpha`。

- **Cross-Encoder（跨编码器）重排（Rerank）**  
  一种更强的模型：把“问题 + 文档”一起输入模型，让模型直接输出“相关性分数”。  
  使用方式：先用 BM25/Dense 快速筛出 Top-N，再用 Cross-Encoder 把这 N 篇重新排序，挑出最终 Top-k。  
  在本项目里：`reranker.py` 实现这一模块，常用模型如 `bge-reranker-base` 等。

- **查询扩展（Query Expansion, 含 PRF + LLM）**  
  目标：帮用户“多想几种问法”，增加能命中相关论文的机会。  
  - PRF（伪相关反馈）：先用原始 query 检索一次，从初排文档里统计高频关键词，再把这些词拼回 query。  
  - LLM 改写：让大模型产生若干等价/更详细的问法。  
  在本项目里：`expansion.py` 组合 PRF 和 LLM 改写，生成一组 query 变体，然后分别检索并合并结果。

- **证据片段高亮（句级证据选择）**  
  不再把整篇摘要全部丢给大模型，而是从中挑出最可能回答问题的几句，作为“高亮证据”。  
  在本项目里：`snippets.py` 负责句子切分、打分和选择，`rag.build_context_with_evidence` 使用这些句子构造上下文。

- **类别感知检索**  
  利用 arXiv 的 `categories`（学科标签），避免检索出跨学科无关的论文。  
  典型做法：  
  - 只保留某些学科前缀（如 `cs.*`）。  
  - 对类别匹配的文档适当加分（boost）。  
  在本项目里：`heuristics.predict_query_category` 预测 query 的大致类别，`retriever.py` 里对结果做过滤或加权。

- **SLA（Service Level Agreement，延迟预算）驱动模式选择**  
  简单理解：提前约定“系统必须多快给出结果”，系统再根据这个时间限制选择不同的策略组合。  
  在本项目里：通过 `latency_budget_ms` 参数控制：  
  - 预算很紧：只用 BM25 等轻量策略；  
  - 预算稍宽松：可以启用 Hybrid、部分扩展；  
  - 预算很宽：可以再加上重排、证据抽取等耗时策略。

---

### 11.6 评测指标与日志

- **召回率 Recall@k**  
  在所有“真正相关”的论文里，有多少被系统的前 k 条结果命中。  
  类比：你在 100 个正确答案中，前 5 条里找到了其中 3 个，Recall@5 = 3%。  
  在本项目里：用于衡量检索层面的“漏掉多少好论文”。

- **MRR（Mean Reciprocal Rank，平均倒数排名）**  
  看“第一条正确结果通常在第几名”。  
  比如正确结果在第 1 名得分 1，在第 2 名得分 1/2，在第 5 名得分 1/5，再对所有问题取平均。  
  在本项目里：用来衡量排序质量，越高说明正确答案越靠前。

- **NDCG（Normalized Discounted Cumulative Gain）**  
  更细致的排序指标：不仅关心有没有命中，还关心多个相关结果的排序位置。  
  对小白来说：可以理解为“综合考虑多个相关论文的排序好坏”的得分。  
  在本项目里：主要在启用重排等策略时用于对比排序效果。

- **Precision@k（前 k 条的精确率）**  
  前 k 条结果中有多少是“真正相关”的。  
  类比：Top-5 里有 4 条是好论文，1 条无关，则 Precision@5 = 80%。  
  在本项目里：结合 Recall 一起看，可判断结果“纯度”。

- **端到端延迟（end2end_ms）**  
  从用户输入问题到收到最终答案，整个流程花的总时间（毫秒）。  
  在本项目里：`eval.py` 会统计整个 pipeline 的平均端到端延迟，用于验证是否满足 SLA 要求。

- **metrics_*.csv / logs/**  
  - `metrics_*.csv`：存放不同策略组合下的评测结果（Recall、MRR、延迟等）。  
  - `logs/traces/`：可选的详细日志，记录每次查询的运行轨迹和耗时。  
  方便后续分析和画图。

---

### 11.7 工程文件与脚本（怎么“跑起来”）

- **`ingest.py`（数据预处理）**  
  职责：从原始 `arxiv-metadata-oai-snapshot.json` 里逐行读取数据，抽取需要的字段并简单清洗，输出到 `data/clean.jsonl`。  
  可以理解为：把一大堆原始杂乱信息，整理成干净、统一格式的“干净表格”。

- **`index_bm25.py`（构建/查询 BM25 索引）**  
  职责：基于 `clean.jsonl` 构建 BM25 搜索索引，并提供查询接口。  
  作用：让系统可以用传统关键词方式快速搜索论文摘要。

- **`index_dense.py`（构建/查询稠密向量索引）**  
  职责：把论文摘要编码成向量，并存入 Chroma 向量库。  
  作用：支持“语义检索”，也就是根据含义相似度找论文。

- **`retriever.py`（统一检索接口）**  
  职责：对外提供一个统一的 `retrieve_enhanced` 接口，内部可以选择 bm25/dense/hybrid，并接入类别过滤、动态 alpha、查询扩展等策略。  
  作用：让上层代码不用关心底层细节，只管“我要检索”。

- **`rag.py`（RAG 流程与 LLM 调用）**  
  职责：负责把检索结果转成上下文，调用大语言模型生成答案，并整理输出格式（答案 + 引用 + 证据）。  
  作用：把“检索”和“生成”串成一条完整的 RAG 流水线。

- **`eval.py`（离线评测）**  
  职责：在一批已经标注好“标准答案”的问答样本上，批量跑系统，计算 Recall/MRR/延迟等指标。  
  作用：用数据说话，比较不同检索模式、不同策略组合的效果和速度。

- **`synth_qa.py`（合成问答数据集）**  
  职责：利用已有论文摘要和大模型自动构造一批“问题-答案-证据”的样本。  
  作用：在没有人工标注数据时，也能做评测和对比实验。

- **`heuristics.py`（启发式规则）**  
  职责：提供一些轻量规则函数，比如根据 query 判断类型、给出动态 alpha、选择策略组合、预测类别等。  
  作用：在不引入复杂模型的前提下，让系统“略带一点聪明”。

- **`reranker.py`（Cross-Encoder 重排）**  
  职责：加载重排模型，对初排出的 Top-N 文档计算更精细的相关性分数，并重新排序。  
  作用：确保真正相关的论文尽量排在最前面。

- **`expansion.py`（查询扩展）**  
  职责：实现 PRF 和 LLM 改写，将原始 query 扩展成多个变体。  
  作用：提升召回率，减少“搜不到”的情况。

- **`snippets.py`（句级证据抽取）**  
  职责：把文档拆成句子，对每个句子打分，挑出最有用的句子作为 evidence。  
  作用：生成更紧凑、可解释的上下文，并在答案中显示关键句。

- **`configs/config.yaml`（配置文件）**  
  用来集中管理各种参数：  
  - 用什么检索模式（bm25/dense/hybrid）；  
  - 是否打开动态 alpha / 重排 / 查询扩展 / 类别过滤；  
  - 延迟预算、Top-k 等。  
  好处：不用改代码就能切换策略组合，方便做实验。

- **`requirements.txt`（依赖列表）**  
  列出了运行项目需要安装的 Python 包及版本。  
  用途：在新环境中一键安装依赖，保证代码能跑起来。

---

### 11.8 其他零散概念

- **Token（词元）**  
  模型处理文本时的最小单位，可能是一个词、一个子词甚至一个符号。  
  在本项目里：token 数会影响模型的输入长度和计算时间。

- **Embedding（嵌入 / 向量表示）**  
  把词或句子变成向量（很多个数字组成的一串），以便用数学方式衡量相似度。  
  在本项目里：稠密检索和句级证据打分都可以用 embedding。

- **延迟（Latency）**  
  从发出请求到收到响应的时间，一般用毫秒 (ms) 表示。  
  在本项目里：延迟包括检索、重排、调用大模型等所有步骤的耗时。

## 12. 疑问
```
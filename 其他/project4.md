```markdown
# 项目 v4 说明文档（加强版创新版）

> 基于现有 `ingest / index_bm25 / index_dense / retriever / rag / eval / synth_qa` 的基础系统，在此版本中引入一组“**可落地的检索增强与控制策略**”，用于提升复杂查询场景下的效果与可解释性。

---

## 0. 项目总览（与 project3 的关系）

- **v3 版本**：完成一个标准的学术领域 RAG 系统：
  - arXiv 摘要数据；
  - BM25 + 稠密向量 + Hybrid 检索；
  - RAG 问答 + LLM 抽象；
  - 合成 QA + 自动评测。
- **v4 版本（本说明文档）**：在 v3 的基础上，增加一组**可选“进阶功能”**：
  - 动态混合权重（自适应 `alpha`）；
  - 跨编码器重排（Cross-Encoder Rerank）；
  - 查询扩展（PRF + LLM 问题改写）；
  - 证据句级高亮；
  - 类别感知检索；
  - SLA（延迟预算）驱动的策略选择。
- **定位**：  
  - 不破坏原有 v3 基础功能（全部可关闭）；  
  - 每个创新点都有清晰的模块边界、配置开关和指标目标，方便课堂展示与后续实现。

---

## 1. 课程任务与项目目标（与 v3 一致）

> 本节与 project3 保持一致，仅简要回顾，方便课堂 Presentation。

### 1.1 课程题目

- 课程要求：**“面向特定领域问答的检索增强生成（RAG）”**
- 要求包括：
  - 使用特定领域语料库（本项目使用 **arXiv 摘要**）；
  - 使用向量数据库（本项目使用 **Chroma**）构建稠密检索；
  - 与开源大语言模型集成（如 Qwen、LLaMA 或任意 OpenAI 兼容模型）；
  - 从准确率、召回率、延迟等维度，对比 **稠密检索 vs BM25**。

### 1.2 本项目 v4 的具体目标

- **核心目标**  
  在 v3 的 RAG 系统之上，引入“**动态 + 精排 + 扩展 + 高亮 + 类别 + SLA**”六类增强模块，在保持系统结构清晰的前提下：
  - 提升复杂查询下的召回和排序质量；
  - 改善答案的可解释性和用户体验；
  - 支持在**不同延迟预算**下自动选择策略。

- **范围：不变**
  - 只使用摘要 + 元数据（不解析 PDF 全文）；
  - 不微调 LLM，仅调用现成开源或云端模型；
  - UI 仍为命令行/脚本级别。

---

## 2. 数据与基础系统（与 v3 一致的部分）

### 2.1 数据来源与预处理

- 数据源：`./data/arxiv-metadata-oai-snapshot.json`（约 4.6GB，JSONL）
- 保留字段：
  - `id/title/abstract/categories/created`
- 预处理：`src/ingest.py` 流式读取 → 写出 `data/clean.jsonl`

### 2.2 基础模块与项目结构

> v4 在 v3 的结构上增量扩展。基础部分与 project3 保持一致：

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
│   ├── retriever.py                          # 统一检索接口 (bm25/dense/hybrid)
│   ├── rag.py                                # RAG 上下文构建 + LLM 调用
│   ├── eval.py                               # 离线评测
│   ├── synth_qa.py                           # 合成问答数据生成
│   ├── heuristics.py                         # [新] 查询/类别/策略 轻量启发式
│   ├── reranker.py                           # [新] Cross-Encoder 重排
│   ├── expansion.py                          # [新] 查询扩展 (PRF + LLM)
│   └── snippets.py                           # [新] 句级证据抽取与高亮
├── configs/
│   └── config.yaml                           # 模型/检索/策略 配置
├── logs/
│   ├── metrics_*.csv                         # 各模式评估结果
│   └── traces/                               # [可选] SLA & 延迟日志
└── requirements.txt
```

基础模块行为同 project3，不再详细展开，重点放在**新加入的创新模块**。

---

## 3. v4 版系统总流水线（含增强模块）

### 3.1 新旧对比：v3 → v4

**v3 流程（简版）**：

```text
raw jsonl → ingest → clean.jsonl
→ index_bm25 / index_dense
→ retriever(bm25/dense/hybrid, 固定 alpha)
→ rag(build_context + LLM)
→ answer + citations
→ eval(Recall/MRR/Latency)
```

**v4 流程（增强版）**：

```text
raw jsonl → ingest → clean.jsonl
→ index_bm25 / index_dense
→ synth_qa(合成 QA, 可选)
→ eval(基础指标, 可选)

在线查询阶段：

query
  ↓
heuristics.classify_query(q)     # [新] 查询风格/类别 → dynamic alpha / category
  ↓
expansion.expand_query(q)        # [新] PRF + LLM 查询扩展（可选）
  ↓
retriever.retrieve_enhanced(     # [改] 支持:
    base_queries=[q, q1, q2...],
    mode=bm25/dense/hybrid,
    dynamic_alpha=on/off,
    category_filter/boost,
    latency_budget_ms
) → 初排候选 Top-N
  ↓
reranker.rerank(query, docs)     # [新] Cross-Encoder 精排（取 Top-k）
  ↓
snippets.select_evidence(...)    # [新] 句级证据抽取
  ↓
rag.enhanced_answer(             # [改] 使用证据片段构上下文 + 返回 evidence
    query, docs, evidences
) → answer + citations + evidence
```

---

## 4. 创新模块 1：动态混合权重（自适应 alpha）

### 4.1 问题与直觉解释

- 原 v3 中的 Hybrid 模式使用**固定 `alpha`** 融合 BM25 与 Dense 分数；
- 现实中：
  - **术语型** 查询（大量关键词、符号、缩写）更适合偏 BM25；
  - **语义型** 查询（长句、自然语言）更适合偏 Dense；
- 固定 `alpha` 容易出现 **query-style mismatch**：  
  某类查询始终被“压制”，导致整体表现不稳定。

### 4.2 方法概述

- 为每个 query 提取一组**轻量级特征**：
  - 长度（token 数、字符数）；
  - 数字/符号占比；
  - 关键词密度；
  - BM25 vs Dense 初次检索的分数差（可选的二次特征）；
- 使用简单的 **规则 + 逻辑回归**（或纯规则）将查询划分为：
  - `term-heavy`（术语型）；
  - `semantic`（语义型）；
  - `mixed`（混合型）；
- 根据类型设定 `alpha`：
  - 术语型：`alpha_dense` 较小，如 0.2；
  - 语义型：`alpha_dense` 较大，如 0.8；
  - 混合型：`alpha_dense` 约 0.5。

### 4.3 模块与接口设计

- 新增文件：`src/heuristics.py`
  - 函数一：`classify_query(q: str) -> Dict`
    - 返回结构示例：
      ```python
      {
        "type": "term" | "semantic" | "mixed",
        "alpha": 0.2,
        "features": {...}
      }
      ```
  - 可附加：类别预测、延迟预算决策等（供后续模块共享）。

- 修改：`src/retriever.py`
  - 在原有 `retrieve` 基础上增加一个“增强版入口”：
    ```python
    def retrieve_enhanced(
        query: str,
        topk: int = 5,
        mode: str = "hybrid",
        alpha: float | None = None,
        config_path: str = "configs/config.yaml",
        use_dynamic_alpha: bool = True,
        **kwargs
    ) -> List[Doc]:
        """
        当 alpha=None 且 use_dynamic_alpha=True 时：
        - 调用 heuristics.classify_query 获取动态 alpha
        - 否则使用传入的 alpha 或配置文件中的默认值
        """
    ```
  - 原 `retrieve` 可保持不变，作为简化接口。

- 配置扩展：`configs/config.yaml`
  ```yaml
  retrieval:
    mode: hybrid
    topk: 5
    alpha: 0.5           # 默认 alpha
    dynamic_alpha: true  # 是否启用自适应 alpha
  ```

### 4.4 评估与指标

- 划分评测数据为：
  - 术语型子集 / 语义型子集 / 混合型子集；
- 对比：
  - 固定 alpha vs 动态 alpha；
- 指标：
  - 各子集的 Recall@k / MRR；
  - 延迟变化（理论上几乎不变，因为新增计算很轻量）。

---

## 5. 创新模块 2：跨编码器重排（Rerank）

### 5.1 问题与直觉

- 初排 Top-k 由 BM25/Dense/Hybrid 产生，速度快但可能排序不够精确；
- 对于 RAG，**前几篇文档的质量直接影响答案与引用质量**；
- 希望在 Top-N 候选中，再做一轮“精排”，类似请更强的模型逐条打分。

### 5.2 方法概述

- 使用 Cross-Encoder（如：
  - `bge-reranker-base`;
  - 或 `cross-encoder/ms-marco-MiniLM-L-6-v2`）；
- 对 `query` 与每个候选 `doc.text` 拼接成输入，让模型直接输出一个相关性分数；
- 对 Top-N 候选按这个分数重新排序，再取前 k 个作为最终结果。

### 5.3 模块与接口设计

- 新增文件：`src/reranker.py`
  - 推荐接口：
    ```python
    def load_reranker(model_name: str):
        """初始化 CrossEncoder 模型，可做缓存。"""

    def rerank(
        query: str,
        docs: List[Dict],
        model_name: str,
        topk: int,
        batch_size: int = 16,
    ) -> List[Dict]:
        """
        输入：query + 初排 docs 列表
        输出：按 Cross-Encoder 打分后的 Top-k docs
        """
    ```

- 修改：`src/retriever.py` 或 `rag.py`
  - 可选模式一：在 `retrieve_enhanced` 中集成：
    ```python
    def retrieve_enhanced(..., enable_rerank: bool = False, rerank_topn: int = 50, rerank_model: str = "..."):
        # 先初排 Top-N
        # 若 enable_rerank: 调用 rerank(query, topN_docs, rerank_model, topk)
        # 否则直接返回初排 Top-k
    ```
  - 可选模式二：由 `rag.enhanced_answer` 控制是否 rerank。

- 配置扩展：`configs/config.yaml`
  ```yaml
  rerank:
    enable: false
    model: bge-reranker-base
    topn: 50       # 初排取前 N 再重排
    batch_size: 16
  ```

### 5.4 评估与指标

- 对比：
  - 不 rerank vs rerank；
- 指标：
  - MRR / NDCG（更关注排序质量）；
  - 引用中“真正相关论文”的比例；
  - 延迟（随 topN 增大而增加，需要 trade-off）。

---

## 6. 创新模块 3：查询扩展（PRF + LLM 改写）

### 6.1 问题与直觉

- 用户问题可能：
  - 太短、信息不足；
  - 使用冷门表述或同义句式；
- 直接检索可能召回不足，Recall@k 上限偏低；
- 希望：**自动“帮用户想几种问法”**，扩大检索覆盖面。

### 6.2 方法概述

- 第一层：PRF（Pseudo Relevance Feedback）
  - 用原始 query 检索一次，取初排 Top-M 文档；
  - 从这些文档中统计高频、代表性的关键词（类似 RM3）；
  - 这些新词用来扩展原始 query。
- 第二层：LLM 改写
  - 调用 LLM，对 query 生成 2–3 个等价问法或更详尽的问法；
- 最终：形成一个 `query variants` 集合：`[q_original, q_prf, q_llm1, q_llm2, ...]`；
  - 对每个变体分别检索；
  - 合并结果，对同一 `id` 做分数归一化 + 融合 + 去重。

### 6.3 模块与接口设计

- 新增文件：`src/expansion.py`
  - PRF 接口：
    ```python
    def prf_terms(
        query: str,
        retriever_config: Dict,
        m_docs: int = 5,
        top_terms: int = 10,
    ) -> List[str]:
        """
        使用当前检索后端，对 query 做一次初排，
        从 top-m 文档中统计关键词，返回若干扩展词。
        """
    ```
  - LLM 改写接口（可复用 `rag.LLMClient`）：
    ```python
    def llm_expand(
        query: str,
        client: LLMClient,
        n_variants: int = 3,
    ) -> List[str]:
        """
        使用 LLM 生成 n 个语义等价或更明确的问法。
        """
    ```
  - 综合扩展：
    ```python
    def expand_query(
        query: str,
        client: Optional[LLMClient],
        use_prf: bool = True,
        use_llm: bool = True,
    ) -> List[str]:
        """
        综合 PRF 和 LLM 改写，返回若干 query 变体。
        """
    ```

- 修改：`src/retriever.py`
  - `retrieve_enhanced` 增加参数：
    ```python
    def retrieve_enhanced(..., enable_expansion: bool = False, expansion_cfg: Dict = None, ...):
        # 若 enable_expansion:
        #   queries = expansion.expand_query(...)
        #   对每个变体并行检索，合并去重
    ```

- 配置扩展：`configs/config.yaml`
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
  - Recall@10 / Recall@20 提升；
  - 新增命中的比例（通过变体检索才命中的样本数）；
  - 延迟增加（可通过并行请求和控制变体数量缓解）。

---

## 7. 创新模块 4：证据片段高亮（句级）

### 7.1 问题与直觉

- 原始 RAG 将完整摘要拼接成上下文：
  - 上下文可能很长；
  - 用户看到答案时，难以直观知道答案依据来自哪一句话；
- 希望：
  - 从摘要中抽取若干“**最可能回答问题的句子**”，构造更紧凑的上下文；
  - 在答案中**高亮这些证据句**，增强可解释性。

### 7.2 方法概述

- 对每篇文档的 `text`（title + abstract）进行**句子切分**；
- 针对每个句子：
  - 可用简单打分方式：
    - 基于 BM25 关键词重合度；
    - 或基于嵌入（query 与句子向量余弦相似度）；
- 对所有候选句子打分，选出前若干个作为 evidence；
- 用这些 evidence 拼接上下文用于 LLM 调用；
- 在返回结构中，附带 evidence 列表，方便前端高亮。

### 7.3 模块与接口设计

- 新增文件：`src/snippets.py`
  - 句子切分：
    ```python
    def sentence_split(text: str) -> List[str]:
        """按简单规则或 NLP 工具将文本切分为句子。"""
    ```
  - 句级打分：
    ```python
    def score_sentences(
        query: str,
        doc: Dict,
        max_sentences: int = 3,
        method: str = "bm25"  # 或 "embedding"
    ) -> List[Dict]:
        """
        返回若干句子及其分数：
        [{"sentence": "...", "score": 0.9}, ...]
        """
    ```
  - 汇总多文档 evidence：
    ```python
    def select_evidence_for_docs(
        query: str,
        docs: List[Dict],
        per_doc: int = 2,
        max_total: int = 10,
    ) -> List[Dict]:
        """
        对 docs 中每篇选若干句子，返回 evidence 列表：
        [{"id": doc_id, "title": ..., "sentence": ..., "score": ...}, ...]
        """
    ```

- 修改：`src/rag.py`
  - 增强版上下文构造：
    ```python
    def build_context_with_evidence(
        evidences: List[Dict],
        max_chars: int = 4000
    ) -> str:
        """
        使用句级 evidence 构造上下文，而不是完整摘要。
        """
    ```
  - 增强版 answer：
    ```python
    def enhanced_answer(
        query: str,
        mode: str = "hybrid",
        topk: int = 5,
        use_evidence: bool = True,
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

- 配置扩展：`configs/config.yaml`
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
  - 人工检查 evidence 是否合理；
  - 答案中引用部分是否更易理解、解释清晰。
- 定量：
  - 上下文长度减少（字符数 / token 数）；
  - 生成延迟是否下降。

---

## 8. 创新模块 5：类别感知检索

### 8.1 问题与直觉

- arXiv 涉及多个学科：
  - 如果问题是“图神经网络用于文本分类”，不希望命中一堆物理、生物文档；
- 希望利用 arXiv 的 `categories` 信息，加强**领域相关性**。

### 8.2 方法概述

- 通过 query 和候选文档的类别信息，做两类操作：
  - **过滤**：仅保留指定类别或类别前缀，比如只保留 `cs.*`；
  - **加权**：对类别匹配的文档给予分数加成。

### 8.3 模块与接口设计

- 复用：`src/heuristics.py`
  - 新增：
    ```python
    def predict_query_category(query: str) -> Optional[str]:
        """
        通过简单规则或小模型，预测 query 的主类别（如 "cs.CL"）。
        也可以返回 None 表示未知。
        """
    ```

- 修改：`src/retriever.py`
  - 在合并 BM25/Dense 结果后：
    - 若配置指定 allowed_categories，则过滤掉不在集合中的文档；
    - 若有 query_category，则对匹配类别的文档按配置增加权重；
  - 示例伪逻辑：
    ```python
    if category_cfg["filter"]:
        docs = [d for d in docs if primary_cat(d) in allowed]
    if category_cfg["boost"]:
        if doc_cat == query_cat:
            score *= (1 + boost_factor)
    ```

- 配置扩展：`configs/config.yaml`
  ```yaml
  category:
    enable_filter: false
    allowed_prefixes: ["cs."]
    enable_boost: true
    boost_factor: 0.1
  ```

### 8.4 评估与指标

- 指标：
  - 精确率（Precision@k）；
  - Off-topic 文档比例下降；
- 需要人工抽样评估若干 query 的命中文档是否属于正确领域。

---

## 9. 创新模块 6：SLA 驱动模式选择

### 9.1 问题与直觉

- 不同使用场景对“速度 vs 效果”的要求不同：
  - Demo / 课堂展示：可以稍慢一些，换更好的效果；
  - 在线调用：可能要求端到端小于 1 秒；
- 希望系统**自动在 BM25 / Dense / Hybrid / Rerank / QE 等组合之间做取舍**。

### 9.2 方法概述

- 在配置或调用参数中加入延迟预算：
  - `latency_budget_ms`，例如 500ms、1000ms；
- 在运行过程记录几个模块的平均耗时（可由 `eval.py` 或 logging 得到）；
- 根据预算选择策略组合，例如：
  - Budget 非常紧：只用 BM25、不开启 QE 和 Rerank；
  - Budget 中等：使用 Hybrid，不开 Rerank；
  - Budget 宽松：使用 Hybrid + Rerank + 段落级 evidence。

### 9.3 模块与接口设计

- 复用：`src/heuristics.py`
  - 新增一个简单策略函数：
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

- 修改：`src/retriever.py` / `src/rag.py`
  - 在入口处读取 `latency_budget_ms` 参数（可从 config 或调用参数中传入）；
  - 调用 `choose_strategy` 决定：
    - 是否启用 dynamic_alpha、expansion、rerank；
    - 选择何种检索模式；
  - 这些选择会影响后续模块行为。

- 配置扩展：`configs/config.yaml`
  ```yaml
  runtime:
    latency_budget_ms: 1000      # 默认端到端预算
  ```

### 9.4 评估与指标

- 使用 `eval.py` 增加对 end2end_ms 的统计；
- 对不同 `latency_budget_ms` 配置：
  - 验证是否满足预算；
  - 比较 Recall/MRR 的变化；
- 说明：这部分更偏工程实践和 trade-off 分析，是课堂展示的加分项。

---

## 10. 总结：v4 版本的教学价值

- **从“标准 RAG 系统”升级到“可调节、可解释的实验平台”**：
  - 动态 alpha：教会如何基于查询特征做轻量自适应；
  - Rerank：引入 Cross-Encoder 精排，兼顾效果与延迟；
  - QE：展示 PRF + LLM 结合的实用技巧；
  - Evidence：强化可解释性与用户体验；
  - Category & SLA：体现工程系统中“领域约束”和“服务级别协议”的重要性。
- 对课堂 10 分钟 Presentation：
  - 可以先用 project3 讲基础流程；
  - 再用 project4 列出上述 6 个创新点；
  - 每个创新点都可以给出“问题 → 方法 → 模块 → 指标”的闭环，非常利于展示深度。

> 后续如果你希望，我可以基于这个 project4 说明，按模块优先级（比如先实现动态 alpha 和 rerank），逐步帮你把 `heuristics.py / reranker.py / expansion.py / snippets.py` 的伪代码和真实实现写出来，并补一套针对这些创新点的测试与评估命令。
```

```markdown
## 11. 面向小白的专业术语解释（小白词典）

> 说明：  
> 下面这些术语，是整个 v4 项目文档里经常出现的关键词。  
> 假设你完全不懂计算机/机器学习，也可以把这一节当成“概念挂件”来查。

---

### 11.1 系统整体相关

- **RAG（Retrieval-Augmented Generation）检索增强生成**  
  - 可以理解成“先查资料，再让大模型写答案”。  
  - 第一步：像搜索引擎一样，从大量文献里找出几篇相关的；  
  - 第二步：把这些文献内容当成“资料”，交给大语言模型，让它在“看过资料”的前提下回答问题。  

- **语料 / 语料库（Corpus）**  
  - 就是“原始文本数据的集合”，在这个项目里就是一大堆 arXiv 论文摘要。  
  - 可以想象成：一个学校图书馆里所有书的“书目卡片”汇总到一个大文件里。

- **arXiv**  
  - 一个公开的科学论文预印本网站，很多计算机、数学、物理的论文都先发在 arXiv 上。  
  - 本项目就是用它的论文“元数据”（标题、作者、摘要、类别等）。

- **JSON / JSONL**  
  - JSON：一种结构化文本格式，用来表示“键值对”，比如：  
    `{"id": "123", "title": "A Paper", "abstract": "..."}`。  
  - JSONL：JSON Lines，每一行都是一个 JSON 对象，用来表示“很多条记录”，例如一行一篇论文。

---

### 11.2 检索（搜索）相关

- **BM25**  
  - 一种经典的“关键词匹配”算法，用了几十年，很多搜索引擎都用过。  
  - 直观理解：  
    - 如果论文里出现了很多你问的问题里的关键词，就得高分；  
    - 如果一个词在所有论文里都非常常见（比如 “the”），它的权重就会降低。  

- **稠密向量 / 向量表示（Embedding）**  
  - 把一句话或一篇摘要，转换成一个长长的数字列表（比如 768 维）——这叫“向量表示”。  
  - 直觉上可以理解为：模型把“这句话的含义”压缩成一串数字，之后就可以比较两个向量有多接近，从而衡量语义是否相似。  

- **向量数据库（Vector DB）**  
  - 专门用来存储和检索“向量”的数据库。  
  - 本项目使用的 **Chroma** 就是一个向量数据库：  
    - 先把每篇论文的摘要转成向量存进去；  
    - 后面给问题也算一个向量，然后找“向量空间里最接近”的那些论文。

- **Chroma**  
  - 一个开源的本地向量数据库，支持持久化到磁盘。  
  - 可以理解为“给你一块磁盘上的空间，专门存这些向量，方便按相似度检索”。

- **Hybrid 检索**  
  - 同时用 BM25（关键词）和 Dense（语义）两个通路做检索，然后把结果融合在一起。  
  - 好处：既保留关键词的精确匹配，又有语义相似的能力，通常比单一路径更稳。

- **α（alpha，混合权重）**  
  - 用来控制 BM25 分数和 Dense 分数在 Hybrid 中各占多少比重。  
  - 简单公式：`最终得分 = alpha * 稠密得分 + (1 - alpha) * BM25 得分`。  
  - 如果 alpha 大一些，就更“相信语义向量”；小一些就更“相信关键词匹配”。

- **Query / 文档 / Top-k**  
  - Query：用户提的问题，就是“一条查询”。  
  - 文档：这里就是一篇论文的“标题 + 摘要”那一条记录。  
  - Top-k：检索后只取分数最高的前 k 条结果，比如 Top-5 就是最相关的 5 篇。

---

### 11.3 模型结构与重排相关

- **Bi-Encoder（双编码器）**  
  - 把 Query 和 文档 各自单独编码成向量（用同一个或类似的模型），然后用向量相似度来计算相关性。  
  - 好处：可以提前算好所有文档的向量，检索时只要算 Query 的向量，速度非常快；  
  - 缺点：因为 Query 和 文档是分开看的，表达能力略有限。

- **Cross-Encoder（交叉编码器）**  
  - 直接把 “Query + 文档” 拼成一个输入序列，让模型“一起看”，然后输出一个相关性分数。  
  - 好处：效果通常比 Bi-Encoder 好，因为模型可以在一句话里同时理解两者的关系；  
  - 缺点：必须对每个候选文档都跑一遍模型，非常慢，所以只适合对少量候选做“精排”。

- **Rerank（重排）**  
  - 在已经有一批候选结果的前提下，再用更强的模型（比如 Cross-Encoder）重新给它们排序。  
  - 可以理解为：先让“普通搜索员”快速筛出前 50 个候选，再请“专家”深入阅读这 50 个，把最靠谱的排在最前面。

---

### 11.4 查询改写与反馈相关

- **PRF（Pseudo Relevance Feedback，伪相关反馈）**  
  - “伪”是因为没有人工标注，只是假设初次检索的 Top-M 文档大部分是相关的。  
  - 做法：  
    1. 用原始 query 检索一次，取前 M 篇文档；  
    2. 从这些文档中统计高频关键词；  
    3. 把这些词当成“额外补充”，用来扩展原始 query；  
  - 效果：有助于扩大召回，让系统找到更多相关文档。

- **Query Expansion（查询扩展）**  
  - 不只用用户最初输入的那一句，而是自动生成几种等价或更详细的问法，一起用来检索。  
  - 在本项目中，扩展来源包括 PRF 和 LLM 改写。  
  - 好处：弥补用户表达不够精确或太简略的问题；但缺点是检索次数会增多，延迟可能增加。

---

### 11.5 证据与解释相关

- **Evidence / 证据片段**  
  - 在本项目中，指的是“从命中文档中选出来的关键句子”，这些句子为大模型的答案提供直接依据。  
  - 举例：  
    - 答案说“对比学习通过拉近正样本、拉远负样本的距离来训练表示”，  
    - Evidence 就可能是一句论文摘要：“We propose a contrastive learning method that pulls positive pairs together and pushes negative pairs apart.”  
  - 在输出中，我们会把这些证据句和对应论文的 id/title 一起展示。

- **句级打分 / Snippets**  
  - 把文档切成一句一句，对每句打一个“跟问题有多相关”的分数；  
  - 选出分数最高的几句拼上下文，既提高解释性，也减少噪音。

---

### 11.6 类别与 SLA 相关

- **arXiv Category / 类别标签**  
  - arXiv 会给每篇论文打一个或多个学科标签，比如：  
    - `cs.CL`：计算机科学 - 计算语言学；  
    - `cs.LG`：计算机科学 - 机器学习；  
    - `astro-ph`：天体物理等。  
  - 我们可以利用这些标签：  
    - 只在某些类别里检索；  
    - 或者对匹配类别的文档加分，避免跨学科“跑偏”。

- **SLA（Service Level Agreement，服务级别协议）**  
  - 可以理解为：系统向用户承诺的一些“服务指标”，比如：  
    - 响应时间必须小于 1 秒；  
    - 可用性必须达到 99.9%。  
  - 在本项目中，我们用一个简单的 `latency_budget_ms` 来表示延迟预算，根据这个预算自动选择检索策略（是否启用 rerank、扩展等）。

---

### 11.7 评测指标相关

- **Recall@k（召回率@k）**  
  - 问：“在前 k 个检索结果里，有没有至少一个是我们认为‘正确答案’的论文？”  
  - 对于每个问题，如果 Top-k 里命中了 gold id，就记 1，否则记 0，再对所有问题求平均。  
  - 数值越接近 1，说明系统越不容易“漏掉”正确文档。

- **MRR（Mean Reciprocal Rank，平均倒数排名）**  
  - 问：“第一个正确文档排在第几名？”  
  - 如果排第 1 名，得分 1.0；第 2 名，得分 1/2=0.5；第 5 名得分 1/5=0.2；如果完全没命中，就是 0。  
  - 对所有问题的得分求平均，就得到 MRR。数值越高，说明“正确答案排得越靠前”。

- **NDCG（Normalized Discounted Cumulative Gain）**  
  - 更细致的排序指标，考虑到多个正确文档及其位置。  
  - 在本项目里，主要用 MRR 和 Recall，比 NDCG 更直观一些。可以理解为：NDCG 是“更复杂版的排序质量分数”。

- **Latency / 延迟（ms）**  
  - 指系统从接到请求到返回结果之间的时间，常用毫秒（ms）来衡量。  
  - 我们通常把它拆成：  
    - `search_ms`：检索花了多少时间；  
    - `gen_ms`：大模型生成答案花了多少时间；  
    - `end2end_ms`：从开始到拿到答案的总时间。

---

### 11.8 大模型与调用相关

- **LLM（Large Language Model，大语言模型）**  
  - 一种可以“看懂文本、写文本”的大型神经网络，比如 ChatGPT、Qwen、LLaMA 等。  
  - 本项目中的 LLM 用来：  
    - 根据检索到的摘要回答问题；  
    - 帮助生成合成问答数据；  
    - 做查询改写等。

- **Prompt / 提示词**  
  - 发送给大模型的“指令文字”，包括：  
    - system prompt：告诉模型“你是谁、该怎么回答”；  
    - user prompt：真正的问题和上下文内容。  
  - 好的 prompt 可以显著提升回答质量。

- **Mock 模式**  
  - 为了方便本地测试，我们提供了一个 `provider="mock"` 的模式。  
  - 在这个模式下，不会真正调用任何外部大模型，而是返回一个简单的、可控制的伪造答案，比如在原问题后面加上 `[Mock Answer]`。  
  - 好处：  
    - 测试代码逻辑不会消耗 API 额度；  
    - 即使没有网络或没有 API Key，也可以跑通整个 pipeline。

---

如果你在阅读 project4 / code4 / ppt4 / pre4 的过程中遇到任何不理解的术语，都可以先来这一节快速查一眼；  
也可以在课堂上把这部分做成**备用附录页**，在问答环节里用来解释给非技术背景的同学听。
```
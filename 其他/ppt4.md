```markdown
# 10 分钟课堂汇报方案（多策略增强 RAG 学术问答系统）

> 使用方式：  
> - 每页 PPT 有三块：  
>   - **展示内容**：这一页大概要放什么。  
>   - **内容来源**：从哪些文件/哪一节/哪一个 CSV 里拿文本或数据。  
>   - **讲稿要点**：你讲的时候可以围绕的 2–4 句主线。  
> - 你可以把本文件保存为 [ppt4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/ppt4.md:0:0-0:0)，做 PPT 时对照使用。

---

## 一、PPT 大纲（每页内容 + 内容来源）

### Slide 1：标题 & 项目概览

**展示内容**

- 标题：  
  “基于 arXiv 摘要的多策略增强 RAG 学术问答系统”
- 副标题：课程名 + 姓名/学号。
- 1 行简介：  
  “构建一个能理解论文摘要的问答系统，并通过多种检索与控制策略提升效果、速度和可解释性。”
- 一张简单示意图：  
  左边“用户问题”，中间“检索 + 多种策略模块”，右边“回答 + 引用论文”。

**内容来源**

- 文本：
  - `project4.md` 第 0 节「项目总览」；
  - `pre4.md` Slide 1 讲稿。
- 图示：
  - 自己在 PPT 中画简单的输入→系统→输出框图。

**讲稿要点**

- 一句话说明这是“基于 arXiv 摘要的学术问答系统”；  
- 强调系统包含：完整的 RAG 架构 + 6 个策略增强模块；  
- 说明汇报顺序：任务与数据 → 系统架构 → 关键方法 → 实验结果 → 总结。

---

### Slide 2：任务背景与项目目标

**展示内容**

- Bullet 列表：
  - 课程主题：面向特定领域问答的检索增强生成（RAG）；
  - 领域：学术论文，数据来源是 arXiv 摘要；
  - 任务：对比 BM25 与稠密检索，并与大语言模型结合完成问答；
  - 项目目标：
    - 从 4.6GB 原始数据构建完整 RAG 系统；
    - 在此基础上设计 6 个检索/控制增强策略；
    - 在准确率、召回率、延迟和可解释性上系统评估。

**内容来源**

- `project4.md`：
  - 第 1 节「课程任务与项目目标」；
- `pre4.md`：
  - Slide 2 讲稿。

**讲稿要点**

- 解释课程要求：要在特定领域语料上做 RAG，并对检索方式进行对比；  
- 说明为什么选 arXiv 摘要（公开、规模大、问题本身就偏学术）；  
- 点出本项目的两个核心目标：  
  1. 搭建一个完整、可运行的学术 RAG 系统；  
  2. 在此基础上实现 6 个策略增强，并用实验验证它们的价值。

---

### Slide 3：数据与系统整体架构

**展示内容**

- 一张整体架构图，建议分为“离线流程”和“在线流程”两部分。
- 离线流程（上半部分）：
  - `arxiv-metadata-oai-snapshot.json` → `ingest.py` → `clean.jsonl`  
  → `index_bm25.py` → `bm25.idx`  
  → `index_dense.py` → `chroma/`  
  → `synth_qa.py` → `synth_qa.jsonl`  
  → `eval.py` → `metrics_*.csv`
- 在线流程（下半部分）：
  - Query → `heuristics.py` → `expansion.py` → `retriever.retrieve_enhanced` →  
    `reranker.py` → `snippets.py` → `rag.enhanced_answer` → Answer + Citations + Evidence。

**内容来源**

- `project4.md`：
  - 第 2 节「数据与基础系统」；
  - 第 3 节「系统总流水线（含增强模块）」；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 0 节“总体实现顺序”；
  - 第 2 节配置结构。

**讲稿要点**

- 从左到右讲一遍数据流：  
  原始 JSONL → 预处理成精简 `clean.jsonl` → 构建 BM25 和稠密索引 → 合成 QA → eval 评测；  
- 再从上到下讲在线流程：  
  用户的问题进入系统后，依次经过启发式分析、查询扩展、初排、重排、证据抽取，最后进入 RAG 生成答案；  
- 强调用一张图把“基础 RAG + 强化策略模块 + 评测闭环”全部串起来。

---

### Slide 4：关键方法 1 —— 动态混合权重（自适应 α）

**展示内容**

- 左半部分（文字）：
  - 问题：  
    - Hybrid 检索需要一个 α 加权 BM25 和 Dense；  
    - 固定 α 无法兼顾“术语型问题”和“语义型问题”；  
  - 方法：  
    - `heuristics.classify_query(q)`，提取长度、数字/符号比例等特征；  
    - 将 query 粗分为 `term / semantic / mixed` 三类；  
    - 按类型给出建议 α：例如 0.2 / 0.8 / 0.5。
  - 接口：
    - `retrieve_enhanced(..., use_dynamic_alpha=True)` 自动接入自适应 α。
- 右半部分（示意图）：
  - 一个小流程：`query → classify_query → alpha → hybrid scoring`。

**内容来源**

- `project4.md`：
  - 第 4 节「动态混合权重（自适应 alpha）」；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 6 节 `heuristics.py`；
  - 第 7 节 `retrieve_enhanced` 中 dynamic_alpha 的逻辑。

**讲稿要点**

- 用例子解释术语型 vs 语义型查询：前者关键字明确，后者更像自然语言描述；  
- 说明 `classify_query` 是一个“轻量级决策器”，只看一些简单特征就粗分三类；  
- 强调动态 α 带来的好处：在几乎不增加计算量的情况下，让 Hybrid 对不同类型问题自动调整 BM25/向量的权重。

---

### Slide 5：关键方法 2 —— 跨编码器重排（Rerank）

**展示内容**

- 左半部分（文字）：
  - 问题：  
    - 初排 Top-k 的质量直接影响 RAG；  
    - Bi-Encoder 模型适合“广撒网”，但前几名排序未必最优；  
  - 方法：  
    - 使用 Cross-Encoder（如 `bge-reranker-base`）对初排 Top-N 做精排；  
    - 接口：`reranker.rerank(query, docs, model_name, topk)`；
    - 在 `retrieve_enhanced` 中，`enable_rerank=True` 时自动调用。
- 右半部分（小示意表）：
  - 上表：初排前几名列表；  
  - 下表：重排后的前几名列表，相关文档移到更靠前。

**内容来源**

- `project4.md`：
  - 第 5 节「跨编码器重排（Rerank）」；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 8 节 `reranker.py`；
  - 第 7 节 `retrieve_enhanced` 中 Rerank 集成说明。

**讲稿要点**

- 解释 Bi-Encoder vs Cross-Encoder 的差别：前者速度快，后者更“仔细读”；  
- 指出 Cross-Encoder 只用于 Top-N 候选的二次打分，因此延迟可控；  
- 为后面实验页埋伏笔：可以看到 Rerank 会显著提升 MRR 和引用质量，但会增加一定检索时间。

---

### Slide 6：关键方法 3 —— 查询扩展（PRF + LLM 改写）

**展示内容**

- 左半部分（文字）：
  - 问题：  
    - 用户提问简短/模糊时，直接检索容易召回不足；  
  - 方法：
    - PRF：用原始 query 检索一次，从 Top-M 文档中抽取高频关键词，扩展 query；  
    - LLM 改写：利用 `LLMClient` 生成若干语义等价或更具体的问法；  
    - 整合：`expand_query` 输出多个 query 变体。
- 右半部分（示意图）：
  - `原始 query` → `PRF terms` + `LLM variants` → 多个 query → 并行检索 → 结果合并去重。

**内容来源**

- `project4.md`：
  - 第 6 节「查询扩展（PRF + LLM 改写）」；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 9 节 `expansion.py`；
  - 第 7 节 `retrieve_enhanced` 中 enable_expansion 的逻辑。

**讲稿要点**

- 用自然语言讲 PRF：先用原问题找一波文档，再从命中的文档里“学几个高频词”回来扩展问题；  
- 解释 LLM 改写的作用：帮用户“多想几种问法”；  
- 强调对 Recall@10/20 的提升，同时也指出会增加检索耗时，这部分可通过 SLA 策略控制是否启用。

---

### Slide 7：关键方法 4 —— 证据片段高亮（句级 Evidence）

**展示内容**

- 左半部分（文字）：
  - 问题：
    - 原始摘要长且冗余，难以明确答案依据；  
  - 方法：
    - `snippets.sentence_split` 按句切分 title+abstract；  
    - `score_sentences` 用关键词重合度或嵌入相似度为句子打分；  
    - `select_evidence_for_docs` 从每篇文档选若干 Top 句作为 evidence；  
    - RAG 中 `build_context_with_evidence` 用这些句子构造上下文。
- 右半部分（示例输出）：
  - 截一段 `enhanced_answer` 的输出：
    - `answer: ...`  
    - `evidence:`  
      - `[id] title :: sentence...` 等几行。

**内容来源**

- `project4.md`：
  - 第 7 节「证据片段高亮（句级）」；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 10 节 `snippets.py`；
  - 第 11 节 `build_context_with_evidence` 和 `enhanced_answer`；
- 示例输出：
  - 实际运行 `enhanced_answer` 命令行的结果。

**讲稿要点**

- 说明 evidence 的直观意义：答案不仅给结论，还标出“来自哪篇论文的哪句摘要”；  
- 展示一个真实例子，让听众看到 evidence 列表增强了可信度和可解释性；  
- 补充一点：句级上下文更短，也有助于减少生成噪音和潜在延迟。

---

### Slide 8：关键方法 5 & 6 —— 类别感知检索 + SLA 策略选择

**展示内容**

- 左半部分（类别感知）：
  - 问题：跨学科语料容易引入 off-topic 命中；  
  - 方法：
    - 利用 arXiv `categories` 字段；
    - 通过 `predict_query_category` 粗略预测 query 所属学科前缀（如 `cs.*`）；  
    - `_apply_category_logic`：
      - 可过滤不在允许前缀集合中的文档；  
      - 对匹配学科的文档做分数加权。  
- 右半部分（SLA 策略）：
  - 延迟预算 `latency_budget_ms` 的定义；  
  - `choose_strategy` 根据预算返回策略组合：  
    - 紧：只用 BM25；  
    - 中：Hybrid + 部分扩展；  
    - 宽：Hybrid + Rerank + 扩展 + Evidence；  
  - 可以画一个小表格展示不同预算对应的策略。

**内容来源**

- `project4.md`：
  - 第 8 节「类别感知检索」；  
  - 第 9 节「SLA 驱动模式选择」；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 6 节 `predict_query_category` 与 `choose_strategy`；  
  - 第 7 节 `_apply_category_logic` 和 `retrieve_enhanced`。

**讲稿要点**

- 用“问 GNN 时不想看到 Astro”这类例子说明类别感知的必要性；  
- 用“有的场景要 300ms 内返回，有的可以等 1~2 秒”说明 SLA 的需求；  
- 强调通过配置文件和 `heuristics` 模块，可以把“延迟预算”变成明确的策略选择，而不是每次手工调参。

---

### Slide 9：实验与消融结果（关键表格）

**展示内容**

- 一张综合对比表（等你跑出结果后填数值），示意如下：

  | setting              | recall@5 | mrr   | search_ms | end2end_ms | note                 |
  |----------------------|----------|-------|-----------|-----------:|----------------------|
  | baseline hybrid      | 0.xx     | 0.xx  | xx.x      | xx.x       | 无增强模块           |
  | + dynamic α          | 0.xx     | 0.xx  | ~         | ~          | 自适应 Hybrid        |
  | + rerank             | 0.xx     | 0.xx↑ | ↑         | ↑          | 排序质量提升         |
  | + QE(PRF)            | 0.xx↑    | 0.xx  | ↑         | ↑          | Recall@10 提升明显   |
  | + evidence snippets  | ~        | ~     | ~         | ↓/≈        | 可解释性大幅提高     |
  | + category + SLA     | 0.xx     | 0.xx  | ~         | 满足预算   | off-topic 减少       |

- 可选再加一张简单柱状图（例如 Recall@5 vs setting）。

**内容来源**

- 结果 CSV（运行 eval.py 后）：
  - `logs/metrics_baseline.csv`；
  - `logs/metrics_dynamic_alpha.csv`；
  - `logs/metrics_rerank.csv`；
  - `logs/metrics_expansion_prf.csv`；
  - `logs/metrics_evidence.csv`；
  - `logs/metrics_sla_*.csv`；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 14.3 节“增强模块与消融实验（示例命令）”。

**讲稿要点（实验完成后再填具体数字）**

- 按“基线 → 加动态 α → 加 Rerank → 加 QE → 加 Evidence → 加类别/SLA”的顺序讲；  
- 重点说明每次打开某个模块，Recall/MRR/延迟的大致变化趋势：  
  - 动态 α：小幅提升，几乎不增加延迟；  
  - Rerank：MRR 明显提升，检索时间略有上升；  
  - QE：Recall@10 提升大，检索时间增加较多；  
  - Evidence：指标变化不大，但解释性提升显著；  
  - SLA：在不同预算下自动选择不同策略，平衡效果和延迟。  
- 强调这些模块都是“可开启/可关闭”的，因此非常适合做系统性的消融实验。

---

### Slide 10：工程实践 & 总结

**展示内容**

- 工程亮点列表：
  - 从 4.6GB 原始数据到可运行 RAG 系统的完整工程链路；
  - config 驱动的模块化设计（所有策略可通过配置开关控制）；
  - `code4.md + test4` 提供新手可复现的实现和测试指南；
- 一句总结 + 三个展望。

**内容来源**

- `project4.md`：
  - 第 10 节「总结」；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 实现顺序与测试顺序部分；
- `pre4.md`：
  - Slide 10 讲稿。

**讲稿要点**

- 工程角度：  
  - 从数据预处理、双索引构建到 RAG 和 eval，形成完整可运行系统；  
  - 多策略模块全部配置化，易于扩展和复用；  
- 方法角度：  
  - 三个关键词总结：自适应检索、排序与解释增强、工程可控性；  
- 展望：  
  - 更强的 embedding 和重排模型；  
  - 更严谨的人工评测数据集；  
  - 简单 Web Demo，把策略开关做成交互控件，展示 RAG 行为变化。

---

## 二、每页简短讲稿思路（精简版）

如果你希望在 PPT 的“备注”区写一句提示，可以参考：

- **Slide 1**：一句话讲清系统做什么；一句话点出“RAG + 6 个增强模块”；一句话说明后面的结构。  
- **Slide 2**：一句话说课程要求；一句话说为什么选 arXiv；一句话强调“完整 RAG + 策略增强”这两个目标。  
- **Slide 3**：用图把离线和在线流程串起来，让老师一眼看到你做的是“全链路系统”。  
- **Slide 4**：讲清楚术语型 vs 语义型查询的例子，然后说明动态 α 的直觉。  
- **Slide 5**：用“普通搜索员 + 专家重排”的比喻解释 Rerank 的作用和代价。  
- **Slide 6**：用“帮你多想几个问法”的比喻解释 PRF + LLM 改写，强调对 Recall 的贡献。  
- **Slide 7**：展示一条带 evidence 的回答，让老师看到系统“既会答，也会举证”。  
- **Slide 8**：用跨学科例子讲类别感知，用不同延迟预算例子讲 SLA。  
- **Slide 9**：重点讲趋势：谁提升了效果，谁增加了时间，谁主要提升解释性。  
- **Slide 10**：收尾时强调“完整工程 + 多策略 + 可实验”三个关键词，再给出 2–3 个未来方向。

这样，这份 `ppt4` 文档就是一个 **不依赖任何“基于 X 版本”说法的独立 PPT 设计说明**，你可以完全按它来搭建最终的 presentation。
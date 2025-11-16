
我来给你写一个专门针对 v4（project4 + code4）的 10 分钟 PPT 思路文稿，结构类似 `ppt3.md`，但重点讲 6 个创新点。每一页我都会标清楚：内容从哪一个文档或哪个文件/CSV 里拿（比如 project4 哪一节、code4 哪个段落、logs 哪个文件），方便你对应复制和截图做 PPT。
# 10 分钟课堂汇报方案 v4（基于 project4 + code4）

> 使用方式：  
> - 每页 PPT 有三块：  
>   - **展示内容**：这一页大概要放什么。  
>   - **内容来源**：从哪些文件/哪一节/哪一个 CSV 里拿文本或数据。  
>   - **讲稿要点**：你讲的时候可以围绕的 2–4 句主线。  
> - 你可以把本文件保存为 `ppt4.md`，做 PPT 时对照使用。

---

## 一、PPT 大纲（每页内容 + 内容来源）

### Slide 1：标题 & v3 → v4 概览

**展示内容**

- 标题：  
  “基于 arXiv 摘要的学术 RAG 系统 v4：从标准 RAG 到可调节实验平台”
- 副标题：课程名 + 姓名/学号。
- 1 行简介：  
  “在 v3 RAG 的基础上，增加动态权重、重排、查询扩展、证据高亮、类别感知和 SLA 等六个创新模块。”
- 一张简单示意图：左边“v3 标准 RAG”，右边“v4 增强模块”云状标注。

**内容来源**

- 文本：
  - [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0) 第 0 节「项目总览（与 project3 的关系）」。
- 图示：
  - 自己在 PPT 里画两个框：  
    - 左框：写 “v3: BM25 + Dense + Hybrid + RAG + Eval”；  
    - 右框：写 “v4: Dynamic α / Rerank / QE / Evidence / Category / SLA”。

**讲稿要点**

- 先一句话介绍 v3：已经有 BM25、Dense、Hybrid、RAG、合成 QA 和评测。  
- 再说 v4 的目标：在此基础上增加 6 个“策略层”的增强点，让系统更聪明、更可控。  
- 最后说明接下来会先回顾基础，再重点讲这 6 个创新点和实验结果。

---

### Slide 2：基础系统回顾（v3 RAG）

**展示内容**

- 简短列表：
  - 数据：arXiv 摘要 JSONL；
  - 双索引：BM25 + Dense（Chroma）；
  - 统一检索接口：`retrieve(bm25/dense/hybrid)`；
  - RAG：`answer` 基于上下文 + LLMClient；
  - 合成 QA + `eval.py` 指标。
- 不需要复杂图，简单 bullet 即可。

**内容来源**

- [project3.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project3.md:0:0-0:0)：
  - 第 1 节「项目目标与任务要求」；
  - 第 2.1 节「功能总览」；
- 代码说明：
  - [code3.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code3.md:0:0-0:0) 顶部“0. 总体实现顺序”。

**讲稿要点**

- 很快回顾一下 v3 做了什么：一个规范的学术 RAG pipeline。  
- 强调：整个 v4 的增强都是在这条成熟 pipeline 上做“外挂扩展”，**不破坏原有功能**。

---

### Slide 3：v4 总体架构 & v3 对比

**展示内容**

- 一张流程图：  
  - 上面画出 v3 流程：  
    `raw → ingest → index_bm25/index_dense → retrieve → rag → eval`  
  - 下面画出 v4 流程：  
    `query → heuristics → expansion → retrieve_enhanced → rerank → snippets → enhanced_answer → eval`
- 在图边列出 6 个模块名称。

**内容来源**

- [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0)：
  - 第 3 节「v4 版系统总流水线（含增强模块）」；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 0 节“实现顺序建议”；
  - 第 14.3 节中提到的 ablation 流程可以作为“实验入口”标注在 eval 位置。

**讲稿要点**

- 指着图解释：上半部分是 v3，主要是一次检索 + 一次 LLM；  
- 下半部分是在“query → retrieve → rag”之间插入了 新的策略层模块；  
- 强调：所有新功能都是可配置、可开关的，便于做消融实验。

---

### Slide 4：创新点 1 —— 动态混合权重（自适应 α）

**展示内容**

- 左半：
  - 问题：固定 α 无法兼顾术语型/语义型查询；  
  - 方法：
    - `heuristics.classify_query(q)` 提取特征，判断 `term / semantic / mixed`；
    - 根据类型给出 α 建议值（0.2 / 0.8 / 0.5）。
  - 接口：
    - `retrieve_enhanced(..., use_dynamic_alpha=True)`。
- 右半：
  - 一小段伪代码或流程框：  
    `query → classify_query → alpha → hybrid 融合打分`。

**内容来源**

- [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0)：
  - 第 4 节「动态混合权重（自适应 alpha）」。
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 3 节 `heuristics.py` 伪代码；
  - 第 7 节 `retrieve_enhanced` 注释中关于 dynamic_alpha 部分。

**讲稿要点**

- 先解释为什么会有“术语型 vs 语义型”查询；  
- 再解释 `classify_query` 用很轻的规则把 query 分三类；  
- 最后强调：在不增加明显延迟的前提下，让 Hybrid 的权重更“因题制宜”。

---

### Slide 5：创新点 2 —— 跨编码器重排（Rerank）

**展示内容**

- 左半：
  - 问题：初排 Top-k 相关性不够稳，影响最终引用质量；  
  - 方法：
    - 使用 Cross-Encoder（`bge-reranker-base` 等）对 Top-N 做精排；
    - 接口：`reranker.rerank(query, docs, model_name, topk)`。
- 右半：
  - 一个小示例：  
    - 上面：初排结果列表（1,2,3,4…）  
    - 下面：rerank 后列表（顺序变化，真正相关的排在前面）。

**内容来源**

- [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0)：
  - 第 5 节「跨编码器重排（Rerank）」。
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 5 节 `reranker.py` 伪代码；
  - 第 7 节中 `retrieve_enhanced(... enable_rerank=True)` 的说明。

**讲稿要点**

- 说明 Cross-Encoder 和 Bi-Encoder 的区别：前者慢但更精确；  
- 在 pipeline 中，它只对“少量 Top-N 候选”重排，所以延迟可控；  
- 实验上重点看 **MRR / NDCG** 是否提升。

---

### Slide 6：创新点 3 —— 查询扩展（PRF + LLM 改写）

**展示内容**

- 左半：
  - 问题：用户问题太短/表述奇怪 → 召回不足；  
  - 方法：
    - PRF 从 Top-M 文档里抽关键词扩展 query；
    - LLM 改写，生成多个语义等价问法；
    - 接口：`expansion.expand_query(...)` → 变体列表。
- 右半：
  - 示意图：  
    `原始 query` → `PRF terms` / `LLM variants` → 多条 query → 检索 → 合并去重。

**内容来源**

- [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0)：
  - 第 6 节「查询扩展（PRF + LLM 改写）」。
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 6 节 `expansion.py` 伪代码；
  - 第 14.3 节中关于 expansion 的 ablation 命令。

**讲稿要点**

- 用自然语言解释 PRF + LLM 改写：  
  先“从命中的文档里学词”，再“让 LLM 帮你换几种说法”；  
- 强调实验里主要看 Recall@10 / Recall@20 的提升，同时注意延迟变化。

---

### Slide 7：创新点 4 —— 证据片段高亮（句级 Evidence）

**展示内容**

- 左半：
  - 问题：上下文长、答案可解释性弱，用户不知道“依据在哪一句”；  
  - 方法：
    - `snippets.sentence_split` 分句；
    - `score_sentences` 对句子打分；
    - `select_evidence_for_docs` 选 Top 句；
    - RAG 中 `build_context_with_evidence` 构造上下文。
- 右半：
  - 展示一段示例输出（从命令行复制）：  
    - `answer: ...`  
    - `Evidence:` 列出几条 `[id] title :: sentence`。

**内容来源**

- [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0)：
  - 第 7 节「证据片段高亮（句级）」。
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 第 7 节 `snippets.py` 伪代码；
  - 第 11 节 `build_context_with_evidence` + `enhanced_answer`；
  - 你实际跑 `enhanced_answer` 的命令输出，可以截终端或整理几条 evidence 放在 PPT 中。

**讲稿要点**

- 强调：RAG 不只是“给出答案”，还要“给出证据”；  
- 展示 evidence 列表，让老师看见系统确实能指出“哪篇论文哪一句话”在支持答案；  
- 可以顺带提一下：句级上下文也有助于减少噪音、缩短生成时间。

---

### Slide 8：创新点 5 & 6 —— 类别感知 + SLA 策略

**展示内容**

- 左半：类别感知检索
  - 利用 arXiv `categories`，
  - 过滤非目标学科文档，或对同类文档加权；
  - 接口：`predict_query_category` + `_apply_category_logic`。
- 右半：SLA 驱动
  - 延迟预算 `latency_budget_ms`；
  - `heuristics.choose_strategy` 决定是否启用 rerank/expansion，以及用 bm25 还是 hybrid；
  - 用一个三档示意表：300ms / 800ms / 1500ms 下所选策略不同。

**内容来源**

- [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0)：
  - 第 8 节「类别感知检索」；
  - 第 9 节「SLA 驱动模式选择」。
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - `heuristics.predict_query_category` 与 `choose_strategy`；  
  - `retriever._apply_category_logic` 与 `retrieve_enhanced` 说明。
- 实验结果：
  - 类别：你可以用 `retrieve_enhanced` 的 demo 输出，展示命中的类别是否更集中于 `cs.*`；  
  - SLA：对比 `logs/ablation_sla_300ms.csv / 800ms / 1500ms` 中的 `end2end_ms` 与 `recall/mrr`。

**讲稿要点**

- 类别感知：避免“问 GNN 却来一堆 astrophysics”；  
- SLA：展示一个真实系统中“效果 vs 延迟”的工程权衡；  
- 说明你用 config.yaml + heuristics 让这些策略可配置、可调。

---

### Slide 9：实验与消融结果（关键表格）

**展示内容**

- 一张综合小表，列出几组典型 ablation（可以只选 3–4 组）：

  | setting                | recall@5 | mrr   | search_ms | end2end_ms | note                    |
  |------------------------|----------|-------|-----------|-----------:|-------------------------|
  | baseline hybrid        | 0.xx     | 0.xx  | xx.x      | xx.x       | v3 pipeline             |
  | + dynamic α            | 0.xx     | 0.xx  | ~         | ~          |                         |
  | + rerank               | 0.xx     | 0.xx↑ | ↑         | ↑          |                         |
  | + QE(PRF)              | 0.xx↑    | 0.xx  | ↑         | ↑          | Recall@10 提升明显      |
  | + evidence snippets    | ~        | ~     | ~         | ↓/≈        | 可解释性提升            |

- 如果时间允许，再加一张简单柱状图（Recall@5 vs setting）。

**内容来源**

- 结果 CSV：
  - `logs/metrics_v3_baseline.csv`  
  - `logs/ablation_dynamic_alpha_on.csv`  
  - `logs/ablation_rerank_on.csv`  
  - `logs/ablation_expansion_prf.csv`  
  - `logs/ablation_evidence_on.csv`  
  - `logs/ablation_sla_*`（如需）。
- 命令参考：
  - [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0) 第 14.3 节所有 ablation 命令。

**讲稿要点**

- 不需要讲具体数字，只讲趋势：  
  - 动态 α 让各类 query 更均衡；  
  - rerank 明显提升排序质量但增加一点延迟；  
  - QE 提升 Recall 尤其是@10，但带来更多检索开销；  
  - evidence 主要改善可解释性，对指标影响不大；  
  - SLA 允许在效果和延迟之间做自动权衡。

---

### Slide 10：工程实践 & 总结

**展示内容**

- 工程亮点列表：
  - 从 4.6GB 原始数据到可扩展 pipeline；
  - config 驱动的模块化设计（可插拔的新策略）；
  - 从 [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0) 和 `test4` 流程保证“新手可复现 + 易调参 + 易做实验”；
- 一句总结 & 三个展望。

**内容来源**

- [project4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/project4.md:0:0-0:0)：
  - 第 10 节「总结：v4 版本的教学价值」；
- [code4.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code4.md:0:0-0:0)：
  - 整体结构（实现顺序 + 测试顺序）。  

**讲稿要点**

- 总结三点收获：方法创新、工程落地、实验可控；  
- 展望：  
  - 可以换更强 embedding / reranker；  
  - 做更严谨的人工标注 QA 集；  
  - 做一个简单 Web Demo 展示这些策略的可视化效果。

---

## 二、每页简短讲稿思路（精简版）

如果你希望在演讲者模式下有一句话提示，可以直接把下面这些写到 PPT 的“备注”里（每页 2–3 句）：

- **Slide 1**：一句话说 v3 基础是什么；一句话说 v4 增加了哪 6 点；一句话说“接下来我按：基础 → 架构 → 六个创新 → 实验，总结”来讲。  
- **Slide 2**：把 v3 当成“干净的 baseline”：数据、BM25/Dense/Hybrid、RAG、合成 QA、eval。强调 v4 不改它，只是往上加模块。  
- **Slide 3**：用图对比 v3/v4 流程，让老师一眼看到“新模块都插在 query → retrieve → rag 这条线”。  
- **Slide 4**：用术语型 vs 语义型的例子解释动态 α 的直觉，让大家觉得“这个想法很自然”。  
- **Slide 5**：用“请专家重排”类比 Cross-Encoder，解释为什么只对 Top-N 做 rerank。  
- **Slide 6**：用“帮你多想几个问法”的比喻解释 PRF + LLM 改写，强调对 Recall 的提升。  
- **Slide 7**：展示一条带 evidence 的回答，让老师看到“这个系统不仅会答，还会举证”。  
- **Slide 8**：用“不要跨学科乱飘”和“不同场景延迟预算不同”的例子说明 Category 和 SLA 的价值。  
- **Slide 9**：只讲趋势，不纠结具体数值：哪几个开关明显带来效果提升，代价是什么。  
- **Slide 10**：收一个“完整闭环”——从原始数据到可扩展 RAG 平台，从代码骨架到实验可复现。

这样你就有了一份专门针对 v4 的 PPT 思路文稿：  
- 每页要放什么内容；  
- 对应内容从哪来；  
- 讲的时候大概要说什么。
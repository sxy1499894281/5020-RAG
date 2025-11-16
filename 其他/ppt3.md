我会先在 todo 里记录这次的任务：设计 10 分钟 PPT 结构、逐页讲稿，以及最后的图表/数据可实现性检查。然后直接给你一个完整的 markdown 文档，包含：每一页 PPT 要点、建议图表以及对应的演讲稿，最后单独一节分析每个图表/数据是否能用当前项目和代码直接产出或简单扩展产出。



```markdown
# 10 分钟课堂汇报方案（基于 project3 + code3）

> 使用方式：  
> - 第一部分：每一页 PPT 要放什么内容、什么图表。  
> - 第二部分：每一页的讲稿示例（你可以照着修改、精简）。  
> - 第三部分：检查每一类图表/数据是否能由当前项目代码直接产出。

---

## 一、PPT 大纲（每页内容与图表）

### Slide 1：标题 & 概览

**页面内容**

- 标题：  
  「基于 arXiv 摘要的学术 RAG 问答系统：双通路检索与合成评测」
- 副标题：课程名称 + 你的姓名/学号。
- 一行项目一句话描述：  
  “用 arXiv 摘要 + BM25 + 稠密向量 + 开源 LLM，做‘懂论文的问答助手’。”
- 小图：一个简单图标式示意（例如：问号 → 数据库 → 书本/机器人）。

**图表/可视化**

- 仅需一张简单的示意图片，可用 PPT 自己画（箭头 + 图标）。  
  无需代码生成。

---

### Slide 2：任务背景 & 课程要求

**页面内容**

- 标题：任务背景
- 列出课程题目关键点（从 project3 第 1 节抽取）：
  - 面向特定领域问答的 RAG；
  - 使用开源向量库（本项目用 Chroma）；
  - 对比 BM25 vs Dense 检索；
  - 不训练大模型，只使用开源/云端 LLM。
- 用 1–2 句话解释：  
  “为什么选择 arXiv 摘要做学术问答？”

**图表/可视化**

- 简单 bullet list；可在侧边放一张 arXiv 网站截图（可选）。  
  不需要代码产出。

---

### Slide 3：数据集与预处理（ingest.py）

**页面内容**

- 标题：数据集与预处理
- 简要说明：
  - 数据源：4.6GB `arxiv-metadata-oai-snapshot.json`（JSONL）；
  - 只保留：`id/title/abstract/categories/created`；
  - 输出：`clean.jsonl`；
- 流式读取的原因：文件大、节省内存。
- 展示一行 `clean.jsonl` 的样例（从 mini 数据生成的小样本即可）。

**图表/可视化**

- 左边：数据流向小图（raw jsonl → ingest → clean.jsonl）。  
- 右边：`clean.jsonl` 中一行 JSON 的截图/代码块。

---

### Slide 4：系统整体架构 / Pipeline

**页面内容**

- 标题：系统架构 / Pipeline
- 用一张总流程图展示（对应 project3 中的 pipeline）：

  `raw jsonl → ingest → clean.jsonl  
   → index_bm25 / index_dense → retriever (bm25/dense/hybrid)  
   → rag (LLM) → answer + citations → eval (metrics)`

- 标出每个模块对应的文件名（`ingest.py / index_bm25.py / index_dense.py / retriever.py / rag.py / eval.py / synth_qa.py`）。

**图表/可视化**

- 一张流程框图（用 PPT 绘制矩形 + 箭头）。  
  不需要代码生成。

---

### Slide 5：关键方法 1 —— 双通路检索 & Hybrid（retriever.py）

**页面内容**

- 标题：双通路检索与 Hybrid 融合
- 分三列简要说明：
  - BM25：关键词匹配，快、经典；
  - Dense：向量语义匹配，懂同义词；
  - Hybrid：  
    - 两路分别检索；  
    - 分数 min-max 归一化；  
    - `final_score = alpha * dense + (1-alpha) * bm25`。
- 强调：可以通过配置 `alpha` 调整偏向。

**图表/可视化**

- 一个小示意图：  
  左侧 BM25 Top-k 列表，右侧 Dense Top-k 列表，中间一个“融合”节点，输出最终 Top-k。  
  用 PPT 绘制即可。

---

### Slide 6：关键方法 2 —— RAG & LLM 抽象（rag.py）

**页面内容**

- 标题：RAG 流程与 LLM 抽象
- 列出 RAG 步骤：
  1. `retrieve(query, mode, topk)` 取文档；
  2. `build_context(docs)` 拼上下文（标题+摘要+分隔符）；
  3. `LLMClient.generate(system_prompt, user_prompt)` 调大模型；
  4. 输出 `answer + citations(id/title)`。
- 说明 LLMClient 的好处：
  - `provider = mock/openai/ollama`；
  - 同一接口可用于“回答问题”和“生成问题（合成 QA）”。

**图表/可视化**

- 一张 RAG 数据流小图：  
  Query → Retriever → Docs → Context → LLM → Answer + Citations。  
  用 PPT 绘制即可。

---

### Slide 7：关键方法 3 —— 合成问答数据集（synth_qa.py）

**页面内容**

- 标题：LLM 合成 QA & 自动评测
- 动机：
  - 人工标注 QA 成本高；
  - 需要统一评测集比较 BM25/Dense/Hybrid。
- 做法：
  - 从 `clean.jsonl` 抽样论文；
  - 用 LLM 根据 `title + abstract` 生成问题；
  - 写成 `synth_qa.jsonl`：`{"q": "...", "gold_ids": ["id"], "category": "...", "source": "synthetic_llm"}`；
- 优点：
  - 便宜、可扩展；
  - 与 `eval.py` 接口完全兼容。

**图表/可视化**

- 一个示意：  
  Paper(title+abstract) → LLM → 多个 `q` → 写成 QA jsonl。  
  不需要代码生成图；示意即可。

---

### Slide 8：评估指标 & 实验设计（eval.py）

**页面内容**

- 标题：评估指标与实验设置
- 指标（解释给非 NLP 同学听得懂）：
  - Recall@k：Top-k 内是否命中正确文献 id；
  - MRR：正确文献越靠前，得分越高；
  - Latency：检索/生成/端到端时间。
- 实验设计：
  - 在 `synth_qa.jsonl` 上分别跑 `bm25/dense/hybrid`；
  - 输出 `metrics_synth.csv`；
  - 对真实/人工 dev 集 `dev_qa.jsonl` 也可以做同样评估。

**图表/可视化**

- 这一页主要是文字+简单公式，无需图表。  
  你可以放一个简单的 Recall@k 说明图（一个列表第 1~k 项里标出命中的位置）。

---

### Slide 9：结果示例 & 对比表（来自 eval.py 输出）

**页面内容**

- 标题：结果对比（示例）
- 一张小表格（你后续真正跑完实验后填数值）：

  | mode   | Recall@5 | MRR  | search_ms | end2end_ms |
  |--------|----------|------|-----------|------------|
  | bm25   | 0.60     | 0.45 | 5.3       | 5.3        |
  | dense  | 0.72     | 0.55 | 12.8      | 12.8       |
  | hybrid | 0.76     | 0.59 | 13.1      | 13.1       |

- 讲清楚 trade-off：
  - Hybrid/Dense 在召回上更好；
  - BM25 在速度上有优势。

**图表/可视化**

- 1 张表格是必须的（可直接从 `metrics_synth.csv` 或 `metrics_real.csv` 复制）。  
- 可选：再加一张简单柱状图（Recall@5 vs mode），可用 Excel/Notebook 从 CSV 画。

---

### Slide 10：工程实践 & 总结

**页面内容**

- 标题：工程实践与总结
- 工程点：
  - 流式预处理 4.6GB 大文件；
  - 统一配置 `config.yaml`；
  - 通过 mini 数据 + test3.md 流程，支持快速端到端冒烟测试；
  - 每个模块可独立脚本运行/调试。
- 总结三点：
  1. 双通路 + Hybrid 检索；
  2. 合成 QA + 自动评测；
  3. 简洁、可复用的工程骨架。

**图表/可视化**

- 不需要图表，文字总结为主。  
- 可以在页尾加一句“未来工作”：更强 LLM、更复杂检索、Web Demo 等。

---

## 二、逐页讲稿示例（可直接照读/微调）

> 下面的讲稿按照 10 分钟预期设计，大约每页 40～70 秒。你可以根据自己的语速删减。

---

### Slide 1 讲稿（标题 & 概览）

> 大家好，我是 XXX，这是我这次课设的项目汇报。  
> 项目名字是「基于 arXiv 摘要的学术 RAG 问答系统：双通路检索与合成评测」。
>
> 用一句话概括，就是：我希望做一个“懂论文的问答小助手”。  
> 输入一个学术问题，比如“什么是对比学习？”，系统会在 arXiv 的论文摘要里先检索相关内容，然后把检索结果给一个大语言模型，让它读完后给出答案，并告诉你参考了哪些论文。
>
> 接下来我会从任务背景、数据与系统设计、几个关键方法和实验评估，快速介绍这个项目。

---

### Slide 2 讲稿（任务背景 & 课程要求）

> 这个项目对应课程的第四个主题：面向特定领域问答的检索增强生成。  
> 要求是：在一个特定领域的语料库上，构建一个 RAG 系统，既要用向量数据库做稠密检索，也要跟传统的 BM25 做对比。
>
> 在这个背景下，我选择了 arXiv 的论文摘要作为数据源，聚焦学术领域的问答。  
> 一方面，这是真实的科研语料，问题比较有挑战性；另一方面，也方便我们后续做检索和评估。
>
> 项目有几个硬性要求：  
> 不训练大型语言模型，只用现成的开源或云端 LLM；  
> 检索层要同时支持 BM25 和稠密向量；  
> 并且从准确率、召回率、延迟等维度比较不同检索策略。

---

### Slide 3 讲稿（数据集与预处理）

> 我们的原始数据是 arXiv 官方提供的 metadata 快照，是一个大约 4.6GB 的 JSON Lines 文件，文件中每一行是一个论文的完整元数据。
>
> 对于 RAG 系统来说，不需要那么多字段，我在预处理阶段只保留了五个核心字段：`id、title、abstract、categories、created`。  
> 其中 `categories` 用来表示学科标签，`created` 来自 versions 列表，方便我们后续做筛选或分析。
>
> 预处理脚本 `ingest.py` 做了两件事：  
> 第一，逐行流式读取，防止一次性读入 4.6GB；  
> 第二，对摘要做了简单的清洗，比如合并多行、去掉明显的 LaTeX 公式。  
> 最终输出的是一个更小的 `clean.jsonl`，每一行都只包含我们后续真正需要的字段。

---

### Slide 4 讲稿（系统整体架构）

> 这是整个系统的架构图。
>
> 从左到右看，第一步是 `ingest.py` 做的数据清洗，从原始 JSONL 得到 `clean.jsonl`。  
> 第二步是两个索引模块：`index_bm25.py` 对摘要构建 BM25 倒排索引；`index_dense.py` 使用 sentence-transformers 加上 Chroma，构建稠密向量索引。
>
> 在在线查询阶段，`retriever.py` 提供统一的检索接口。它可以选择只用 BM25、只用稠密，或者做一个 Hybrid 的混合检索。  
> 然后 `rag.py` 会根据检索结果拼接上下文，调用大语言模型生成答案，并附上引用的论文 id 和标题。
>
> 在侧边还有一个 `eval.py`，它用统一格式的 QA 数据集，来评估不同检索策略的召回、MRR 和延迟。  
> 此外，根据这次的扩展需求，我还新增了一个 `synth_qa.py` 用来生成合成问答数据集，后面会重点讲。

---

### Slide 5 讲稿（双通路检索 & Hybrid）

> 在检索层，我做了三种模式：BM25、稠密向量和 Hybrid。
>
> BM25 是经典的关键词匹配方法，优点是实现简单、速度快、解释性强，但对同义词、语义相近表达不太敏感。  
> 稠密检索则是把标题和摘要编码成向量，用向量相似度来衡量相关性，能够更好地处理不同表述之间的语义相似。
>
> Hybrid 模式的思路是把两条路的结果融合起来：  
> 首先分别用 BM25 和 Dense 检索，得到各自的 Top-k 结果；  
> 然后对两边的分数做 min-max 归一化；  
> 最后用一个权重参数 alpha，把稠密分数和 BM25 分数加权合并，得到一个最终分数。
>
> 通过调节 alpha，可以在“更相信 BM25”与“更相信稠密检索”之间做平衡。  
> 这个融合逻辑都被封装在 `retriever.py` 的 `retrieve` 函数里，对上层代码是透明的。

---

### Slide 6 讲稿（RAG & LLM 抽象）

> 在 RAG 部分，`rag.py` 的核心流程非常标准：
>
> 第一步，调用 `retrieve(query, mode, topk)` 来获取若干篇最相关的论文摘要；  
> 第二步，用 `build_context` 函数把这些文档拼成一个上下文字符串，格式大概是 `[id] title + abstract`，中间用分隔符隔开，并控制整体长度不超过模型的上下文限制；
> 第三步，构造好 system prompt 和 user prompt，把问题和上下文一起送给大语言模型。
>
> 为了方便切换不同的模型和部署方式，我做了一个简单的 `LLMClient` 抽象。  
> 它支持三种 provider：`mock` 用于本地调试，直接返回一个伪造答案；`openai` 用于任何 OpenAI 兼容接口；`ollama` 用于本地部署模型。  
> 具体调用大模型 API 的逻辑，就封装在 `LLMClient.generate` 这个函数里。
>
> 最后，`answer` 函数会把生成的答案字符串，和这次检索到的 Top-k 文档的 `id/title` 一起打包成一个统一的输出结构，供前端或者命令行打印使用。

---

### Slide 7 讲稿（合成问答数据集）

> 在原始需求上，我增加的一个比较重要的创新点，是用大模型自动构造一个合成问答数据集，用来做检索评测。
>
> 动机是这样的：  
> 如果我们完全依赖人工标注问题和对应的论文 id，成本会非常高；而只评估少数人工问题又不够稳定。  
> 所以我希望利用现成的大语言模型，基于论文的标题和摘要，自动生成可以在摘要范围内回答的问题。
>
> 具体做法被封装在 `synth_qa.py` 里：  
> 从 `clean.jsonl` 里抽样一批论文，对于每篇论文，调用 LLM，输入 title 和 abstract，请模型生成若干个问题；  
> 然后把每个问题和对应的论文 id 组成一条记录，写入 `synth_qa.jsonl`。  
> 每行的格式大致是：`{"q": 问题, "gold_ids": [论文id], "category": 主类别, "source": "synthetic_llm"}`。
>
> 这样，我们就构造出了一个规模可以调、成本很低的 QA 数据集，并且格式上直接兼容后面的 eval 脚本。

---

### Slide 8 讲稿（评估指标 & 实验设计）

> 有了 QA 数据集之后，我们就可以在检索层做系统化评估。  
> 评估逻辑由 `eval.py` 完成，它支持对多个模式，比如 `bm25/dense/hybrid`，分别计算指标。
>
> 主要指标有三个：
> - 第一是 Recall@k，也就是在前 k 个检索结果中，是否至少命中了一个 gold 文献 id。对于一个问题，如果命中则记 1，没命中记 0，然后求平均；
> - 第二是 MRR，Mean Reciprocal Rank。简单说，就是正确答案在检索结果中排得越前，得分越高，比如排第一得 1.0，排第五就得 0.2；
> - 第三是延迟，包括检索耗时、生成耗时和端到端耗时。  
>
> 实验设计上，合成 QA 和未来可能的人工标注 QA 共用同一个评测脚本。  
> 我们只需要切换输入文件，比如 `dev_qa.jsonl` 或 `synth_qa.jsonl`，就可以直接得到各策略的表现，输出到 `metrics.csv` 里面。

---

### Slide 9 讲稿（结果对比表）

> 这一页展示的是一个结果对比的示例表格。  
> 实际实验可以在合成 QA 集和真实 QA 集上分别跑一遍，得到类似这样的数字。
>
> 从表里我们大致可以观察到几点趋势：  
> - 在 Recall@5 和 MRR 上，稠密检索和 Hybrid 通常要优于纯 BM25，说明语义向量对于复杂提问是有帮助的；  
> - Hybrid 模式因为整合了两条信息源，往往会略高于单一路径；
> - 在检索耗时上，BM25 通常是最快的，而稠密和 Hybrid 会稍慢一些，不过在这个项目规模下，延迟通常仍然在可接受范围内。
>
> 通过这个表格，我们验证了课程作业里关于“对比不同检索策略效果和延迟”的要求，也体现出我们做的双通路 + Hybrid 设计是确实有意义的。

---

### Slide 10 讲稿（工程实践 & 总结）

> 最后简单总结一下工程实践和项目收获。
>
> 在工程层面，首先，我们针对 4.6GB 的原始数据做了流式预处理，并把后续所有模块都设计成读 `clean.jsonl` 的形式，这样后续可以很方便换别的语料。  
> 其次，我们用 `config.yaml` 集中管理索引路径、模型名称和检索参数，使得整个 pipeline 配置化程度比较高。  
> 再次，我们设计了 mini 数据和一套清晰的测试流程，从 ingest 到 eval 都可以用小样本快速冒烟测试。
>
> 在方法层面，三个关键词：  
> 一是双通路检索和 Hybrid 分数融合；  
> 二是用 LLM 自动构造合成 QA 数据集，并让评测流程自动化；  
> 三是通过统一的 RAG 和 LLM 抽象，让整个系统既能跑 mock，又能对接真实的大模型 API。
>
> 后续如果有时间，我希望尝试更强的 embedding 模型，加入重排序或者多轮问答，并且把这个系统封装成一个简单的 Web Demo，方便同学直接体验。

---

## 三、图表与数据可实现性检查

下面逐类检查 PPT 中用到的图表/数据，是否可以由当前项目 + 设计好的代码直接产出。

### 1. 流程图 / 架构图 / 示意图

- 所在页：  
  - Slide 1（简单示意图）  
  - Slide 3（数据流向）  
  - Slide 4（总体 pipeline 图）  
  - Slide 5（BM25/Dense/Hybrid 融合示意）  
  - Slide 6（RAG 流程示意）  
  - Slide 7（合成 QA 流程示意）
- 实现方式：
  - 这些都是**概念图**，直接用 PPT 自带的形状和箭头绘制即可；
  - 不需要项目代码生成，项目代码也没有必要自动画图。
- 结论：  
  **可以完全由 PPT 手动绘制，不依赖代码输出。**

### 2. 数据样例（clean.jsonl & synth_qa.jsonl 行展示）

- 所在页：
  - Slide 3：展示一行 `clean.jsonl`；
  - Slide 7：展示一行 `synth_qa.jsonl`。
- 代码产出情况：
  - `clean.jsonl`：  
    - 由 `src/ingest.py` 生成，命令：  
      ```bash
      python src/ingest.py --in ./data/mini_raw.jsonl --out ./data/clean.jsonl
      ```
    - 可用 `head -n 1 data/clean.jsonl` 获取样例复制到 PPT。
  - `synth_qa.jsonl`：  
    - 由 `src/synth_qa.py` 生成（按你后续实现），命令：  
      ```bash
      python src/synth_qa.py \
        --in ./data/clean.jsonl \
        --out ./data/synth_qa.jsonl \
        --sample_size 3 \
        --questions_per_doc 1
      ```
    - 同样可用 `head -n 1` 获取样例。
- 结论：  
  **在完成对应脚本实现后，可以直接由代码产出真实示例行，拷贝到 PPT。**

### 3. 指标公式 / 概念图（Recall@k / MRR）

- 所在页：
  - Slide 8：指标解释。
- 实现方式：
  - 这部分是数学定义和概念说明，不依赖代码；
  - `eval.py` 内部会实现 `_recall_at_k` 和 `_mrr_at_k`，但 PPT 上只需要文字+简单示意。
- 结论：  
  **无需单独从代码导出图像，手动写即可。**

### 4. 结果对比表（metrics_synth.csv / metrics_real.csv）

- 所在页：
  - Slide 9：结果对比表格。
- 代码产出情况：
  - 表格数据可由 `src/eval.py` 直接生成 CSV：
    - 在合成 QA 集上：
      ```bash
      python src/eval.py \
        --qa ./data/synth_qa.jsonl \
        --modes bm25 dense hybrid \
        --out ./logs/metrics_synth.csv \
        --k 5
      ```
    - 如果有人工标注的 `dev_qa.jsonl`，也可类似地生成 `metrics_real.csv`。
  - 生成的 CSV 列包括：  
    `mode, k, recall, mrr, search_ms, gen_ms, end2end_ms`  
    你可以在 Excel / 表格工具中打开，然后直接复制到 PPT 中做成表。
- 结论：  
  **在实现 eval.py 后，结果表格可以完全由项目代码自动生成，PPT 只需复制格式化。**

### 5. 柱状图（可选）

- 所在页：
  - Slide 9（可选的 Recall@5 柱状图）。
- 代码产出情况：
  - 当前项目设计中，`eval.py` 只输出 CSV，不直接画图；
  - 你可以选一种方式：
    1. 用 Jupyter Notebook 或 Python 脚本，简单读入 CSV，用 matplotlib/seaborn 画柱状图；  
    2. 把 CSV 导入 Excel / WPS，直接插入柱状图；
  - 这不需要改项目主代码，只是一个外部小工具/Notebook。
- 结论：  
  **项目代码已经能提供原始数值；图形化可由 Notebook 或表格软件完成，工作量很小。**

---

## 总体结论

- 这份 PPT 大纲 + 讲稿可以支撑完整的 **10 分钟课堂汇报**，内容量和深度都足够；  
- 所有需要展示的**数据与表格**（clean 样例、合成 QA 样例、评测结果表）都可以由你设计好的项目代码直接生成或间接生成（通过 CSV）；  
- 所有的流程图和示意图可以在 PPT 中手动绘制，不依赖额外工具。

你后续的实际工作主要是：

1. 按 [code3.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/code3.md:0:0-0:0) 把各个脚本实现完成并跑通 [test3.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/%E5%85%B6%E4%BB%96/test3.md:0:0-0:0) 的测试流程；
2. 在合成 QA 或真实 QA 上跑 `eval.py` 得到一份真实的 `metrics_*.csv`；
3. 把这里的 PPT 大纲和讲稿拷贝到你的 PPT 软件中，配上真实的数据表与部分示意图，即可完成汇报准备。
```
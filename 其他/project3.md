# 更新说明

下面是基于你当前代码（`ingest/index_bm25/index_dense/retriever/rag/eval`）和新需求重新整理后的 **完整项目文档**，已经：

- **补充了“合成问答数据 + 自动评测”的创新点与 pipeline**；
- **梳理并精简了结构**，对每个模块的功能和调用流程做了新手友好解释；
- **明确指出了“调用大模型 API 的位置”**；
- 给出了建议的新文件 `src/synth_qa.py` 与 `data/synth_qa.jsonl` 的设计（你后续按这个实现即可）。

---

```markdown
# 项目总览（小白友好版）

本项目的目标：做一个“懂论文的问答小助手”。

你给它一个问题，比如：“什么是对比学习（contrastive learning）？”，系统会：

- 先在一大堆 arXiv 论文摘要里“查资料”（检索模块）；
- 再把查到的摘要丢给一个大语言模型，让它读完后，用自然语言给出解释（生成模块）；
- 最后告诉你：答案主要参考了哪些论文（给出 `id/title/categories`）。

可以把整个系统想象成一座图书馆：

- **arXiv 数据** = 图书馆里所有书的“目录卡片”（标题 + 摘要）。
- **索引（BM25 / 向量库）** = 图书馆的“搜索系统”。
- **`retriever` 检索模块** = 帮你下检索指令的“图书管理员”。
- **大语言模型（LLM）** = 会读论文、会总结的“专业助教”。
- **RAG** = “查资料 + 让助教回答”连在一起的一整套流程。

---

# 1. 项目目标与任务要求

## 1.1 课程题目

**主题 4：面向特定领域问答的检索增强生成（RAG）**

主要要求：

- 选取一个特定领域语料库（本项目使用 **arXiv 摘要数据**）；
- 使用 FAISS / Milvus / Chroma 等工具构建向量数据库（本项目用 **Chroma**）；
- 将检索流程与开源大模型集成（如 Qwen2.5-7B-Instruct、LLaMA-3-8B-Instruct 或任意 OpenAI 兼容模型），完成问答系统；
- 从 **准确率/召回率/延迟** 等维度，对比 **稠密检索 vs BM25**。

## 1.2 本项目的具体目标

- **目标**  
  基于 arXiv 摘要构建一个学术领域 RAG 问答系统，支持：
  - 传统 BM25 检索；
  - 稠密向量检索（Chroma + `sentence-transformers`）；
  - 两者的混合（Hybrid）检索，并做效果 & 性能对比。

- **范围**  
  - 只使用 **摘要 + 元数据**，不解析 PDF 全文；
  - 不微调 LLM，只使用现成开源/云端模型；
  - 聚焦算法与工程实现，UI 只做最简命令行。

- **输出**  
  - 给定用户问题 `query`：
    - 输出：`answer` 文本；
    - 以及若干命中文档的 `id/title/categories`（Top-k）。

- **评估维度**  
  - **检索层**：Recall@k、MRR；
  - **性能层**：检索耗时、生成耗时、端到端耗时；
  - **资源层**：索引大小、内存占用（定性描述为主）。

- **非目标**  
  - 不做复杂 Web UI；
  - 不涉及多用户管理、权限控制等工程化功能。

---

# 2. 主要功能与创新点

## 2.1 功能总览

- **大文件预处理**：从 4.6GB 原始 arXiv JSONL 中流式抽取 `id/title/abstract/categories/created`；
- **双通路索引**：
  - BM25 倒排索引（关键词匹配）；
  - 稠密向量索引（语义匹配，Chroma + embedding 模型）；
- **统一检索接口**：`retrieve(query, topk, mode="bm25|dense|hybrid")`；
- **RAG 生成**：从检索结果构建上下文，调用 LLM API 生成答案并附带引用；
- **离线评估**：对 BM25 / Dense / Hybrid 做召回与延迟对比；
- **合成问答数据**（新需求）：利用 LLM 对摘要自动生成问题，构造评测集，在该集上计算准确率/召回率。

## 2.2 创新点（重点写在报告里的部分）

- **创新点 1：双通路 + 混合检索策略**
  - 传统 BM25 & 稠密检索在同一套代码框架下统一（`src/retriever.py`）；
  - 通过简单的 min-max 归一化 + 权重 `alpha` 实现 Hybrid 融合；
  - 提供多种模式对比：`bm25 / dense / hybrid`。

- **创新点 2：LLM 驱动的合成问答数据集 + 自动评测**（新）
  - 新增「合成问答生成模块」：从 `clean.jsonl` 中抽样若干篇论文摘要；
  - 使用 LLM 读取每篇的 `title + abstract`，自动生成 1~N 个相关问题；
  - 以 `(question, gold_ids=[论文 id])` 形式写入 `data/synth_qa.jsonl`；
  - 直接复用现有 `src/eval.py` 在该数据集上计算 **检索准确率/召回率/MRR**；
  - 大幅减少人工标注成本，构成 **“合成评测集”** 的闭环。

- **创新点 3：统一 LLM 抽象 & 配置化**
  - `src/rag.py` 中的 `LLMClient` 对大模型调用做了简单封装：
    - 支持 `provider = openai / ollama / mock`；
    - 通过 `configs/config.yaml` 配置 `model / max_tokens` 等；
  - 问答生成与合成问答（问题生成）可以共享同一 LLMClient 抽象。

- **创新点 4：简洁的工程化实践（大文件 + 测试 + mini 数据）**
  - 预处理阶段采用流式读取，适应 4.6GB 大文件；
  - 提供 `tests/data/mini.jsonl`，方便用小样本在本机快速跑通全流程；
  - 核心模块均提供可单独调用的函数接口，利于后续扩展。

---

# 3. 输入数据说明

## 3.1 数据路径

```text
./data/arxiv-metadata-oai-snapshot.json
```

- 体量约 4.6GB；
- 格式：JSON Lines（每行一个 JSON 对象），UTF-8。

## 3.2 数据示例与结构

（以下是单行解析后的 JSON 示例）

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

本项目预处理后保留的核心字段：

- `id`：arXiv 唯一标识；
- `title`：论文标题；
- `abstract`：摘要（可能包含 LaTeX）；
- `categories`：空格分隔的学科标签（如 `cs.CL stat.ML`）；
- `created`：从 `versions[0].created` 中解析出的时间字符串（可进一步提取年份）。

## 3.3 大文件读取建议

- 使用标准库逐行读取并 `json.loads`，不要一次性加载整个文件；
- 对调试可加 `max_rows` 限制，只看前几行。

---

# 4. 项目结构总览

```text
5020-RAG/
├── data/
│   ├── arxiv-metadata-oai-snapshot.json      # 原始 JSONL (4.6GB)
│   ├── clean.jsonl                           # [阶段1输出] 精简字段
│   ├── bm25.idx                              # [阶段2输出] BM25 序列化索引
│   ├── chroma/                               # [阶段2输出] Chroma 稠密向量库
│   └── synth_qa.jsonl                        # [新] 合成问答评测集（待实现）
├── src/
│   ├── ingest.py                             # 阶段1：数据预处理
│   ├── index_bm25.py                         # 阶段2：BM25 索引
│   ├── index_dense.py                        # 阶段2：稠密向量索引（Chroma）
│   ├── retriever.py                          # 阶段3：统一检索接口（bm25/dense/hybrid）
│   ├── rag.py                                # 阶段4：RAG 上下文构建 + 调用大模型 API
│   ├── eval.py                               # 阶段5/6：离线评测（检索 & 可选生成）
│   └── synth_qa.py                           # [新拟增] 合成问答数据生成脚本（需按设计实现）
├── tests/                                    # 单元/集成测试（可按需要补充）
│   └── data/
│       └── mini.jsonl                        # 迷你样本（5~50 行）
├── configs/
│   └── config.yaml                           # 索引/检索/生成/运行 参数配置
├── logs/
│   └── metrics.csv                           # 评估输出（真实或合成 QA）
└── requirements.txt
```

> 说明：`synth_qa.py` 和 `synth_qa.jsonl` 是本次新需求的**设计目标**，你可以按下面的流程实现。

---

# 5. 模块与代码流程（按文件）

## 5.1 `src/ingest.py` —— 数据预处理

**作用**：  
从大文件 `arxiv-metadata-oai-snapshot.json` 流式读取每行 JSON，抽取并清洗：

- 字段：`id/title/abstract/categories/created`；
- 摘要中简单去除多余换行和明显的 LaTeX 公式片段。

**核心函数（设计）**

- `stream_clean_arxiv(input_path: str, output_path: str) -> None`  
  - 输入：原始 JSONL 路径；
  - 输出：`clean.jsonl`，每行一个精简 JSON。

**命令示例**

```bash
python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl
```

---

## 5.2 `src/index_bm25.py` —— BM25 索引

**作用**：  
对 `clean.jsonl` 中的 `title + abstract` 建立 BM25 检索索引。

**流程**

1. 逐行读取 `clean.jsonl`；
2. 构造文本 `text = title + "\n" + abstract`；
3. 用 `rank_bm25.BM25Okapi` 建立索引；
4. 将 `docs` + `tokens` 序列化到 `bm25.idx`。

**主要函数**

- `build_bm25_index(clean_path: str, index_path: str) -> int`  
- `search_bm25(index_path: str, query: str, topk: int = 5) -> List[Dict]`  

返回的每个文档为：

```python
{"id": str, "title": str, "abstract": str, "text": str, "score": float}
```

---

## 5.3 `src/index_dense.py` —— 稠密向量库（Chroma）

**作用**：  
将同样的 `title + abstract` 编码成向量，并存入 Chroma 向量库。

**流程**

1. 读取 `clean.jsonl`，构造 `{"id", "title", "text"}`；
2. 使用 `SentenceTransformer(model_name)` 编码文本，得到向量；
3. 使用 `chromadb.PersistentClient`，创建 `collection="arxiv"`；
4. 向 collection 中批量 `add(ids, documents, metadatas, embeddings)`。

**主要函数**

- `embed_and_build_vector_db(clean_path, db_path, model_name, collection="arxiv", batch_size=64) -> int`  
- `search_dense(db_path, query, topk, model_name, collection="arxiv") -> List[Dict]`

输出结构（单条）：

```python
{"id": str, "title": str, "text": str, "score": float}  # 这里的 score = 1 - distance
```

---

## 5.4 `src/retriever.py` —— 统一检索接口

**作用**：  
统一封装 BM25 & Dense & Hybrid 三种检索模式，对上层提供同一函数：

```python
retrieve(query: str, topk: int = 5, mode: str = "bm25|dense|hybrid", alpha: float = 0.5)
```

**Hybrid 融合思路**

- 分别调用 `search_bm25` 和 `search_dense`；
- 将两路结果按 `id` 对齐，并对分数做 min-max 归一化；
- 用 `final_score = alpha * dense_norm + (1-alpha) * bm25_norm` 合并；
- 按 `final_score` 排序，取 Top-k。

---

## 5.5 `src/rag.py` —— RAG 上下文构建 & 调大模型 API

**作用**：

1. 调用 `retriever.retrieve` 得到 Top-k 文档；
2. 将这些文档的 `title + abstract` 拼接成上下文；
3. 调用 LLM API 生成最终答案，并附带引用列表。

**大模型 API 在哪里调用？**

- 在 `src/rag.py` 的类 `LLMClient` 中：
  - 当 `provider="openai"` 或 `"ollama"` 时，内部使用 `OpenAI` 客户端发起 **真正的 HTTP API 调用**；
  - 当 `provider="mock"` 时，只返回一个模拟答案字符串（方便离线调试）。
- 函数 `answer(...)` 会：
  1. 读取配置 `configs/config.yaml` 中的 generation 参数；
  2. 调 `LLMClient.generate(system_prompt, user_prompt)`；
  3. 返回结构：

```python
{"answer": str, "citations": [{"id": ..., "title": ...}, ...]}
```

---

## 5.6 `src/synth_qa.py` —— 合成问答数据生成（新需求，设计稿）

> 说明：此文件是本次新增的 **创新模块**，下面是推荐的设计与接口，方便你实现。

**目标**：  
利用 LLM 读取论文 `title + abstract`，自动生成若干自然语言问题，构造成用于评测检索效果的合成 QA 数据集。

**推荐数据格式：`data/synth_qa.jsonl`**

每行一个 JSON 对象：

```json
{
  "q": "What is the main contribution of this paper?",
  "gold_ids": ["0704.0001"],
  "source": "synthetic_llm",
  "category": "cs.CL"
}
```

- `q`：由 LLM 生成的英文或中文问题；
- `gold_ids`：正确答案对应的论文 id 列表（这里通常只包含当前论文的 id）；
- `source`：标记这是合成数据；
- `category`：可选，来自原始 `categories` 的主标签（如 `cs.CL`）。

**建议函数接口**

```python
def generate_questions_for_doc(title: str, abstract: str, n_q: int, client: LLMClient) -> List[str]:
    """
    输入单篇论文标题+摘要，让 LLM 生成 n_q 个相关问题。
    """

def generate_synthetic_qa(
    clean_path: str,
    out_path: str,
    sample_size: int = 1000,
    questions_per_doc: int = 2,
    category_filter: Optional[str] = None,
) -> int:
    """
    从 clean.jsonl 中抽样若干篇论文，为每篇生成问题并写出 JSONL。
    """
```

**命令示例**

```bash
# 从 clean.jsonl 抽样 500 篇论文，每篇生成 2 个问题，输出到 data/synth_qa.jsonl
python src/synth_qa.py \
  --in ./data/clean.jsonl \
  --out ./data/synth_qa.jsonl \
  --sample_size 500 \
  --questions_per_doc 2
```

**与 LLM 的关系**

- 可以直接复用 `rag.py` 中的 `LLMClient`：
  - system_prompt：例如「你是学术助教，根据给定论文标题和摘要，生成可以用该摘要回答的若干问题」；
  - user_prompt：包含论文标题与摘要，并要求输出一个问题列表（JSON 格式）。

---

## 5.7 `src/eval.py` —— 离线评测（可用合成数据）

**作用**：

- 读取评测集 JSONL（无论是人工标注的 `dev_qa.jsonl`，还是合成的 `synth_qa.jsonl`）；
- 针对每条数据：
  - 使用 `retrieve(q, k, mode)` 得到检索结果；
  - 计算 Recall@k 与 MRR；
  - 记录检索耗时 & 可选生成耗时；
- 按检索模式聚合后输出到 CSV。

**评测集格式要求**

每行 JSON 至少包含字段：

```json
{"q": "...", "gold_ids": ["0704.0001", "..."]}
```

这与我们设计的 `synth_qa.jsonl` 完全兼容。

**主要函数**

```python
evaluate(qa_path: str, modes: List[str], out_csv: str, k: int = 5, include_gen: bool = False) -> None
```

**命令示例**

- 在 **合成问答数据集** 上评测检索性能：

```bash
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes bm25 dense hybrid \
  --out ./logs/metrics_synth.csv \
  --k 5
```

- 若想同时计入 LLM 生成答案的耗时，可以加 `--include_gen`。

---

# 6. 端到端 Pipeline（带命令）

## 阶段 0：环境准备

```bash
pip install -U -r requirements.txt
```

## 阶段 1：数据预处理

```bash
python src/ingest.py \
  --in ./data/arxiv-metadata-oai-snapshot.json \
  --out ./data/clean.jsonl
```

## 阶段 2：索引构建（BM25 + 稠密）

```bash
# 2.1 BM25 索引
python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx

# 2.2 稠密向量索引（Chroma）
python src/index_dense.py \
  --in ./data/clean.jsonl \
  --db ./data/chroma \
  --model bge-small-en-v1.5
```

## 阶段 3：检索（单路/混合）

在 Python 中：

```bash
python -c "from src.retriever import retrieve; print(retrieve('contrastive learning', 5, 'hybrid'))"
```

## 阶段 4：RAG 回答（调用大模型 API）

```bash
python -c "from src.rag import answer; print(answer('What is contrastive learning?', 'hybrid', 5))"
```

- 此步骤内部会：
  - 调 `retriever.retrieve` 拿到文档；
  - 用 `build_context` 拼接上下文；
  - 用 `LLMClient.generate` **调用大模型 API**；
  - 返回答案和引用。

## 阶段 5：合成问答数据生成（新）

> 需要你按设计实现 `src/synth_qa.py` 后使用。

```bash
python src/synth_qa.py \
  --in ./data/clean.jsonl \
  --out ./data/synth_qa.jsonl \
  --sample_size 500 \
  --questions_per_doc 2
```

## 阶段 6：离线评估（支持真实 & 合成 QA）

- 对真实/人工标注的 dev 集：

```bash
python src/eval.py \
  --qa ./data/dev_qa.jsonl \
  --modes bm25 dense hybrid \
  --out ./logs/metrics_real.csv \
  --k 5
```

- 对 **合成 QA 集 `synth_qa.jsonl`**：

```bash
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes bm25 dense hybrid \
  --out ./logs/metrics_synth.csv \
  --k 5
```

比较两份 metrics，可以分析：

- 不同检索策略的检索准确率/召回率；
- 在合成问题 vs 人工问题上的表现差异；
- 各模式的延迟表现。

---

# 7. 配置文件与环境变量

示例：`configs/config.yaml`（已简化）

```yaml
data:
  raw: ./data/arxiv-metadata-oai-snapshot.json
  clean: ./data/clean.jsonl

bm25:
  index_path: ./data/bm25.idx

dense:
  db: ./data/chroma
  provider: chroma
  model: bge-small-en-v1.5
  collection: arxiv

retrieval:
  mode: hybrid
  topk: 5
  alpha: 0.5

generation:
  provider: mock        # mock | openai | ollama
  model: gpt-4o-mini    # 或 Qwen2.5/LLaMA 等
  max_tokens: 512

runtime:
  max_context_chars: 6000
  seed: 42
```

若使用 OpenAI 兼容 API，需要设置：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`（如用本地 OpenAI 兼容服务）

---

# 8. 常见术语解释（小白版）

- **BM25**：一种传统的“关键词匹配”检索算法，适合问题与文档中词面高度重合的场景。
- **稠密检索（Dense Retrieval）**：将文本编码成向量（Embedding），类似“把一句话变成一串数字”，语义相近的句子距离更近。
- **向量数据库（Vector DB）**：专门存储和检索这些向量的数据库，如 FAISS、Chroma、Milvus。
- **RAG**：检索增强生成（Retrieval-Augmented Generation），先查资料，再让大模型基于这些资料回答。

---

# 9. 常见问题（FAQ）

**Q1：我们调用大模型 API 的步骤是在代码的哪里？**

- 在 `src/rag.py` 中：
  - `LLMClient.generate(...)` 内部使用 `OpenAI` 客户端发起 API 请求；
  - 函数 `answer(query, mode, topk)` 会：
    1. 调用 `retrieve` 得到文档；
    2. 调用 `build_context` 构造上下文；
    3. 最终通过 `LLMClient.generate` 调大模型 API，得到回答。
- 未来新增的 `src/synth_qa.py` 也会复用 `LLMClient`，用同一个 API 实例来 **“生成问题”**，从而构造合成评测数据集。

```

---

## 总结

- 我已经把你原来的项目说明 **完整重组**，并按照你的新需求：
  - 设计了 **合成问答数据生成模块 `synth_qa.py`**；
  - 说明了如何利用 `synth_qa.jsonl + eval.py` 在合成数据上计算准确率/召回率等指标；
  - 明确标注了 **大模型 API 调用位置** 和完整代码流程。

如果你希望，我可以下一步帮你 **按这个设计真正实现 `src/synth_qa.py` 的代码**（包括 prompt 模板和命令行参数），再配一个最小化的样例运行脚本。
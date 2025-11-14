我现在不能直接改文件，只能把**最终版 [project.md](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG/project.md:0:0-0:0) 完整给你**。  
你可以直接“全选替换”为下面这份内容，会比你现在的版本更适合完全小白阅读。

---

```markdown
## 项目总览（小白友好版）

本项目的目标：做一个“懂论文的问答小助手”。

你给它一个问题，比如“什么是对比学习（contrastive learning）？”，它会：

- 先在一大堆论文摘要里“查资料”；
- 再把查到的摘要丢给一个大语言模型，让模型读完之后，用自然语言给你解释；
- 最后告诉你：答案主要参考了哪几篇论文（给出 id 和标题）。

### 用生活类比快速理解

可以把整个系统想象成一座图书馆：

- **arXiv 数据**：图书馆里所有书的“目录卡片”（标题 + 摘要）。
- **索引（BM25 / 向量库）**：图书馆的“检索系统”，帮你快速从几百万条记录里找到可能相关的几十条。
- **检索模块 `retriever`**：帮你用搜索系统下指令的“图书馆管理员”。
- **大语言模型（LLM）**：会读论文、会总结的“专业助教”。
- **RAG**：把“查资料”和“请助教回答”这两步连在一起的完整流程。

### 一句话看懂整体流程

从你提问到拿到答案，大致分成 6 步：

1. 准备原始数据：一份很大的 arXiv 元数据文件，每一行是一篇论文的信息（包含标题和摘要）。
2. 用 `src/ingest.py` 把大文件“瘦身”，只留下后续真正需要的字段，生成体积更小的 `clean.jsonl`。
3. 用 `src/index_bm25.py` 和 `src/index_dense.py` 对这些摘要建立两套检索索引（BM25 和稠密向量）。
4. 当有人提问时，用 `src/retriever.py` 读这两套索引，找出最可能相关的若干篇论文。
5. 用 `src/rag.py` 把这些论文摘要拼成一个“上下文”，再连同你的问题一起发送给大语言模型 API。
6. 大模型读完上下文，写出答案，并附上它参考到的论文 `id/title`，最后把结果返回给你。

### 数据是如何一步步流动的？

先看“数据文件”的流向：

```text
原始数据：arxiv-metadata-oai-snapshot.json  （几 GB 的大文件）
        │  经由 src/ingest.py 清洗精简
        ▼
中间数据：clean.jsonl                         （只保留 id/title/abstract 等核心字段）
        │  经由 src/index_bm25.py / src/index_dense.py 建索引
        ▼
索引数据：bm25.idx  和  chroma/              （两套检索索引）
        │  被 src/retriever.py 和 src/rag.py 使用
        ▼
最终输出：answer 文本 + 引用的论文 id/title/categories
```

再看“从提问到答案”的流程：

```text
用户问题（自然语言）
        │
        ▼
src/retriever.py: retrieve(...)   —— 根据索引找出 Top-k 篇最相关的论文
        │
        ▼
src/rag.py: build_context(...)    —— 把这些论文的标题和摘要拼成一个长上下文
        │
        ▼
src/rag.py: answer(...)           —— 调用大语言模型 API，要求它“只根据上下文回答问题”
        │
        ▼
得到：回答文本 + 引用论文列表（id/title/categories）
```

### 如果你不会编程，只想跑通一遍

按顺序在命令行里执行（前提是：装好 Python，知道怎么打开终端）：

1. 安装依赖（只需要做一次）：

   ```bash
   pip install -U -r requirements.txt
   ```

2. 做数据清洗（把大文件变成精简版）：

   ```bash
   python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl
   ```

3. 构建两套索引（关键词检索 + 语义检索）：

   ```bash
   python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
   python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-zh-v1.5
   ```

4. 实际提问，得到答案：

   ```bash
   python -c "from src.rag import answer; print(answer('What is contrastive learning?', 'hybrid', 5))"
   ```

5. （可选）做离线评估，看不同检索方式的效果与速度：

   ```bash
   python src/eval.py --qa ./data/dev_qa.jsonl --modes bm25 dense hybrid --out ./logs/metrics.csv
   ```

跑通 1～4 步，你就完成了一个简化版的 RAG 问答系统。

---

## prompt

"""
我有一个课程项目作业主题如下：
##主题4：面向特定领域问答的检索增强生成（RAG）
本项目聚焦构建无需训练大型语言模型的检索增强问答系统。
学生需完成以下工作：
- 收集或选择特定领域语料库。
- 使用FAISS、Milvus或Chroma等工具构建向量数据库，用于文档检索。
- 将检索流程与开源大语言模型（例如Qwen2.5-7B-Instruct、LLaMA-3-8B-Instruct）集成，以回答用户查询。
- 从准确率、召回率和延迟等维度，对比不同检索策略（稠密检索与BM25）并评估系统性能。

数据：
- arXiv开放摘要数据集

我需要你按照给我完成以下项目需求文档，按照精简的来，不需要过于复杂和健壮
"""

---

## 项目需求

- **目标**：基于 arXiv 摘要构建面向学术领域的 RAG 问答系统；同时支持稠密检索与 BM25，进行效果与性能对比。
- **范围**：只使用摘要与元数据，不做 PDF 全文解析；不微调大模型，仅集成现成开源模型（如 Qwen2.5-7B-Instruct、LLaMA-3-8B-Instruct 或本地/云端 API）。
- **输出**：答案文本 + 引用的文档 `id/title/categories`（可返回 Top-k 命中文档）。
- **对比维度**：
  - 准确率 / 召回率（通过小规模标注或简单评测集）
  - 延迟（检索耗时、生成耗时、端到端耗时）
  - 资源占用（索引大小、内存）
- **非目标**：不涉及复杂 UI、长流程编排、在线多用户管理。

---

## 输入数据

### 数据路径

```text
./data/arxiv-metadata-oai-snapshot.json
```

该文件为 arXiv 元数据快照，体量约 4.6GB，采用 JSON Lines（每行一个 JSON 对象）格式，UTF-8 编码。

### 数据示例

```json
{
  "id": "0704.0001",
  "submitter": "John Doe",
  "authors": "John Doe, Jane Smith",
  "title": "An Example Paper Title",
  "comments": "12 pages, 3 figures",
  "journal-ref": "J. Example 12 (2007) 34-56",
  "doi": "10.1000/example.doi",
  "report-no": null,
  "categories": "cs.CL stat.ML",
  "license": null,
  "abstract": "We propose an approach to ...",
  "versions": [
    { "version": "v1", "created": "Mon, 2 Apr 2007 12:00:00 GMT", "source": "arXiv" }
  ],
  "update_date": "2007-04-10",
  "authors_parsed": [["Doe", "John", ""], ["Smith", "Jane", ""]]
}
```

> 实际文件中每一行即为一个类似上面的完整 JSON 对象。

### 数据字段说明（口语版）

- `id`：论文在 arXiv 上的唯一编号。
- `title`：论文标题。
- `abstract`：论文摘要（主要用来“查资料”）。
- `categories`：论文所属领域（例如 `cs.CL` 表示计算语言学）。
- `versions`：该论文的多个版本信息，我们只用里面的时间字段 `created` 来推年份等。
- 其他字段（作者、期刊信息等）在本项目里不是核心，但可以保留。

### 读取与处理建议（因为文件很大）

- 采用“按行读取”的方式，而不是一次性全部读入内存。
- 每读一行，就 `json.loads` 一次，抽取我们需要的字段。
- 出错的行（坏 JSON）可以跳过，并做简单日志记录。

示例（只看前 3 行）：

```python
import json

path = "./data/arxiv-metadata-oai-snapshot.json"
max_rows = 3

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= max_rows:
            break
        rec = json.loads(line)
        print({
            "id": rec.get("id"),
            "title": rec.get("title"),
            "categories": rec.get("categories"),
            "created": (rec.get("versions") or [{}])[0].get("created")
        })
```

---

## pipeline 总览（按阶段分步执行）

为了方便落地，把整个项目拆成几个“阶段”（你可以按顺序实现）：

### 阶段 0：准备

- 操作：安装依赖、准备配置。
- 代码：`requirements.txt`、`configs/config.yaml`
- 命令：

  ```bash
  pip install -U -r requirements.txt
  ```

### 阶段 1：数据预处理（清洗精简）

- 操作：流式读取原始 JSONL，抽取并清洗核心字段。
- 代码：`src/ingest.py: stream_clean_arxiv`
- 输入：`data/arxiv-metadata-oai-snapshot.json`
- 输出：`data/clean.jsonl`，字段 `id/title/abstract/categories/created`
- 示例命令：

  ```bash
  python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl
  ```

### 阶段 2：索引构建（双通路）

- 操作 2.1：BM25 倒排索引构建
  - 代码：`src/index_bm25.py: build_bm25_index`
  - 输入：`data/clean.jsonl`
  - 输出：`data/bm25.idx`
  - 示例命令：

    ```bash
    python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
    ```

- 操作 2.2：稠密向量库构建（Chroma/FAISS）
  - 代码：`src/index_dense.py: embed_and_build_vector_db`
  - 输入：`data/clean.jsonl`
  - 输出：`data/chroma/` 或 FAISS 索引文件
  - 示例命令：

    ```bash
    python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-zh-v1.5
    ```

### 阶段 3：检索（单路 / 混合）

- 操作 3.1：仅 BM25 检索
  - 代码：`src/retriever.py: retrieve`（`mode="bm25"`）
  - 依赖：`src/index_bm25.py: search_bm25`

- 操作 3.2：仅稠密检索
  - 代码：`src/retriever.py: retrieve`（`mode="dense"`）
  - 依赖：`src/index_dense.py: search_dense`

- 操作 3.3：混合检索（分数归一化 + 加权融合）
  - 代码：`src/retriever.py: retrieve`（`mode="hybrid"`, `alpha=0.5` 可调）
  - 示例：

    ```bash
    python -c "from src.retriever import retrieve; print(retrieve('contrastive learning', 5, 'hybrid', 0.5))"
    ```

### 阶段 4：上下文构建与生成回答

- 操作 4.1：拼接上下文（标题 + 摘要，分隔符，长度裁剪）
  - 代码：`src/rag.py: build_context`
- 操作 4.2：调用 LLM 生成答案并附引用
  - 代码：`src/rag.py: answer`
  - 示例：

    ```bash
    python -c "from src.rag import answer; print(answer('What is contrastive learning?', 'hybrid', 5))"
    ```

### 阶段 5：离线评估

- 操作：对 bm25/dense/hybrid 做召回与延迟对比
  - 代码：`src/eval.py: evaluate`
  - 输入：`data/dev_qa.jsonl`（字段：`q` / `gold_ids`）
  - 输出：`logs/metrics.csv`（Recall@k / MRR / Latency）
  - 示例命令：

    ```bash
    python src/eval.py --qa ./data/dev_qa.jsonl --modes bm25 dense hybrid --out ./logs/metrics.csv
    ```

### 阶段 6：测试与验收（自动化 + 冒烟）

- 单元测试：针对各模块的小范围测试。
- 集成 / 冒烟测试：用 `tests/data/mini.jsonl` 之类的小数据，从头跑到尾，看是否能输出合理结果。

---

## 项目结构与模块关系

```text
5020-RAG/
├── data/
│   ├── arxiv-metadata-oai-snapshot.json      # 原始 JSONL (4.6GB)
│   ├── clean.jsonl                           # [预处理输出] 精简字段
│   ├── bm25.idx                              # [索引输出] BM25 序列化文件
│   └── chroma/                               # [索引输出] Chroma/FAISS 向量库
├── src/
│   ├── ingest.py                             # 数据读取与清洗
│   ├── index_bm25.py                         # 构建/查询 BM25
│   ├── index_dense.py                        # 构建/查询 稠密向量库
│   ├── retriever.py                          # 统一检索接口（bm25/dense/hybrid）
│   ├── rag.py                                # 检索-拼接-生成 策略
│   └── eval.py                               # 评测脚本
├── tests/                                    # 自动化测试（单元/集成/验收）
│   ├── data/
│   │   └── mini.jsonl                        # 迷你样本（5~50 行）
│   ├── test_ingest.py                        # 测 ingest 清洗/字段完整性
│   ├── test_index_bm25.py                    # 测 BM25 构建/查询
│   ├── test_index_dense.py                   # 测 向量库构建/查询
│   ├── test_retriever.py                     # 测 混合检索融合/去重/Top-k
│   ├── test_rag.py                           # 测 上下文拼接/返回结构
│   └── test_eval.py                          # 测 指标计算与输出
├── configs/
│   └── config.yaml                           # 模型/索引/检索/生成 参数
├── notebooks/
│   └── arxiv-abstracts-exploration.ipynb     # 数据探索
├── logs/                                     # 运行日志/评估输出
└── requirements.txt
```

- 模块关系：`ingest` → `index_bm25` / `index_dense` → `retriever` → `rag` → `eval`（评估在侧）

---

## 各阶段详细说明（写给新手）

这一节把前面 pipeline 的每一步，用更口语化的方式再解释一遍。

### 1）数据预处理：`src/ingest.py`

**在做什么？**

- 把一个超大的原始 JSONL 文件，变成一个“只保留关键字段的小文件”。
- 关键字段：`id/title/abstract/categories/created`。

**为什么要做？**

- 原始文件非常大、字段很多。
- 如果直接拿来建索引，会：
  - 读取慢；
  - 占用内存高；
  - 后续调试很不方便。
- 先做精简，相当于给后面的步骤“减负”。

**输入 / 输出长什么样？**

- 输入：`arxiv-metadata-oai-snapshot.json`（每行一个完整论文 JSON）。
- 输出：`clean.jsonl`（每行一个“精简版 JSON”，结构类似）：

  ```json
  {"id": "0704.0001", "title": "...", "abstract": "...", "categories": "cs.CL stat.ML", "created": "Mon, 2 Apr 2007 ..."}
  ```

**主要函数建议**

- `stream_clean_arxiv(input_path: str, output_path: str) -> None`
  - 按行读取；
  - `json.loads` 后取字段；
  - 对摘要去掉多余换行、简单处理 LaTeX；
  - 每成功处理一行，就写一行新的 JSON 到输出文件。

**如何简单自检？**

- 只处理前几百行（加一个调试参数 `--max_rows`）。
- 打印前几行输出，检查字段是否齐全、有无严重乱码。

---

### 2）BM25 索引：`src/index_bm25.py`

**BM25 是什么？（超简版）**

- 一种传统“按关键词匹配”的检索方法。
- 问题里的词在文档中出现得多，而且这些词在整个库里比较少见 ⇒ 文档得分高。

**这一步在做什么？**

- 读取 `clean.jsonl`。
- 把每篇论文的「标题 + 摘要」作为一个“文档”。
- 用 BM25 建立一个检索索引。
- 把索引（以及 id 映射）存到 `bm25.idx`。

**主要函数建议**

- `build_bm25_index(clean_path: str, index_path: str) -> None`
  - 把文档分词并喂给 BM25；
  - 序列化保存。
- `search_bm25(index_path: str, query: str, topk: int)`
  - 读入索引；
  - 对查询分词，计算得分，返回 Top-k 文档（含 `id/title/score`）。

**简单自检**

- 选一个明显的技术词（如 `contrastive`），看返回的论文标题是否“看起来对”。

---

### 3）稠密向量库：`src/index_dense.py`

**“稠密嵌入”是啥？**

- 把一句话变成一串数字（比如长度 768 的向量）。
- 语义相近的句子 ⇒ 数字向量之间的距离比较近。

**这一步在做什么？**

- 给每篇论文的“标题 + 摘要”算出一个向量表示。
- 把这些向量存入一个向量数据库（如 Chroma）。
- 将来用户提问时，只要：
  - 把问题转成向量；
  - 去库里找距离最近的若干向量，
  - 即可找到“语义相近”的论文。

**主要函数建议**

- `embed_and_build_vector_db(clean_path: str, db_path: str, model_name: str) -> None`
- `search_dense(db_path: str, query: str, topk: int, model_name: str)`

**模型选择建议**

- 选一个轻量模型，如 `bge-small-zh-v1.5` 或 `e5-base`。
- 用 `cpu` 就能跑，只是慢一点；有 GPU 则更快。

---

### 4）统一检索接口：`src/retriever.py`

**为什么需要统一接口？**

- 我们有两种检索方式：BM25（关键词）和稠密检索（语义）。
- 直接让上层代码处理两个接口会很乱。
- 封装成一个统一函数 `retrieve(...)`，只需要指定 `mode`：

  - `mode="bm25"`：只用 BM25；
  - `mode="dense"`：只用稠密检索；
  - `mode="hybrid"`：混合，两路都用。

**混合检索怎么理解？**

- BM25：对“词面匹配”敏感；
- 稠密：对“语义相似”敏感；
- 两种各有优缺点。
- 做法：
  - 分别算两套得分；
  - 把分数归一化到 0～1；
  - 用 `final_score = alpha * dense_score + (1-alpha) * bm25_score` 做加权平均；
  - 对相同 `id` 的文档去重，只保留最高得分。

---

### 5）RAG 组装与回答：`src/rag.py`

**RAG 的核心思想**

- 模型不要“凭空瞎编”，而是“先查资料，再回答”。
- 我们先用检索模块拿到相关论文，再把这些论文的摘要拼成上下文，最后让大模型在这个上下文基础上作答。

**`build_context` 在做什么？**

- 输入：若干篇检索到的论文（含标题、摘要、得分）。
- 它会：
  - 按得分排序；
  - 使用一个分隔符（如 `\n\n---\n\n`）把不同论文隔开；
  - 限制总长度不超过 `max_chars`，避免超出模型上下文长度。

**`answer` 在做什么？**

- 输入：`query`（用户问题）、`mode`（检索方式）、`topk`。
- 过程：
  1. 调 `retrieve(...)` 找论文；
  2. 调 `build_context(...)` 拼接上下文；
  3. 构造一个 prompt（说明：只能根据上下文回答，并在结尾给出引用）；
  4. 调用大语言模型 API；
  5. 返回一个字典：`{"answer": "...", "citations": [{"id": ..., "title": ...}, ...]}`。

> **注意**：真正“调用大模型 API”的代码，就在这一步（`rag.answer` 内部或它调用的一个子函数里）。

---

### 6）评测：`src/eval.py`

**我们想评什么？**

- 不同检索方式（BM25 / dense / hybrid）：
  - 能否把真正相关的论文捞出来？
  - 捞得靠不靠前？
  - 速度如何？

**常见指标（只要大致理解就够）**

- `Recall@k`：在应该命中的文献中，有多少出现在 Top-k 里。
- `MRR`：第一个命中的正确文献排得越靠前，分数越高。
- `Latency`：一个问题从发出到拿到答案的时间。

**评测流程**

- 准备一个 `dev_qa.jsonl`，每行包含一个问题 `q` 和对应的标准答案文献 `gold_ids`。
- 对每个模式跑一遍，统计指标，输出到 `logs/metrics.csv`。

---

### 7）测试与验收（简单版理解）

**单元测试**

- 针对单一模块的“小快灵测试”，例如：
  - `test_ingest.py`：检查输出字段是否完整，有没有把摘要完全删空。
  - `test_index_bm25.py`：能否成功建索引并返回有序结果。
  - `test_rag.py`：`build_context` 是否有分隔符、是否做了截断等。

**集成 / 冒烟测试**

- 用一个很小的样本（如 `tests/data/mini.jsonl`），走一遍完整流程：
  - 预处理 → 建索引 → 检索 → RAG 回答；
  - 确认不会报错，并且能输出看起来合理的答案和 citations。

---

## 调用示例（命令行）

```bash
# 0）安装依赖（示例，按需调整）
pip install -U rank-bm25 faiss-cpu chromadb sentence-transformers pyyaml tqdm

# 1）数据清洗
python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl

# 2）构建索引
python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-zh-v1.5

# 3）交互查询（示例）
python -c "from src.rag import answer; print(answer('What is contrastive learning?', 'hybrid', 5))"

# 4）评估（示例）
python src/eval.py --qa ./data/dev_qa.jsonl --modes bm25 dense hybrid --out ./logs/metrics.csv
```

---

## 配置与环境变量

示例 `configs/config.yaml`：

```yaml
embedding:
  model: bge-small-zh-v1.5
  batch_size: 64
  device: auto   # cpu|cuda|mps|auto

index:
  dense_backend: chroma   # chroma|faiss
  chroma_path: ./data/chroma
  bm25_path: ./data/bm25.idx

retrieve:
  mode: hybrid            # bm25|dense|hybrid
  topk: 5
  alpha: 0.5              # hybrid 融合权重

generation:
  provider: openai_like   # 本地或 OpenAI 兼容
  model: qwen2.5-7b-instruct
  max_tokens: 512
```

如使用 OpenAI 兼容 API，可通过环境变量配置：

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

本地推理则按对应服务要求配置。

---

## 评估与指标（略带公式）

评测集格式示例（`data/dev_qa.jsonl`）：

```json
{"q": "What is contrastive learning?", "gold_ids": ["2002.05709", "2010.00001"], "k": 5}
```

- **Recall@k**：`= 命中 gold 的数量 / gold 的总数`
- **MRR**：对每个问题，找到第一个命中 gold 的排名 `r`，用 `1/r` 做分数，然后取平均。
- **P@k**：Top-k 内命中的比例（可选）。
- **Latency(ms)**：检索 / 生成 / 总耗时的均值、95 分位等。

输出文件：

- `logs/metrics.csv`，包含字段：
  - `mode`
  - `recall@k`
  - `mrr`
  - `p@k`
  - `t_retrieve_ms`
  - `t_generate_ms`
  - `t_total_ms`

---

## 数据 I/O 结构（再看一次总览）

```text
data/
├── arxiv-metadata-oai-snapshot.json      # 原始
├── clean.jsonl                           # 预处理输出
├── bm25.idx                              # BM25 索引
└── chroma/                               # 稠密向量库
logs/
└── metrics.csv                           # 评估结果
```

---

## 术语解释（小白友好版）

- **arXiv**：一个公开的学术论文预印本网站，可以理解成“全球研究人员上传论文的公共仓库”。
- **摘要（abstract）**：论文的浓缩版简介，通常几百字，告诉你这篇论文研究了什么、怎么做的、结论大概是什么。
- **元数据（metadata）**：描述论文的“数据的说明书”，比如标题、作者、发表日期、所属领域等，不包含具体正文内容。
- **JSON / JSONL（JSON Lines）**：
  - JSON：一种常见的数据表示格式，看起来很像“带花括号的字典”。
  - JSONL：每一行都是一个独立 JSON 对象的大文件，适合流式处理大规模数据。
- **RAG（Retrieval-Augmented Generation，检索增强生成）**：
  - 先“检索”外部资料，再让大模型“根据这些资料回答问题”的一整套方法。
  - 这样可以大幅减少模型“瞎编”的概率，并且可以随数据更新。
- **检索（retrieval）**：在一堆文档里找出和问题最相关的那一小部分，就像在图书馆 / 搜索引擎里搜关键词。
- **索引（index）**：为了加速检索而提前建好的“目录结构”，有了索引之后，就不用每次从头翻完整个库。
- **倒排索引 / BM25**：
  - 倒排索引：记录“某个词出现在哪些文档里”。
  - BM25：一种基于关键词匹配的经典检索算法，是搜索引擎的老牌基础工具。
- **向量 / 向量嵌入（embedding）**：
  - 把一句话或一段文字转换成一串数字（向量），使得“意思相近的句子”在这个数字空间里彼此更接近。
- **稠密检索（dense retrieval）**：
  - 先把所有文档编码成向量，再把问题也编码成向量，最后“找距离最近的向量”。
  - 优点是能识别语义相近但用词不同的表达。
- **向量数据库（Vector DB）**：专门存储这些向量并支持“相似度搜索”的数据库，比如 FAISS、Chroma、Milvus 等。
- **混合检索（hybrid retrieval）**：
  - 同时使用 BM25（关键词）和稠密检索（语义），再把两边的得分融合在一起，取长补短。
- **Top-k**：取“得分最高的前 k 个结果”，例如 Top-5 就是只保留评分最高的 5 篇文档。
- **召回率（Recall / Recall@k）**：
  - 在“真正应该被找到的答案”中，有多少被系统成功找了出来。
  - 举例：本来有 2 篇论文算“正确答案”，Top-5 结果里命中了 1 篇，那 Recall@5 = 1/2。
- **精确率 / 准确率（Precision / P@k）**：
  - 系统返回的 Top-k 结果里，有多少是真正正确的。
- **MRR（Mean Reciprocal Rank，平均倒数排名）**：
  - 关注“第一个正确答案排在第几位”，排得越靠前，分数越高。
- **延迟（Latency）**：
  - 用户发出一个问题，到系统返回答案之间的时间。
  - 可以拆分成检索耗时、生成耗时和端到端总耗时。
- **上下文（context）**：
  - 在这里指“我们提前喂给大模型的相关资料”，包括若干论文的标题和摘要。
- **大语言模型（LLM）**：
  - 类似 ChatGPT、Qwen、LLaMA 这类可以理解和生成自然语言的大模型。
- **API**：
  - 一种“程序和程序之间通话的标准方式”，你写的 Python 代码通过 API 把问题发给大模型服务，再收到返回的答案。
- **评测集 / `dev_qa.jsonl`**：
  - 一小份“已知问题 + 标准参考文献 id”的数据，用来客观比较不同检索策略的好坏，而不是只靠主观感受。

---

## 疑问：大模型 API 在哪一步被调用？

**问题 1：我们调用大模型 API 的步骤实现在哪里？**

**回答（简明版）：**

- 代码设计上，这一步放在 `src/rag.py` 的 `answer` 函数中完成。
- 在这之前，`answer` 会先调用 `retriever.retrieve(...)` 找文档，再调用 `build_context(...)` 拼上下文。
- 然后，`answer` 会把“用户问题 + 上下文 + 提示语”组装成一个完整的请求，调用你配置的大模型 API（例如 OpenAI 兼容接口、本地 Qwen/LLaMA 服务等）。

**回答（稍微技术一点）：**

- 具体使用哪个大模型，由 `configs/config.yaml` 里的 `generation` 配置和环境变量（如 `OPENAI_API_KEY`, `OPENAI_BASE_URL`）决定。
- 实现代码时，你会在 `answer` 函数内部（或它调用的一个小辅助函数中）使用相应的 SDK / HTTP 客户端：
  - 把 `query` 和由检索得到的上下文一起发给模型；
  - 从返回结果里取出模型生成的文本；
  - 再把这个文本和检索时返回的文献元信息拼成：

    ```json
    {
      "answer": "...",
      "citations": [
        {"id": "...", "title": "..."},
        ...
      ]
    }
    ```

从“项目流程”的角度记住一句话：

> **大模型只在最后一步被调用，且它拿到的是“已经检索并整理好的资料”，不会“凭空胡编”。**
```
## 项目总览（小白友好版）

本项目的目标：做一个“懂论文的问答小助手”。

你给它一个问题，比如“什么是对比学习（contrastive learning）？”，它会：

- 先在一大堆论文摘要里“查资料”；
- 再把查到的摘要丢给一个大语言模型，让模型读完之后，用自然语言给你解释；
- 最后告诉你：答案主要参考了哪几篇论文（给出 id 和标题）。

### 用生活类比快速理解

可以把整个系统想象成一座图书馆：

- **arXiv 数据** = 图书馆里所有书的“目录卡片”（标题 + 摘要）。
- **索引（BM25 / 向量库）** = 图书馆的“检索系统”，帮你快速从几百万条记录里找到可能相关的几十条。
- **检索模块 `retriever`** = 帮你用搜索系统下指令的“图书馆管理员”。
- **大语言模型（LLM）** = 会读论文、会总结的“专业助教”。
- **RAG** = 把“查资料”和“请助教回答”这两步连在一起的完整流程。

### 一句话看懂整体流程

从你提问到拿到答案，大致分成 6 步：

1. 准备原始数据：一份很大的 arXiv 元数据文件，每一行是一篇论文的信息（包含标题和摘要）。
2. 用 `src/ingest.py` 把大文件“瘦身”，只留下后续真正需要的字段，生成体积更小的 `clean.jsonl`。
3. 用 `src/index_bm25.py` 和 `src/index_dense.py` 对这些摘要建立两套检索索引（BM25 和稠密向量）。
4. 当有人提问时，用 `src/retriever.py` 读这两套索引，找出最可能相关的若干篇论文。
5. 用 `src/rag.py` 把这些论文摘要拼成一个“上下文”，再连同你的问题一起发送给大语言模型 API。
6. 大模型读完上下文，写出答案，并附上它参考到的论文 `id/title`，最后把结果返回给你。

如果下面有你看不懂的名词，可以先继续往下看，文末有一个《术语解释（小白友好版）》专门做词汇说明。

### 数据是如何一步步流动的？

先看“数据文件”层面的流向：

原始数据：arxiv-metadata-oai-snapshot.json  （几 GB 的大文件）
        │  经由 src/ingest.py 清洗精简
        ▼
中间数据：clean.jsonl                         （只保留 id/title/abstract 等核心字段）
        │  经由 src/index_bm25.py / src/index_dense.py 建索引
        ▼
索引数据：bm25.idx  和  chroma/              （两套检索索引）
        │  被 src/retriever.py 和 src/rag.py 使用
        ▼
最终输出：answer 文本 + 引用的论文 id/title

## prompt
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
- 

我需要你按照给我完成以下项目需求文档，按照精简的来，不需要过于复杂和健壮

## 项目需求

- **目标**: 基于 arXiv 摘要构建面向学术领域的 RAG 问答系统；同时支持稠密检索与 BM25，进行效果与性能对比。
- **范围**: 只使用摘要与元数据，不做 PDF 全文解析；不微调大模型，仅集成现成开源模型（如 Qwen2.5-7B-Instruct、LLaMA-3-8B-Instruct 或本地/云端 API）。
- **输出**: 答案文本 + 引用的文档 `id/title/categories`（可返回 Top-k 命中文档）。
- **对比维度**:
  - 准确率/召回率（小规模标注或自动生成评测集）
  - 延迟（检索耗时、生成耗时、端到端耗时）
  - 资源占用（索引大小、内存）
- **非目标**: 不涉及复杂 UI、长流程编排、在线多用户管理。

## 输入数据

#### 数据路径
"""
./data/arxiv-metadata-oai-snapshot.json
"""

该文件为 arXiv 元数据快照，体量约 4.6GB，采用 JSON Lines（每行一个 JSON 对象）格式，UTF-8 编码。

#### 数据示例

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

> 说明：实际文件中每一行即为一个类似上面的完整 JSON 对象。

#### 数据结构

- **id (string)**: arXiv 唯一标识，如 `0704.0001` 或 `arXiv:YYMM.NNNNN`。
- **submitter (string)**: 提交人姓名。
- **authors (string)**: 作者字符串，逗号分隔。
- **title (string)**: 论文标题。
- **comments (string|null)**: 作者备注，如页数、图表数等。
- **journal-ref (string|null)**: 期刊引用信息。
- **doi (string|null)**: DOI 标识。
- **report-no (string|null)**: 报告编号，部分条目存在。
- **categories (string)**: 用空格分隔的学科标签，如 `cs.CL stat.ML`。
- **license (string|null)**: 许可证信息，部分条目存在。
- **abstract (string)**: 论文摘要，可能包含 LaTeX 语法或数学公式。
- **versions (array<object>)**: 版本列表；典型字段：
  - **version (string)**: 如 `v1`、`v2`。
  - **created (string)**: 提交时间（RFC 822 文本）。
  - **source (string)**: 来源标识（如 `arXiv`）。
- **update_date (string|null)**: 最近更新时间（`YYYY-MM-DD`）。
- **authors_parsed (array<array<string>>)**: 解析后的作者姓名列表，典型为 `[lastname, firstname, suffix]`。

补充说明：
- `categories` 为空格分隔，可将首个标签作为主学科（如 `cs.CL`）。
- `abstract` 可能包含换行与 LaTeX 标记，检索前建议做清洗与标准化。

#### 读取与处理建议（大文件）

- 按行流式读取，避免一次性加载至内存；必要时仅抽取所需字段。
- Apple Silicon/本机解析建议使用标准库逐行 `json.loads` 或使用 `ijson` 流式解析。
- 样例代码：

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

常见派生字段：
- **primary_category**: `categories.split()[0]`；
- **year**: 由 `versions[0].created` 解析年份；
- **author_list**: `authors.split(", ")` 或由 `authors_parsed` 组装结构化作者列表。


## pipeline

为便于落地，按阶段列出每个操作、对应代码文件与函数、I/O 与示例命令。

### 阶段0：准备
- 操作：安装依赖、准备配置
  - 代码：`requirements.txt`、`configs/config.yaml`
  - 命令：
    ```bash
    pip install -U -r requirements.txt
    ```

### 阶段1：数据预处理（清洗精简）
- 操作：流式读取原始 JSONL，抽取与清洗核心字段
  - 代码：`src/ingest.py: stream_clean_arxiv`
  - 输入：`data/arxiv-metadata-oai-snapshot.json`
  - 输出：`data/clean.jsonl`，字段 `id/title/abstract/categories/created`
  - 示例命令：
    ```bash
    python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl
    ```

### 阶段2：索引构建（双通路）
- 操作2.1：BM25 倒排索引构建
  - 代码：`src/index_bm25.py: build_bm25_index`
  - 输入：`data/clean.jsonl`
  - 输出：`data/bm25.idx`
  - 示例命令：
    ```bash
    python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
    ```
- 操作2.2：稠密向量库构建（Chroma/FAISS）
  - 代码：`src/index_dense.py: embed_and_build_vector_db`
  - 输入：`data/clean.jsonl`
  - 输出：`data/chroma/` 或 FAISS 索引文件
  - 示例命令：
    ```bash
    python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-zh-v1.5
    ```

### 阶段3：检索（单路/混合）
- 操作3.1：仅 BM25 检索
  - 代码：`src/retriever.py: retrieve`（`mode="bm25"`）
  - 依赖：`src/index_bm25.py: search_bm25`
  - 输入：用户查询字符串、`data/bm25.idx`
  - 输出：命中文档列表 `[Doc{id,title,score,...}]`
- 操作3.2：仅稠密检索
  - 代码：`src/retriever.py: retrieve`（`mode="dense"`）
  - 依赖：`src/index_dense.py: search_dense`
  - 输入：用户查询字符串、`data/chroma/`、嵌入模型名
  - 输出：命中文档列表 `[Doc{id,title,score,...}]`
- 操作3.3：混合检索（分数归一化+加权融合）
  - 代码：`src/retriever.py: retrieve`（`mode="hybrid"`, `alpha=0.5` 可调）
  - 依赖：3.1 与 3.2 两路结果
  - 输出：融合后的 Top-k 文档列表
  - 示例（Python 一行）：
    ```bash
    python -c "from src.retriever import retrieve;print(retrieve('contrastive learning', 5, 'hybrid', 0.5))"
    ```

### 阶段4：上下文构建与生成回答
- 操作4.1：拼接上下文（标题+摘要，分隔符，长度裁剪）
  - 代码：`src/rag.py: build_context`
  - 输入：检索出的文档列表、`max_chars`
  - 输出：用于 LLM 的上下文字符串
- 操作4.2：调用 LLM 生成答案并附引用
  - 代码：`src/rag.py: answer`
  - 输入：`query`、检索模式 `mode`、`topk`
  - 输出：`{"answer": str, "citations": [{"id","title"}...]}`
  - 示例（Python 一行）：
    ```bash
    python -c "from src.rag import answer; print(answer('What is contrastive learning?', 'hybrid', 5))"
    ```

### 阶段5：离线评估
- 操作：对 bm25/dense/hybrid 做召回与延迟对比
  - 代码：`src/eval.py: evaluate`
  - 输入：`data/dev_qa.jsonl`（字段：`q`/`gold_ids`）
  - 输出：`logs/metrics.csv`（Recall@k/MRR/Latency）
  - 示例命令：
    ```bash
    python src/eval.py --qa ./data/dev_qa.jsonl --modes bm25 dense hybrid --out ./logs/metrics.csv
    ```

### 阶段6：测试与验收（自动化 + 冒烟）
- 操作6.1：单元测试（函数级）
  - 覆盖：`ingest` 字段与清洗、`index_bm25/index_dense` 构建与查询、`retriever` 融合与去重、`rag` 上下文拼接、`eval` 指标计算。
  - 代码：`tests/test_*.py`（见“测试与验收”章节详细说明）
  - 命令：
    ```bash
    pytest -q
    ```
- 操作6.2：集成/冒烟测试（小样本端到端）
  - 数据：`tests/data/mini.jsonl`（5~50 行）
  - 预期：能成功构建 BM25/向量库、检索返回 Top-k、`rag.answer` 返回非空答案且包含 citations。
  - 命令：
    ```bash
    python src/index_bm25.py --in tests/data/mini.jsonl --index ./data/bm25.idx
    python src/index_dense.py --in tests/data/mini.jsonl --db ./data/chroma --model bge-small-zh-v1.5
    python -c "from src.rag import answer;print(answer('what is contrastive learning?', 'hybrid', 3))"
    ```

## 项目结构与模块关系

```
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
│   ├── conftest.py                           # 公共 fixture（如 mini 数据）
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

- 模块关系：`ingest` → `index_bm25`/`index_dense` → `retriever` → `rag` → `eval`（评估在侧）

## 文件描述

### src/ingest.py
编写思路：
- 目标：以最小内存占用流式读取大 JSONL，抽取核心字段并做轻量清洗。
- 字段：`id/title/abstract/categories/created(来自versions[0])`。
- 清洗：去除摘要中的多余换行、简单移除 LaTeX 包围符（尽量无损）。
- 产物：`clean.jsonl`（一行一个 JSON 对象）。

导出函数：
- `stream_clean_arxiv(input_path: str, output_path: str) -> None`

使用示例：
```bash
python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl
```

### src/index_bm25.py
编写思路：
- 目标：快速落地的基线检索；无需外部服务。
- 语料：`title + "\n" + abstract`；保留 `id/title/categories` 以便返回引用。
- 持久化：序列化 `corpus/id_map/bm25_config` 到 `data/bm25.idx`。

导出函数：
- `build_bm25_index(clean_path: str, index_path: str) -> None`
- `search_bm25(index_path: str, query: str, topk: int) -> List[Tuple[id, score]]]`

使用示例：
```bash
python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
```

### src/index_dense.py
编写思路：
- 目标：更强召回；支持 `FAISS` 或 `Chroma` 后端（默认 Chroma 本地目录）。
- 嵌入：选 `bge-small-zh-v1.5` 或 `e5-base` 之类轻量模型；批量编码、显示进度。
- 存储：向量 + 元数据（`id/title/categories`）。

导出函数：
- `embed_and_build_vector_db(clean_path: str, db_path: str, model_name: str) -> None`
- `search_dense(db_path: str, query: str, topk: int, model_name: str) -> List[Tuple[id, score]]`

使用示例：
```bash
python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-zh-v1.5
```

### src/retriever.py
编写思路：
- 目标：统一入口，屏蔽 BM25/稠密/混合 差异。
- 混合：分数归一化（min-max 或 z-score）后加权合并，提供 `alpha` 权重。

导出函数：
- `retrieve(query: str, topk: int, mode: str = "hybrid", alpha: float = 0.5) -> List[Doc]`
  - `Doc = {"id": str, "title": str, "score": float, "source": "bm25|dense", ...}`

### src/rag.py
编写思路：
- 目标：将检索到的摘要构造成上下文，调用 LLM 生成答案并返回引用。
- 截断：按字符/Token 上限截断，优先保留高分文档；可加去重与段间分隔符。
- LLM：可接本地或远程接口（如 Qwen/Llama/OpenAI 兼容 API）。

导出函数：
- `build_context(docs: List[Doc], max_chars: int = 4000) -> str`
- `answer(query: str, mode: str = "hybrid", topk: int = 5) -> Dict`

### src/eval.py
编写思路：
- 目标：离线对比多策略（bm25/dense/hybrid）的召回与端到端延迟。
- 指标：Recall@k、MRR、P@k、平均/95P 延迟等。

导出函数：
- `evaluate(qa_path: str, modes=("bm25","dense","hybrid")) -> MetricsFrame`

## 实现方式（细化到步骤/参数/边界）

本章节按“阶段化模板”组织，不删减既有内容，仅增加结构化小节：
- 概述（编写思路）
- 做什么与为什么（新手可读）
- 概念入门
- I/O 与参数
- 边界与异常
- 验证与资源预估 / 观测与性能
- 伪代码
- 对应测试（文件、断言、命令、伪代码）

### 1) 数据预处理（src/ingest.py）
#### 概述
编写思路：
- 流式读取，逐行 `json.loads`；异常行写 warn 并跳过；统计读取/跳过数量。
- 摘要清洗：合并多行、移除成对 `$...$` 与 `\(...\)` 的 LaTeX 片段（保留非数学文本）。
- 解析 `created`: 优先 `versions[0].created`，解析失败置 `null` 并记录。

#### 做什么与为什么
这一阶段在做什么（新手向）：
- 把一个巨大的“每行一个 JSON 的大文件”，变成一个“只保留我们需要字段的小文件”，方便后续更快处理。

为什么需要这一步：
- 原始文件太大、字段太多，直接建索引会慢、占用内存高；先精简可以“提速+省内存”。

#### 概念入门
核心概念入门：
- JSON Lines：每一行都是一个独立 JSON；处理时“逐行读→逐行解析”。
- 清洗：去掉无关字符（如换行、数学公式），让后面的检索/建模更稳定。

#### 常见坑
常见坑：
- 行里有坏 JSON（少括号、编码问题）会解析失败：不要整个报错终止，记录行号并跳过。
- 摘要可能很长且包含 LaTeX，简单删除数学公式时要“尽量少删正文”。

#### 验证与资源预估
如何快速验证：
- 抽样打印前 3 行，检查字段是否存在、是否被意外删空。
- 统计“成功/失败/跳过条数”，确认与预期一致。

时间与资源预估：
- 纯 I/O + 轻解析，4.6GB 量级一般“几分钟到十几分钟”，与磁盘读速有关。

#### I/O 与参数
CLI 参数（示例）：
- `--in` 输入 JSONL 路径
- `--out` 输出 `clean.jsonl`
- `--max_rows` 仅处理前 N 行（调试用）

I/O 约定：
- 输入：原始 JSONL（大文件）
- 输出：每行 `{"id","title","abstract","categories","created"}`

#### 边界与异常
边界与异常：
- 空摘要/标题：按空字符串写出；categories 为空则写 `""`。
- 非法 JSON 行：记录行号与片段，继续。

#### 观测与性能
性能与可观测性：
- 每处理 10000 行打点一次日志；最终输出用时与吞吐（行/秒）。

#### 伪代码
伪代码：
```python
def stream_clean_arxiv(input_path, output_path, max_rows=None):
    # 打开输入/输出文件，逐行读取与解析
    # 提取/清洗字段，异常 try/except 捕获并 warn
    # 定期 flush，记录进度
    pass
```

#### 对应测试
对应测试（tests/test_ingest.py）：
- 目的：功能刚实现后，验证输出字段完整、清洗生效、异常行被跳过。
- 命令：`pytest tests/test_ingest.py -q`
- 伪代码：
```python
# tests/test_ingest.py
def test_stream_clean_arxiv_fields(tmp_path):
    # 准备 mini 输入：2~3 行正常 + 1 行坏 JSON
    # 调用 stream_clean_arxiv(in_path, out_path)
    # 读取 out_path 首行，断言包含 id/title/abstract/categories/created
    # 断言 abstract 无多余换行或成对公式残留
    # 验证成功条数≥1、坏行被跳过（可用日志或计数返回）
    pass
```

### 2) BM25 索引（src/index_bm25.py）
#### 概述
编写思路：
- 读取 `clean.jsonl`，构造 `corpus = [title + "\n" + abstract]`。
- 使用 `rank_bm25.BM25Okapi`；保存 `corpus`、`id_map`、BM25 必要参数。

#### 概念入门
BM25 是什么（新手向）：
- 一种“关键词匹配”的传统检索方法。简单说：
  - 出现越多次的词更重要（TF），在全库里越少见的词更关键（IDF）。
  - BM25 根据句子里关键词的覆盖程度给分，适合“问题里和文档里词面重叠比较高”的场景。

#### 做什么与为什么
为什么要用 BM25：
- 快、稳定、无需模型就能跑，是很好的“基线方案”和“容灾回退”。

优缺点（直观）：
- 优点：速度快、可解释；缺点：不懂同义词/语义相近（“车”和“汽车”可能匹配不到）。

#### I/O 与参数
CLI 参数：
- `--in` clean 文件路径
- `--index` 序列化输出路径

I/O：
- 输入：`clean.jsonl`
- 输出：`bm25.idx`（可用 `pickle` 或 `joblib`）

#### 边界与异常
边界与异常：
- 过滤极短文档（如 < 10 字符），减少噪声；记录过滤数。

#### 验证
如何快速验证：
- 用两个简单查询测试：一个出现高频词的（如“learning”），一个专业词（如“contrastive”），看是否能返回合理标题。

#### 伪代码
伪代码：
```python
def build_bm25_index(clean_path, index_path):
    # 加载 clean.jsonl → corpus, id_map
    # 构建 BM25Okapi(corpus_tokens)
    # dump 到 index_path
    pass

def search_bm25(index_path, query, topk):
    # 加载索引与映射 → 计算 scores → 取 topk → 返回文档元信息
    pass
```

#### 对应测试
对应测试（tests/test_index_bm25.py）：
- 目的：索引可构建、查询能返回有序的 Top-k。
- 命令：`pytest tests/test_index_bm25.py -q`
- 伪代码：
```python
# tests/test_index_bm25.py
def test_build_and_search_bm25(tmp_path):
    # 使用 tests/data/mini.jsonl （或临时生成 clean.jsonl）
    # 调 build_bm25_index → 断言 index 文件存在/非空
    # 调 search_bm25('contrastive', topk=3)
    # 断言 len==3，id 唯一，score 按降序排列，返回含 id/title
    pass
```

### 3) 稠密向量库（src/index_dense.py）
#### 概述
编写思路：
- 选择轻量嵌入模型（如 `bge-small-zh-v1.5`），批量编码；支持 `--batch_size`、`--device`。
- 默认 Chroma 本地目录；可选 FAISS 索引保存。

#### 概念入门
什么是“稠密嵌入”（新手向）：
- 把一句话编码成一个“向量”（一串数字），语义相近的句子向量距离更近。
- 检索时“把问题也编码成向量”，再“找向量最接近的文档”。

为什么要加稠密检索：
- 它能“理解语义相似”，即使没有关键词完全重合也能找对内容（比如“对象对比学习”和“对比表征学习”）。

向量库（Vector DB）是啥：
- 专门管理“向量 + 元信息（id、标题等）”的数据库，支持“相似度搜索”。常见：FAISS、Chroma、Milvus。

取舍建议：
- 本机入门：Chroma（简单好用）或 FAISS（速度快、轻量）。云端/大规模：Milvus、Weaviate 等。
- 设备：`cpu` 最通用；Mac 可用 `mps`；有 NVIDIA 则 `cuda` 更快。

#### I/O 与参数
CLI 参数：
- `--in` clean 文件路径
- `--db` 向量库目录
- `--model` 嵌入模型名
- `--batch_size` `--device`（cpu/cuda/mps）

I/O：
- 输入：`clean.jsonl`
- 输出：`data/chroma/`（含向量与 metadata）

#### 边界与异常
边界与异常：
- 编码失败的样本跳过并记录；长文本可截断到 `max_chars`。

#### 验证
如何快速验证：
- 选 10 条文档手工记下标题，问一个“同义表述”的问题，看看是否能检回相应文档。

#### 伪代码
伪代码：
```python
def embed_and_build_vector_db(clean_path, db_path, model_name, batch_size=64):
    # 逐批生成向量 → 写入 Chroma/FAISS（含 id/title/categories）
    pass

def search_dense(db_path, query, topk, model_name):
    # 对 query 编码 → 检索 topk → 返回 (doc, score)
    pass
```

#### 对应测试
对应测试（tests/test_index_dense.py）：
- 目的：能编码写库且可查询；接口返回结构完整。
- 命令：`pytest tests/test_index_dense.py -q`
- 伪代码：
```python
# tests/test_index_dense.py
def test_dense_index_and_search(tmp_path):
    # 使用 mini.jsonl 构建 data/chroma/（或临时目录）
    # 调 embed_and_build_vector_db(..., model_name='bge-small-zh-v1.5')
    # 调 search_dense(query='graph neural network', topk=3, model_name=同上)
    # 断言 len==3，含 id/title/score，score 为 float
    pass
```

### 4) 统一检索（src/retriever.py）
#### 概述
编写思路：
- `mode= bm25 | dense | hybrid`；hybrid 采用 min-max 归一化与 `alpha` 加权。
- 去重：按 `id` 去重，只保留最高分一次。

#### 概念入门 / 直觉
hybrid（混合）直觉解释：
- BM25 擅长“词面匹配”，稠密擅长“语义匹配”。
- 先各自算分，再把分数做“0~1 归一化”，用权重 `alpha` 融合：
  - `final_score = alpha * dense_score + (1-alpha) * bm25_score`
  - `alpha` 越大越信稠密，越小越信 BM25。

为什么要归一化：
- 两种分数“量纲不同”（一个可能 0~10、一个 0~1），直接相加会被某一方“支配”，归一化后更公平。

#### 边界与异常
常见坑：
- 同一个 `id` 从两路来时要去重；否则 Top-k 里会重复同一篇文档。

#### 伪代码（签名）
签名：
```python
def retrieve(query: str, topk: int = 5, mode: str = "hybrid", alpha: float = 0.5) -> list:
    # 调用各后端并融合，返回标准化 Doc 列表
    pass
```

#### 对应测试
对应测试（tests/test_retriever.py）：
- 目的：三种模式行为正确；hybrid 归一化与去重生效。
- 命令：`pytest tests/test_retriever.py -q`
- 伪代码：
```python
# tests/test_retriever.py
def test_hybrid_merge_and_deduplicate(monkeypatch):
    # monkeypatch search_bm25/search_dense 返回可控结果（含重复 id）
    # 调 retrieve('contrastive', topk=3, mode='hybrid', alpha=0.5)
    # 断言：id 唯一、len==3、0<=score<=1、按降序
    pass
```

### 5) RAG 组装与生成（src/rag.py）
#### 概述
编写思路：
- 上下文构造：标题 + 摘要；分隔符 `\n\n---\n\n`；限制 `max_chars`。
- 指令模板：要求模型基于上下文作答并给出引用 `id/title`。
- LLM 抽象：可对接本地或 OpenAI 兼容 API（通过 `configs/config.yaml` 指定）。

#### 做什么与为什么
这一阶段在做什么：
- 把“检索来的内容”拼成上下文，连同“问题”一起喂给大模型，让模型“基于上下文作答”（降低幻觉）。

为什么要做上下文截断：
- 模型有“上下文长度限制”（能看多少字/Token），超过就会被截断；应优先保留高分、与问题更相关的段落。

#### 概念与提示词要点
提示词（Prompt）要点：
- 明确要求“只依据提供的上下文回答”，并“在结尾给出引用 id/title”。
- 不要把太多无关文本塞进去，避免“稀释关键信息”。

#### 边界与异常
常见坑：
- 上下文重复或无关内容过多，模型会“迷路”；要控制 Top-k 与上下文长度。

#### 伪代码（签名）
签名：
```python
def build_context(docs, max_chars=4000) -> str:
    # 按得分降序拼接，超限截断
    pass

def answer(query: str, mode: str = "hybrid", topk: int = 5) -> dict:
    # retrieve → build_context → 调 LLM → 返回 answer + citations
    pass
```

#### 对应测试
对应测试（tests/test_rag.py）：
- 目的：上下文截断/分隔符正确；`answer` 返回结构完整（可 stub LLM）。
- 命令：`pytest tests/test_rag.py -q`
- 伪代码：
```python
# tests/test_rag.py
def test_build_context_truncation():
    # 构造若干 doc（title/abstract/score），设定 max_chars 很小
    # 调 build_context → 断言包含分隔符与截断生效
    pass

def test_answer_structure(monkeypatch):
    # monkeypatch retrieve 返回固定 docs
    # monkeypatch LLM 调用为固定字符串
    # 调 answer → 断言含 'answer' 字段与 citations（id/title）
    pass
```

### 6) 评测（src/eval.py）
#### 概述
编写思路：
- 评测集：`dev_qa.jsonl`，字段 `{"q": str, "gold_ids": [str], "k": int?}`。
- 指标：`Recall@k = |hit ∩ gold| / |gold|`，`MRR = 1/rank_first_hit`。
- 延迟：分别记录检索/生成/总耗时，输出 csv。

#### 概念入门
这些指标是什么意思（新手向）：
- Recall@k（在前 k 条里，有没有把“正确答案”检回来）：
  - 举例：gold_ids = [A, B]；如果检索 Top-5 里有 A（或 B），Recall@5=1；都没有则为 0。
- MRR（Mean Reciprocal Rank，平均倒数排名）：
  - 先找“第一个命中的正确 id”在结果里的排名 r，然后取 1/r；比如第 1 个命中则得分 1.0，第 5 个命中得分 0.2。对所有问题求平均。
- Latency（延迟）：
  - 检索耗时（建好索引后每次查询的时间）、生成耗时（LLM 推理时间）、总耗时（端到端）。

#### 为什么
为什么选择这些指标：
- Recall@k 直观衡量“能否把对的文档捞上来”；MRR 关注“第一个对的出现得有多靠前”；延迟反映“用户等待多久”。

#### 验证
如何快速验证：
- 先只评检索（不连模型），看 Recall@k/MRR；再加上 LLM，观察总耗时与答案质量的变化。

#### 伪代码（签名）
签名：
```python
def evaluate(qa_path: str, modes=("bm25","dense","hybrid")):
    # 遍历问答对 → 三种模式 → 统计与落盘
    pass
```

#### 对应测试
对应测试（tests/test_eval.py）：
- 目的：指标计算正确、CSV 输出存在且列齐全。
- 命令：`pytest tests/test_eval.py -q`
- 伪代码：
```python
# tests/test_eval.py
def test_metrics_and_csv(tmp_path, monkeypatch):
    # 构造 mini qa（1~3 条），gold_ids 指向可控文档
    # monkeypatch 检索/答案以便可控命中
    # 调 evaluate(..., modes=('bm25','dense')) 输出到 tmp_path/metrics.csv
    # 读取 CSV：断言存在列 recall@k/mrr/t_total_ms 等，数值在合理范围
    pass
```

## 调用示例

```bash
# 0) 安装依赖（示例，按需调整）
pip install -U rank-bm25 faiss-cpu chromadb sentence-transformers pyyaml tqdm

# 1) 数据清洗
python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl

# 2) 构建索引
python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-zh-v1.5

# 3) 交互查询（示例）
python -c "from src.rag import answer; print(answer('What is contrastive learning?', 'hybrid', 5))"

# 4) 评估（示例）
python src/eval.py --qa ./data/dev_qa.jsonl --modes bm25 dense hybrid --out ./logs/metrics.csv
```


## 配置与环境变量

- `configs/config.yaml`（建议示例）：

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

- 环境变量（如使用 OpenAI 兼容 API）：
  - `OPENAI_API_KEY`、`OPENAI_BASE_URL`
  - 或本地推理无需密钥，改用本地端点。

## 评估与指标

- 评测集格式（示例 `data/dev_qa.jsonl`）：
```json
{"q": "What is contrastive learning?", "gold_ids": ["2002.05709", "2010.00001"], "k": 5}
```

- 指标定义：
  - **Recall@k**: 命中任一 gold 的比例；多 gold 则以集合交求比。
  - **MRR**: 首个命中 gold 的倒数排名平均值。
  - **P@k**: Top-k 内命中的比例（可选）。
  - **Latency(ms)**: 检索/生成/总耗时的均值、95 分位。

- 输出：`logs/metrics.csv`，包含 `mode, recall@k, mrr, p@k, t_retrieve_ms, t_generate_ms, t_total_ms`。

## 数据 I/O 结构
```
data/
├── arxiv-metadata-oai-snapshot.json      # 原始
├── clean.jsonl                           # 预处理输出
├── bm25.idx                              # BM25 索引
└── chroma/                               # 稠密向量库
logs/
└── metrics.csv                           # 评估结果
```

## 疑问：
问题1:我们调用大模型api的步骤实现在哪里
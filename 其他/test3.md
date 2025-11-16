下面这套流程是**从零开始一步步测试所有核心代码文件**的操作说明，新手照着敲命令就能跑通、确认每个模块是“能用的”。

假设你的项目根目录是 [5020-RAG/](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG:0:0-0:0)。

---

# 0. 环境准备

## 步骤 0-1：创建并激活虚拟环境（可选但推荐）

```bash
cd 5020-RAG

python -m venv .venv
source .venv/bin/activate    # Windows 用 .venv\Scripts\activate
```

## 步骤 0-2：安装依赖

保证 `requirements.txt` 已按文档写好，然后：

```bash
pip install -U pip
pip install -U -r requirements.txt
```

---

# 1. 准备一个“小样本数据集”用来测试

你可以先用**自己写一个小 JSONL 文件**来测试整个 pipeline，而不需要一上来就处理 4.6GB 大文件。

## 步骤 1-1：创建最小原始数据 `data/mini_raw.jsonl`

在 `data/` 目录下，新建文件 `mini_raw.jsonl`，内容类似（每行一个 JSON 对象）：

```json
{"id": "mini-0001", "title": "An Introduction to Contrastive Learning", "abstract": "This paper gives a basic introduction to contrastive learning methods.", "categories": "cs.LG", "versions": [{"created": "Mon, 1 Jan 2020 00:00:00 GMT"}]}
{"id": "mini-0002", "title": "Graph Neural Networks for Text", "abstract": "We apply graph neural networks to text classification tasks.", "categories": "cs.CL", "versions": [{"created": "Mon, 2 Jan 2020 00:00:00 GMT"}]}
{"id": "mini-0003", "title": "A Survey on Transformers", "abstract": "This survey summarises transformer architectures in NLP.", "categories": "cs.CL", "versions": [{"created": "Mon, 3 Jan 2020 00:00:00 GMT"}]}
```

> 不要求完全真实，只要格式对、字段在就行。

---

# 2. 测试 `src/ingest.py`（预处理）

目标：确认能把大 JSONL 变成 `clean.jsonl`。

## 步骤 2-1：运行预处理脚本

```bash
python src/ingest.py \
  --in ./data/mini_raw.jsonl \
  --out ./data/clean.jsonl \
  --max_rows 100
```

## 步骤 2-2：检查输出内容

```bash
head -n 5 ./data/clean.jsonl
```

你应该能看到每行都是这样的结构（字段名对就行）：

```json
{"id": "mini-0001", "title": "...", "abstract": "...", "categories": "cs.LG", "created": "Mon, 1 Jan 2020 00:00:00 GMT"}
```

**说明**：到这一步，`ingest.py` 流程 OK。

---

# 3. 测试 `src/index_bm25.py`（BM25 索引）

目标：确认 BM25 索引能被构建和查询。

## 步骤 3-1：构建 BM25 索引

```bash
python src/index_bm25.py \
  --in ./data/clean.jsonl \
  --index ./data/bm25.idx
```

命令结束时应打印类似：

```text
Built BM25 index with 3 docs → ./data/bm25.idx
```

## 步骤 3-2：在 Python 里简单查询一次

```bash
python -c "from src.index_bm25 import search_bm25; \
print(search_bm25('./data/bm25.idx', 'contrastive learning', 2))"
```

预期：终端打印一个包含若干 dict 的列表，里面有 `id/title/abstract/score` 等字段。  
如果能看到 `mini-0001` 被检出，说明 BM25 功能正常。

---

# 4. 测试 `src/index_dense.py`（稠密向量索引）

目标：确认向量库能构建和查询。

## 步骤 4-1：构建稠密向量库（Chroma）

```bash
python src/index_dense.py \
  --in ./data/clean.jsonl \
  --db ./data/chroma \
  --model bge-small-en-v1.5
```

命令结束时应打印类似：

```text
Built dense DB with 3 docs → ./data/chroma [arxiv]
```

## 步骤 4-2：简单测试一次 dense 检索

```bash
python -c "from src.index_dense import search_dense; \
print(search_dense('./data/chroma', 'graph neural networks', 2, 'bge-small-en-v1.5'))"
```

预期：输出列表中能看到 `mini-0002`，说明 dense 流程 OK。

---

# 5. 测试 `src/retriever.py`（统一检索接口）

目标：验证三种模式 `bm25 / dense / hybrid` 都能用。

> 确认 `configs/config.yaml` 中的路径和我们刚刚构建的一致：
> - `bm25.index_path: ./data/bm25.idx`
> - `dense.db: ./data/chroma`
> - `dense.model: bge-small-en-v1.5`

## 步骤 5-1：BM25 模式

```bash
python -c "from src.retriever import retrieve; \
print(retrieve('contrastive learning', 3, mode='bm25'))"
```

## 步骤 5-2：Dense 模式

```bash
python -c "from src.retriever import retrieve; \
print(retrieve('graph neural networks', 3, mode='dense'))"
```

## 步骤 5-3：Hybrid 模式

```bash
python -c "from src.retriever import retrieve; \
print(retrieve('transformer architectures', 3, mode='hybrid'))"
```

预期：

- 三个命令都能正常返回一个列表；
- 列表元素中有 `id/title/score` 字段；
- 不报错，说明 `retriever.py` 通路 OK。

---

# 6. 测试 `src/rag.py`（RAG + 大模型调用）

先用 **mock 模式** 测试，不需要真实 API。

## 步骤 6-1：在配置里使用 `provider: mock`

确保 `configs/config.yaml` 中：

```yaml
generation:
  provider: mock
  model: gpt-4o-mini
  max_tokens: 512
```

## 步骤 6-2：直接调用 `answer` 函数

```bash
python -c "from src.rag import answer; \
print(answer('What is contrastive learning?', mode='hybrid', topk=3))"
```

预期：

- 返回一个 dict，形如：

  ```python
  {'answer': '...', 'citations': [{'id': 'mini-0001', 'title': '...'}, ...]}
  ```

- `answer` 字段是字符串（mock 的话里面可能带 `[Mock Answer]`），说明：
  - `rag.py` 能成功调用 `retriever`；
  - 能构造上下文；
  - 能走完 LLMClient.generate（即使是 mock）。

> 若后续要接真实大模型，只需：
> - 把 `generation.provider` 改为 `openai` 或 `ollama`；
> - 设置好 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`；
> - 再次运行同样的命令即可。

---

# 7. 测试 `src/synth_qa.py`（合成问答数据）

目标：确认能从 `clean.jsonl` 生成 `synth_qa.jsonl`。

> 如果你暂时还没实现真正的 LLM 调用逻辑，也可以在 `generate_questions_for_doc` 里先写一个简单的假实现：
> 返回 `[f"What is the main idea of paper: {title}?"]` 这种固定问题，只要脚本能跑通就行。

## 步骤 7-1：生成一小份合成 QA

```bash
python src/synth_qa.py \
  --in ./data/clean.jsonl \
  --out ./data/synth_qa.jsonl \
  --sample_size 3 \
  --questions_per_doc 1
```

## 步骤 7-2：检查输出格式

```bash
head -n 5 ./data/synth_qa.jsonl
```

预期：每行是一个 JSON，包含：

```json
{"q": "...", "gold_ids": ["mini-0001"], "source": "synthetic_llm", "category": "cs.LG"}
```

字段名对即可，问题内容可以先不追求智能。

---

# 8. 测试 `src/eval.py`（离线评测）

目标：在刚刚生成的 `synth_qa.jsonl` 上，比较 `bm25/dense/hybrid` 检索的效果，确认评测脚本跑得起来。

## 步骤 8-1：在合成 QA 上评测检索层

```bash
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes bm25 dense hybrid \
  --out ./logs/metrics_synth.csv \
  --k 5
```

运行结束后，应打印类似：

```text
Metrics written to ./logs/metrics_synth.csv
```

## 步骤 8-2：查看评测结果

```bash
cat ./logs/metrics_synth.csv
```

预期：

- 第一行是表头，例如：`mode,k,recall,mrr,search_ms,gen_ms,end2end_ms`
- 后面每行对应一个模式，包含均值指标（数字即可，不必太“好看”）。

如果你想 **连生成时间一起测**，可加 `--include_gen`（这时会调 RAG 的大模型逻辑）：

```bash
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes bm25 dense hybrid \
  --out ./logs/metrics_synth_with_gen.csv \
  --k 5 \
  --include_gen
```

---

# 9. 一键回顾：每个文件是如何被测试到的？

- **`src/ingest.py`**  
  - 步骤 2：运行 `ingest.py`，生成 `data/clean.jsonl` 并检查输出。

- **`src/index_bm25.py`**  
  - 步骤 3：构建 `data/bm25.idx`，并用 `search_bm25` 查一次。

- **`src/index_dense.py`**  
  - 步骤 4：构建 `data/chroma/`，并用 `search_dense` 查一次。

- **`src/retriever.py`**  
  - 步骤 5：分别以 `bm25/dense/hybrid` 调用 `retrieve`，看返回结构。

- **`src/rag.py`**  
  - 步骤 6：用 `answer('What is contrastive learning?', 'hybrid', 5)`，检查返回的 `answer + citations`。

- **`src/synth_qa.py`**  
  - 步骤 7：运行脚本生成 `data/synth_qa.jsonl`，检查每行格式是否正确。

- **`src/eval.py`**  
  - 步骤 8：用 `synth_qa.jsonl` 评测，生成 `logs/metrics_synth.csv` 并查看内容。

到这里，**所有核心代码文件都至少被调用过一次完整流程**，新手按顺序照做即可确认代码端到端是“跑得起来”的。

---

## 10. 后续扩展（可选）

- 把 `mini_raw.jsonl` 换成真正的 `arxiv-metadata-oai-snapshot.json`，重复步骤 2～8 就是完整实验版；
- 补充/使用 `tests/` 目录里的单元测试文件（如果你后续按 project 文档写了 `tests/test_*.py`），再用：

```bash
pytest -q
```

做更细粒度的自动化验证。

---

**总结**：  
以上步骤保证了：从预处理、双索引构建、统一检索、RAG 生成、合成 QA 到评估，每个模块都有对应的测试命令，新手只要按章节顺序执行，就能验证整个项目的代码是完备可跑的。
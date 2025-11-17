我会基于你当前的 project3.md，把实现流程拆成“阶段 + 文件”两层结构：先用中文步骤解释每个文件应该做什么，然后给出对应的 Python 伪代码骨架（函数签名 + 注释 + pass / TODO 占位），方便你直接复制到源码里逐块填充逻辑。不会和你现有实现冲突，只是更“新手友好”的版本。



# 从零实现指南（新手向）  
—— 基于 project3.md 的代码实现步骤 + 伪代码骨架

> 阅读方式建议：  
> 先通读第 0～1 步，了解整体流程，再按“阶段”一节一节照着文件去写代码。

---

## 0. 总体实现顺序（建议）

1. **准备项目结构与环境**
   - 建好 [5020-RAG/](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG:0:0-0:0) 目录结构。
   - 准备 `requirements.txt`、`configs/config.yaml`。
2. **实现数据预处理：`src/ingest.py`**
3. **实现 BM25 索引：`src/index_bm25.py`**
4. **实现 稠密向量索引（Chroma）：`src/index_dense.py`**
5. **实现 统一检索接口：`src/retriever.py`**
6. **实现 RAG 调用大模型接口：`src/rag.py`**
7. **实现 合成问答生成：`src/synth_qa.py`（新需求）**
8. **实现 评测脚本：`src/eval.py`**
9. **最后跑通端到端命令**（project3 文档中的 Pipeline 一节）

下面按文件给出：  
- 先解释“这个文件是干嘛的”；  
- 再给出 **可以直接复制进去的伪代码骨架**（带注释和 `pass` / `TODO` 占位）。

---

## 1. `requirements.txt`（依赖文件）

**目标**：把要用到的库列出来，方便 `pip install -r requirements.txt`。

**最简推荐内容（示例）**：

```text
pyyaml
tqdm
rank-bm25
chromadb
sentence-transformers
openai
```

后续如果用到别的库（比如 `pytest`），再往里加。

---

## 2. `configs/config.yaml`（配置文件）

**目标**：把路径、模型名等写在配置里，而不是写死在代码里。

**示例骨架**：

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
  provider: mock          # mock | openai | ollama
  model: gpt-4o-mini      # 或 Qwen/LLaMA 等
  max_tokens: 512

runtime:
  max_context_chars: 6000
  seed: 42
```

---

## 3. `src/ingest.py` —— 大文件预处理

### 3.1 要做什么（给小白看的）

- 从超大的 `arxiv-metadata-oai-snapshot.json` 中 **一行一行地读**数据；
- 每一行是一个 JSON，解析后 **只留下**：
  - `id / title / abstract / categories / created`；
- 写入 `clean.jsonl`（也是一行一个 JSON）。

### 3.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
ingest.py

功能：从原始 arXiv JSONL 中流式抽取核心字段，写入更小的 clean.jsonl。
"""

import argparse
import json
from typing import Optional


def _parse_created(versions) -> Optional[str]:
    """
    从 versions 字段里解析 created 时间。
    - versions 通常是一个列表，比如 [{"version": "v1", "created": "..."}]
    - 如果解析失败，返回 None。
    """
    # TODO: 实现简单的 "取第一个版本的 created 字段"
    # 示例：
    # if not versions: return None
    # return versions[0].get("created")
    pass


def _clean_abstract(raw: str) -> str:
    """
    对摘要做非常轻量的清洗：
    - 去掉多余的换行
    - （可选）移除简单的 LaTeX 公式片段，比如 $...$ 或 \\(...\\)
    """
    # TODO: 基本实现示例：
    # text = (raw or "").replace("\n", " ")
    # 再做一些简单替换或正则清洗
    pass


def stream_clean_arxiv(input_path: str, output_path: str, max_rows: Optional[int] = None) -> None:
    """
    核心函数：流式读取 input_path，写出精简字段到 output_path。
    - max_rows 用于调试，只处理前 N 行。
    """
    # 1. 打开输入/输出文件（注意 encoding="utf-8"）
    # 2. 用 for i, line in enumerate(f): 逐行读取
    # 3. 对每一行：
    #    - strip 去掉首尾空白
    #    - 如果空行则 continue
    #    - 用 json.loads 解析成字典，注意 try/except 防止坏行
    #    - 从字典中取出 id/title/abstract/categories/versions
    #    - 调 _parse_created / _clean_abstract
    #    - 组合成一个新的 dict：{"id": ..., "title": ..., "abstract": ..., "categories": ..., "created": ...}
    #    - 用 json.dumps 写到输出文件，每行一个
    # 4. 每处理若干行可以 print 进度（可选）
    pass


def main():
    """
    命令行入口：
    python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="原始 arXiv JSONL 路径")
    parser.add_argument("--out", dest="out_path", required=True, help="输出 clean.jsonl 路径")
    parser.add_argument("--max_rows", type=int, default=None, help="调试用，只处理前 N 行")
    args = parser.parse_args()

    stream_clean_arxiv(args.in_path, args.out_path, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
```

---

## 4. `src/index_bm25.py` —— BM25 索引

### 4.1 要做什么

- 读 `clean.jsonl`；
- 把每条的 `title + "\n" + abstract` 作为文档文本；
- 用 BM25 建一个传统关键词索引，并存成 `bm25.idx`；
- 提供 `search_bm25(index_path, query, topk)` 查询函数。

### 4.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
index_bm25.py

功能：从 clean.jsonl 构建 BM25 索引，并提供搜索接口。
"""

import argparse
import json
import os
import pickle
import re
from typing import List, Dict

# from rank_bm25 import BM25Okapi  # 真正实现时需要导入


def _tokenize(text: str) -> List[str]:
    """
    非常简单的分词函数：
    - 全部转小写
    - 用正则把连续的字母/数字当作一个 token
    """
    # TODO: 使用 re.findall 之类实现
    pass


def build_bm25_index(clean_path: str, index_path: str) -> int:
    """
    从 clean.jsonl 构建 BM25 索引，并保存到 index_path。
    返回索引中的文档数量。
    """
    docs: List[Dict] = []
    tokens: List[List[str]] = []

    # 1. 逐行读取 clean.jsonl
    # 2. 对每行：
    #    - json.loads 得到 rec
    #    - 取出 id/title/abstract
    #    - 组合 text = title + "\n" + abstract
    #    - docs 里保存 {"id": ..., "title": ..., "abstract": ..., "text": text}
    #    - 用 _tokenize(text) 得到 token 列表，保存到 tokens
    # 3. 构建 BM25 索引对象 BM25Okapi(tokens)
    # 4. 把 docs 和 tokens（以及必要信息）用 pickle.dump 存到 index_path
    pass


def search_bm25(index_path: str, query: str, topk: int = 5) -> List[Dict]:
    """
    用 BM25 索引搜索 query，返回 Top-k 结果列表。
    每条结果包含：id/title/abstract/text/score。
    """
    # 1. 用 pickle.load 读取 index_path 中的 docs 和 tokens
    # 2. 初始化 BM25Okapi(tokens)
    # 3. 对 query 做 _tokenize → tokens_q
    # 4. 用 bm25.get_scores(tokens_q) 得到每篇文档的分数列表
    # 5. 取分数最高的 topk 个索引，组装输出结构
    pass


def main():
    """
    命令行入口示例：
    python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="clean.jsonl 路径")
    parser.add_argument("--index", dest="index_path", required=True, help="BM25 索引输出路径")
    args = parser.parse_args()

    # TODO: 调用 build_bm25_index，并打印构建了多少篇文档
    pass


if __name__ == "__main__":
    main()
```

---

## 5. `src/index_dense.py` —— 稠密向量索引（Chroma）

### 5.1 要做什么

- 从 `clean.jsonl` 读取文档（同样是 `title + "\n" + abstract`）；
- 用 `SentenceTransformer` 把文本编码成向量；
- 存入 Chroma 本地向量库（`data/chroma/`）；
- 提供 `search_dense(db_path, query, topk, model_name, collection)` 查询接口。

### 5.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
index_dense.py

功能：用 sentence-transformers + Chroma 构建稠密向量库，并提供搜索接口。
"""

import argparse
import json
import os
from typing import List, Dict

# import chromadb
# from sentence_transformers import SentenceTransformer


def _load_docs(clean_path: str) -> List[Dict]:
    """
    从 clean.jsonl 加载所有文档，返回列表：
    [{"id": ..., "title": ..., "text": ...}, ...]
    """
    docs: List[Dict] = []
    # 1. 逐行读取 clean.jsonl
    # 2. 对每行解析 JSON，取 id/title/abstract
    # 3. 组合 text = title + "\n" + abstract
    # 4. append 到 docs
    pass


def embed_and_build_vector_db(
    clean_path: str,
    db_path: str,
    model_name: str,
    collection: str = "arxiv",
    batch_size: int = 64,
) -> int:
    """
    从 clean.jsonl 构建稠密向量库。
    返回写入的文档数量。
    """
    # 1. 创建 Chroma PersistentClient（path = db_path）
    # 2. 如果 collection 已存在，可以先删掉再新建
    # 3. 调 _load_docs 读入文档列表
    # 4. 初始化 SentenceTransformer(model_name)
    # 5. 按 batch_size 切分 docs，循环：
    #    - texts = [d["text"] for d in batch]
    #    - 用 model.encode(texts, normalize_embeddings=True) 得到 embeddings
    #    - ids = [d["id"] ...]，metadatas = [{"title": d["title"]} ...]
    #    - col.add(ids=..., documents=texts, metadatas=metadatas, embeddings=embeddings)
    pass


def search_dense(
    db_path: str,
    query: str,
    topk: int,
    model_name: str,
    collection: str = "arxiv",
) -> List[Dict]:
    """
    用稠密向量检索 query，返回 Top-k 文档。
    每条包含：id/title/text/score（其中 score = 1 - 距离）。
    """
    # 1. 初始化 Chroma PersistentClient(path=db_path)
    # 2. get_collection(name=collection)
    # 3. 初始化 SentenceTransformer(model_name)
    # 4. 对 query 编码成向量
    # 5. 用 col.query(...) 检索，取 n_results=topk
    # 6. 把返回的 ids/documents/metadatas/distances 组装成标准字典列表
    pass


def main():
    """
    命令行入口示例：
    python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-en-v1.5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--db", dest="db_path", required=True)
    parser.add_argument("--model", dest="model", required=True)
    parser.add_argument("--collection", dest="collection", default="arxiv")
    args = parser.parse_args()

    # TODO: 调用 embed_and_build_vector_db，并打印构建数量
    pass


if __name__ == "__main__":
    main()
```

---

## 6. `src/retriever.py` —— 统一检索接口

### 6.1 要做什么

- 从 `configs/config.yaml` 读 BM25 和 dense 的路径、模型名；
- 对外提供 `retrieve(query, topk, mode, alpha)`：
  - `mode="bm25"`：仅 BM25；
  - `mode="dense"`：仅稠密；
  - `mode="hybrid"`：融合两路结果。

### 6.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
retriever.py

功能：统一封装 BM25 / Dense / Hybrid 检索接口。
"""

import argparse
import os
from typing import List, Dict

# import yaml
# from .index_bm25 import search_bm25
# from .index_dense import search_dense


def _load_config(path: str) -> Dict:
    """
    从 YAML 文件加载配置，如果不存在则返回空 dict。
    """
    # TODO: 用 yaml.safe_load 实现
    pass


def _norm(scores: List[float]) -> List[float]:
    """
    把一组分数做 min-max 归一化到 0~1。
    用于 hybrid 融合。
    """
    # TODO: 实现 min-max 归一化逻辑，注意所有分数都相同时的情况
    pass


def _to_map(items: List[Dict]) -> Dict[str, Dict]:
    """
    把检索结果列表转成 {id: doc} 的映射，方便后续按 id 查。
    """
    # TODO: 遍历 items，把每个 d 存入 m[d["id"]] = d
    pass


def retrieve(
    query: str,
    topk: int = 5,
    mode: str = "bm25",
    alpha: float = 0.5,
    config_path: str = "configs/config.yaml",
) -> List[Dict]:
    """
    对外统一的检索入口。
    - mode = "bm25" / "dense" / "hybrid"
    - alpha 只在 hybrid 模式下使用。
    """
    # 1. 读配置：bm25.index_path / dense.db / dense.model / dense.collection
    # 2. 如果 mode == "bm25": 直接调用 search_bm25
    # 3. 如果 mode == "dense": 调用 search_dense
    # 4. 如果 mode == "hybrid":
    #    - 各自取结果（可以多取一些，比如 max(topk, 10)）
    #    - 转成 id → doc 的 map
    #    - 取所有 id 的并集
    #    - 分别准备 bm25 和 dense 分数字典，做归一化
    #    - final_score = alpha * dense_norm + (1 - alpha) * bm25_norm
    #    - 组装统一的 doc 结构，并加入 final_score
    #    - 按 final_score 从大到小排序，取前 topk
    pass


def main():
    """
    简单命令行调试入口：
    python src/retriever.py "contrastive learning" --mode hybrid --topk 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--mode", default="bm25")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    # TODO: 调用 retrieve 并 print 结果
    pass


if __name__ == "__main__":
    main()
```

---

## 7. `src/rag.py` —— RAG & 调大模型 API

### 7.1 要做什么

- 调 `retriever.retrieve` 拿文档；
- 构造上下文字符串；
- 调用 LLM（API 或 mock）生成答案；
- 返回 `{"answer": ..., "citations": [...]}`。

### 7.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
rag.py

功能：把检索 + 拼上下文 + 调大模型 API 这一整套串起来。
"""

import argparse
import os
from typing import List, Dict

# import yaml
# from openai import OpenAI  # 使用 OpenAI 兼容接口
# from .retriever import retrieve


def _load_config(path: str) -> Dict:
    """
    复用读取 YAML 配置的逻辑（也可以直接复制 retriever 里的实现）。
    """
    pass


def build_context(docs: List[Dict], max_chars: int = 6000) -> str:
    """
    把若干篇文档拼成一个大的上下文字符串：
    [title]\n[abstract]\n\n---\n\n[下一篇...]
    - 超过 max_chars 就截断。
    """
    # 1. 按得分从高到低遍历 docs
    # 2. 对每一篇：构造一小段 context_text = f"[{id}] {title}\n{abstract}\n\n---\n\n"
    # 3. 不断往总 context 里 append，直到长度超出 max_chars 就停止
    pass


class LLMClient:
    """
    一个简单的大模型客户端封装：
    - provider="mock" 时不真正调用 API，只返回测试字符串；
    - provider="openai"/"ollama" 时用 OpenAI 兼容接口发请求。
    """

    def __init__(self, provider: str, model: str, base_url: str = None, api_key: str = None, max_tokens: int = 512):
        # TODO:
        # - 存储 provider/model/max_tokens
        # - 如果是 openai/ollama，初始化 OpenAI 客户端
        pass

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用模型生成回答。
        """
        # 伪逻辑：
        # if self.provider in ("openai", "ollama"):
        #     调用 self.client.chat.completions.create(...)
        #     返回第一条回复
        # elif self.provider == "mock":
        #     直接返回 user_prompt 部分 + "[Mock Answer]"，用于离线测试
        pass


def answer(
    query: str,
    mode: str = "hybrid",
    topk: int = 5,
    config_path: str = "configs/config.yaml",
) -> Dict:
    """
    对外的 RAG 问答入口。
    - 输入自然语言问题 query
    - 内部检索 + 构造 context + 调用 LLM
    - 输出答案文本 + 引用的 id/title 列表
    """
    # 1. 读取 YAML 配置，拿到 generation/runtime/retrieval 参数
    # 2. 调 retrieve(query, topk, mode, alpha=...) 拿到 docs
    # 3. 调 build_context(docs, max_chars=...) 得到 context 字符串
    # 4. 组装 system_prompt 和 user_prompt
    # 5. 用环境变量 OPENAI_BASE_URL / OPENAI_API_KEY 初始化 LLMClient
    # 6. 调 client.generate(...) 得到答案字符串
    # 7. 把 docs 里的 id/title 做成 citations 列表
    pass


def main():
    """
    命令行示例：
    python src/rag.py "What is contrastive learning?" --mode hybrid --topk 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    # TODO: 调用 answer 并 print 结果
    pass


if __name__ == "__main__":
    main()
```

---

## 8. `src/synth_qa.py` —— 合成问答数据生成（新模块）

### 8.1 要做什么

- 从 `clean.jsonl` 抽样若干篇论文；
- 对每篇论文的 `title + abstract`，让 LLM 生成 1~N 个问题；
- 写出 `data/synth_qa.jsonl`，格式：

```json
{"q": "...", "gold_ids": ["0704.0001"], "source": "synthetic_llm", "category": "cs.CL"}
```

### 8.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
synth_qa.py

功能：利用 LLM 为论文摘要自动生成“可用于检索评测的问题”。
"""

import argparse
import json
import random
from typing import List, Dict, Optional

# from .rag import LLMClient, _load_config  # 可以重用 RAG 里的封装


def _iter_clean_docs(clean_path: str):
    """
    生成器：逐行读取 clean.jsonl，yield 单篇文档字典。
    """
    # for line in open(...):
    #   解析 JSON，yield
    pass


def _primary_category(categories: str) -> Optional[str]:
    """
    从 categories 字符串中取出第一个标签，比如 "cs.CL stat.ML" -> "cs.CL"。
    """
    # TODO: split + 取第一个
    pass


def generate_questions_for_doc(
    title: str,
    abstract: str,
    n_q: int,
    client: "LLMClient",
) -> List[str]:
    """
    输入单篇论文的 title + abstract，调用 LLM 生成 n_q 个问题。
    返回问题字符串列表。
    """
    # 1. 构造 system_prompt：说明“你是学术助教，请根据论文标题和摘要生成若干可以用该摘要回答的问题”
    # 2. 构造 user_prompt：把 title/abstract 放进去，要求输出固定格式（比如 JSON 数组）
    # 3. 调 client.generate(system_prompt, user_prompt)
    # 4. 对返回的文本做简单解析（可以先假定 LLM 输出一行一个问题，简化处理）
    pass


def generate_synthetic_qa(
    clean_path: str,
    out_path: str,
    sample_size: int = 1000,
    questions_per_doc: int = 2,
    category_filter: Optional[str] = None,
    config_path: str = "configs/config.yaml",
) -> int:
    """
    核心函数：抽样 sample_size 篇论文，对每篇生成 questions_per_doc 个问题，并写入 out_path。
    返回生成的 QA 条数。
    """
    # 1. 读取配置，初始化 LLMClient（可使用单独的 provider/model 专门用于问题生成）
    # 2. 遍历 _iter_clean_docs(clean_path)，可以先存到列表 docs 或边遍历边抽样
    #    - 如果 category_filter 不为空，只保留主类别相等的文档
    # 3. 对通过筛选的文档做随机抽样（数量 = sample_size）
    # 4. 对每篇抽样到的文档：
    #    - 调 generate_questions_for_doc(title, abstract, questions_per_doc, client)
    #    - 对每个生成的问题 q：
    #        组装 record = {
    #          "q": q,
    #          "gold_ids": [id],
    #          "source": "synthetic_llm",
    #          "category": primary_category
    #        }
    #        用 json.dumps(record, ensure_ascii=False) 写入 out_path
    pass


def main():
    """
    命令行示例：
    python src/synth_qa.py \
      --in ./data/clean.jsonl \
      --out ./data/synth_qa.jsonl \
      --sample_size 500 \
      --questions_per_doc 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--questions_per_doc", type=int, default=2)
    parser.add_argument("--category", dest="category_filter", default=None)
    args = parser.parse_args()

    # TODO: 调用 generate_synthetic_qa 并打印生成条数
    pass


if __name__ == "__main__":
    main()
```

---

## 9. `src/eval.py` —— 离线评测（支持合成 QA）

### 9.1 要做什么

- 读取 QA 数据（真实或合成），每行至少有：
  - `q`：问题字符串；
  - `gold_ids`：正确论文 id 列表；
- 对每个模式 (`bm25/dense/hybrid`)：
  - 调 `retriever.retrieve(q, k, mode)`；
  - 计算 Recall@k 和 MRR；
  - 统计平均检索时间、可选生成时间；
- 最终写出 CSV 文件。

### 9.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
eval.py

功能：在真实或合成 QA 集上，比较不同检索策略的效果和延迟。
"""

import argparse
import csv
import json
import os
import time
from statistics import mean
from typing import List, Dict

# from .retriever import retrieve
# from .rag import answer as rag_answer


def _read_qa(path: str) -> List[Dict]:
    """
    读取 QA JSONL 文件，返回列表。
    每行至少包含 "q" 和 "gold_ids"。
    """
    items: List[Dict] = []
    # 1. 逐行读取 path
    # 2. strip 后如果是空行就跳过
    # 3. json.loads 加入 items
    pass


def _recall_at_k(retrieved: List[str], gold: List[str]) -> float:
    """
    如果 Top-k 结果中至少命中一个 gold id，则返回 1，否则返回 0。
    （最简单版本的 Recall@k）
    """
    # TODO: 使用集合判断是否有交集
    pass


def _mrr_at_k(retrieved: List[str], gold: List[str]) -> float:
    """
    Mean Reciprocal Rank 的单样本版本：
    找出第一个命中 gold 的位置 i（从 0 开始），返回 1/(i+1)；如果完全没命中返回 0。
    """
    # TODO: 遍历 retrieved，遇到在 gold 中的 id 就计算并返回
    pass


def evaluate(
    qa_path: str,
    modes: List[str],
    out_csv: str,
    k: int = 5,
    include_gen: bool = False,
) -> None:
    """
    核心评测函数。
    - 对每种检索模式分别统计 recall/mrr/平均耗时
    - 写出到 CSV 文件
    """
    # 1. 调 _read_qa 读取所有样本
    # 2. 对每个 mode in modes：
    #    - 初始化列表：recalls/mrrs/search_times/gen_times/total_times
    #    - 遍历每条样本 item：
    #        q = item["q"]; gold = item["gold_ids"]
    #        t0 = time.perf_counter()
    #        res = retrieve(q, k, mode)
    #        t1 = time.perf_counter()
    #        ids = [r["id"] for r in res]
    #        recalls.append(_recall_at_k(ids, gold))
    #        mrrs.append(_mrr_at_k(ids, gold))
    #        search_ms = (t1 - t0) * 1000
    #        如果 include_gen:
    #            调 rag_answer(q, mode, k)，测生成时间
    #    - 统计平均 recall/mrr/时间，构成一行字典加入 rows
    # 3. 写 CSV 文件，表头包含 ["mode", "k", "recall", "mrr", "search_ms", "gen_ms", "end2end_ms"]
    pass


def main():
    """
    命令行示例：
    python src/eval.py \
      --qa ./data/synth_qa.jsonl \
      --modes bm25 dense hybrid \
      --out ./logs/metrics_synth.csv \
      --k 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", dest="qa", required=True)
    parser.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--out", dest="out_csv", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--include_gen", action="store_true")
    args = parser.parse_args()

    # TODO: 调用 evaluate，然后打印输出路径
    pass


if __name__ == "__main__":
    main()
```

---

## 10. 最后如何使用这份“实现步骤文档”

- **建议做法**：
  - 每个文件新建/打开后，把对应的伪代码骨架复制进去；
  - 按照注释里的“TODO” 一条条实现逻辑；
  - 每完成一个文件，就尝试运行该文件对应的命令行示例，确认能跑；
  - 全部完成后，再按 project3.md 中的 Pipeline 顺序跑通端到端流程。

如果你希望下一步，我可以帮你 **从某个具体文件开始，把 TODO 逐步填成真实代码**（比如先把 `ingest.py` 写完整再测试）。
```markdown
# 从零实现指南 v4（code4 文档，完整版）  
—— 从 0 搭建 + 实现 project4 中的所有模块与测试流程

> 使用方式：  
> - 把本文件当成“从零实现 + 自测 + 消融实验”的总说明。  
> - 每个模块都有两部分：  
>   - 先用中文说明“要干什么”；  
>   - 再给一个可以直接复制到代码文件里的 **Python 伪代码骨架**（函数签名 + 注释 + `pass` / `TODO`）。  
> - 全流程覆盖：基础 v3 系统 + v4 所有创新模块 + 测试与消融实验命令。

---

## 0. 总体实现顺序（建议）

1. **准备项目结构与环境**
   - 目录 [5020-RAG/](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG:0:0-0:0)；
   - 安装虚拟环境和依赖（`requirements.txt`）。
2. **准备配置文件 `configs/config.yaml`**
   - 路径、模型名、策略开关（包括 v4 的 dynamic_alpha/rerank/expansion 等）。
3. **实现基础数据与索引模块**
   - `src/ingest.py`：从原始 arXiv JSONL 抽取核心字段 → `clean.jsonl`；
   - `src/index_bm25.py`：BM25 索引；
   - `src/index_dense.py`：稠密向量 + Chroma 向量库。
4. **实现基础检索与 RAG**
   - `src/retriever.py`：基础 `retrieve`（bm25/dense/hybrid）；
   - `src/rag.py`：RAG（`answer` + `LLMClient`）。
5. **实现合成 QA 与评测**
   - `src/synth_qa.py`：合成问答数据生成；
   - `src/eval.py`：离线评测（Recall/MRR/Latency）。
6. **实现 v4 创新模块**
   - `src/heuristics.py`：query 启发式 + dynamic alpha + SLA 策略；
   - `src/reranker.py`：Cross-Encoder rerank；
   - `src/expansion.py`：PRF + LLM 查询扩展；
   - `src/snippets.py`：句级 evidence 抽取与高亮；
   - 扩展 `retriever.py` 增加 `retrieve_enhanced`；
   - 扩展 `rag.py` 增加 `enhanced_answer`。
7. **实现并执行完整测试与消融实验**
   - 基础功能测试（ingest/index/retriever/rag/synth_qa/eval）；
   - 每个创新点的开关 + 消融实验命令；
   - 查看并对比 `logs/*.csv` 指标。

---

## 1. 环境准备与 `requirements.txt`

### 1.1 环境创建

```bash
cd 5020-RAG

python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -U pip
```

### 1.2 `requirements.txt` 建议内容

在项目根目录创建或编辑 `requirements.txt`：

```text
pyyaml
tqdm
rank-bm25
chromadb
sentence-transformers
openai
# 如需交互式测试，可加：ipython / jupyter（可选）
```

安装依赖：

```bash
pip install -U -r requirements.txt
```

---

## 2. 配置文件 `configs/config.yaml`

### 2.1 要做什么

- 集中管理数据路径、索引路径、模型名称；
- 控制检索模式、v4 策略开关、RAG 参数等。

在 `configs/` 目录中创建 `config.yaml`，示例骨架：

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
  alpha: 0.5           # v3: 固定 alpha
  dynamic_alpha: true  # v4: 是否启用自适应 alpha
  expansion:
    enable: false      # v4: 查询扩展总开关
    prf:
      enable: true
      m_docs: 5        # PRF 使用的 top-m 文档数
      top_terms: 10    # 从 PRF 中选多少高频词
    llm:
      enable: false    # 初期可先关掉 LLM 改写
      n_variants: 3

rerank:
  enable: false
  model: bge-reranker-base
  topn: 50
  batch_size: 16

rag:
  use_evidence_snippets: true
  evidence:
    per_doc: 2         # 每篇文档最多选多少句
    max_total: 10      # 总共最多多少句
    method: bm25       # 句级打分方式：bm25 | embedding

generation:
  provider: mock       # mock | openai | ollama
  model: gpt-4o-mini
  max_tokens: 512

category:
  enable_filter: false
  allowed_prefixes: ["cs."]
  enable_boost: true
  boost_factor: 0.1

runtime:
  max_context_chars: 6000
  seed: 42
  latency_budget_ms: 1000
```

---

## 3. `src/ingest.py` —— 大文件预处理

### 3.1 要做什么

- 从大 JSONL 流式读取；
- 抽取 `id/title/abstract/categories/created`；
- 输出为 `clean.jsonl`。

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
    # TODO:
    # if not versions:
    #     return None
    # return versions[0].get("created")
    pass


def _clean_abstract(raw: str) -> str:
    """
    对摘要做非常轻量的清洗：
    - 去掉多余的换行
    - （可选）移除简单的 LaTeX 公式片段，比如 $...$ 或 \\(...\\)
    """
    # TODO:
    # text = (raw or "").replace("\n", " ")
    # # 如需，做一些正则清洗
    # return text.strip()
    pass


def stream_clean_arxiv(input_path: str, output_path: str, max_rows: Optional[int] = None) -> None:
    """
    核心函数：流式读取 input_path，写出精简字段到 output_path。
    - max_rows 用于调试，只处理前 N 行。
    """
    # 1. 打开输入/输出文件（encoding="utf-8"）
    # 2. 用 for i, line in enumerate(f): 逐行读取
    # 3. 对每一行：
    #    - strip 去掉首尾空白
    #    - 如果空行则 continue
    #    - try: obj = json.loads(line)
    #    - except: continue
    #    - 从 obj 中取出 id/title/abstract/categories/versions
    #    - created = _parse_created(versions)
    #    - abstract = _clean_abstract(abstract)
    #    - new_rec = {"id": ..., "title": ..., "abstract": ..., "categories": ..., "created": created}
    #    - json.dump(new_rec, out_file, ensure_ascii=False); 写一个换行
    # 4. 若 max_rows 不为 None，处理到 N 行后 break
    pass


def main():
    """
    命令行：
    python src/ingest.py --in ./data/arxiv-metadata-oai-snapshot.json --out ./data/clean.jsonl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()

    stream_clean_arxiv(args.in_path, args.out_path, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
```

---

## 4. `src/index_bm25.py` —— BM25 索引

### 4.1 要做什么

- 读 `clean.jsonl`；
- 构建 BM25 索引；
- 序列化到 `bm25.idx`；
- 提供 `search_bm25(index_path, query, topk)`。

### 4.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
index_bm25.py

功能：从 clean.jsonl 构建 BM25 索引，并提供搜索接口。
"""

import argparse
import json
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
    # TODO:
    # text = text.lower()
    # return re.findall(r"[a-z0-9]+", text)
    pass


def build_bm25_index(clean_path: str, index_path: str) -> int:
    """
    从 clean.jsonl 构建 BM25 索引，并保存到 index_path。
    返回索引中的文档数量。
    """
    docs: List[Dict] = []
    tokens: List[List[str]] = []

    # TODO:
    # with open(clean_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         rec = json.loads(line)
    #         text = (rec.get("title", "") + "\n" + rec.get("abstract", "")).strip()
    #         docs.append({"id": rec["id"], "title": rec["title"], "abstract": rec["abstract"], "text": text})
    #         tokens.append(_tokenize(text))
    #
    # bm25 = BM25Okapi(tokens)
    # with open(index_path, "wb") as f_out:
    #     pickle.dump({"docs": docs, "tokens": tokens}, f_out)
    #
    # return len(docs)
    pass


def search_bm25(index_path: str, query: str, topk: int = 5) -> List[Dict]:
    """
    用 BM25 索引搜索 query，返回 Top-k 结果列表。
    每条结果包含：id/title/abstract/text/score。
    """
    # TODO:
    # with open(index_path, "rb") as f:
    #     data = pickle.load(f)
    # docs, tokens = data["docs"], data["tokens"]
    # bm25 = BM25Okapi(tokens)
    # q_tokens = _tokenize(query)
    # scores = bm25.get_scores(q_tokens)
    # 取分数最高的 topk 索引，组装结果列表
    pass


def main():
    """
    命令行：
    python src/index_bm25.py --in ./data/clean.jsonl --index ./data/bm25.idx
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--index", dest="index_path", required=True)
    args = parser.parse_args()

    # n_docs = build_bm25_index(args.in_path, args.index_path)
    # print(f"Built BM25 index with {n_docs} docs -> {args.index_path}")
    pass


if __name__ == "__main__":
    main()
```

---

## 5. `src/index_dense.py` —— 稠密向量索引（Chroma）

### 5.1 要做什么

- 从 `clean.jsonl` 读文档；
- 使用 `SentenceTransformer` 生成向量；
- 写入 Chroma 持久化向量库；
- 提供 `search_dense(db_path, query, topk, model_name, collection)`。

### 5.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
index_dense.py

功能：用 sentence-transformers + Chroma 构建稠密向量库，并提供搜索接口。
"""

import argparse
import json
from typing import List, Dict

# import chromadb
# from sentence_transformers import SentenceTransformer


def _load_docs(clean_path: str) -> List[Dict]:
    """
    从 clean.jsonl 加载所有文档，返回：
    [{"id": ..., "title": ..., "text": ...}, ...]
    """
    docs: List[Dict] = []
    # TODO:
    # with open(clean_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         rec = json.loads(line)
    #         text = (rec.get("title", "") + "\n" + rec.get("abstract", "")).strip()
    #         docs.append({"id": rec["id"], "title": rec["title"], "text": text})
    # return docs
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
    # TODO:
    # 1. client = chromadb.PersistentClient(path=db_path)
    # 2. 如果 collection 已存在，可删掉重建
    # 3. docs = _load_docs(clean_path)
    # 4. model = SentenceTransformer(model_name)
    # 5. 按 batch_size 切分 docs，循环 encode + col.add(...)
    # 6. 返回 len(docs)
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
    每条包含：id/title/text/score（可用 1-距离 作为 score）。
    """
    # TODO:
    # 1. client = chromadb.PersistentClient(path=db_path)
    # 2. col = client.get_collection(collection)
    # 3. model = SentenceTransformer(model_name)
    # 4. q_emb = model.encode([query], normalize_embeddings=True)
    # 5. col.query(query_embeddings=q_emb, n_results=topk, include=["metadatas","documents","distances"])
    # 6. 整理结果字段
    pass


def main():
    """
    命令行：
    python src/index_dense.py --in ./data/clean.jsonl --db ./data/chroma --model bge-small-en-v1.5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--db", dest="db_path", required=True)
    parser.add_argument("--model", dest="model", required=True)
    parser.add_argument("--collection", dest="collection", default="arxiv")
    args = parser.parse_args()

    # n_docs = embed_and_build_vector_db(args.in_path, args.db_path, args.model, args.collection)
    # print(f"Built dense DB with {n_docs} docs -> {args.db_path} [{args.collection}]")
    pass


if __name__ == "__main__":
    main()
```

---

## 6. `src/heuristics.py` —— 查询/类别/SLA 启发式（v4 新增）

```python
#!/usr/bin/env python3
"""
heuristics.py

功能：
- 对 query 做简单特征分析，输出查询类型（term/semantic/mixed）和建议 alpha。
- 可选：对 query 做粗略类别预测。
- 根据延迟预算给出策略选择建议（是否启用 rerank / expansion 等）。
"""

from typing import Dict, Optional


def _basic_query_features(q: str) -> Dict:
    """
    提取 query 的简单特征：
    - 字符长度 / token 数
    - 数字和符号的占比
    - 大写字母/缩写比例等
    """
    # TODO:
    # len_chars = len(q)
    # tokens = q.split()
    # num_digits = sum(ch.isdigit() for ch in q)
    # num_punct = sum(ch in ",.;:!?[]" for ch in q)
    # return {"len_chars": len_chars, "len_tokens": len(tokens), ...}
    pass


def classify_query(q: str) -> Dict:
    """
    将 query 粗分为 'term' / 'semantic' / 'mixed'，并给出建议 alpha。
    返回示例：
    {
      "type": "term",
      "alpha": 0.2,
      "features": {...}
    }
    """
    # TODO:
    # feat = _basic_query_features(q)
    # 简单规则：
    # if feat["len_tokens"] <= 4 and (num_digits+num_punct) 比例高 -> term
    # elif feat["len_tokens"] >= 8 -> semantic
    # else -> mixed
    # 然后按类型设定 alpha
    pass


def predict_query_category(q: str) -> Optional[str]:
    """
    可选：根据 query 中关键词粗略预测一个 arXiv 类别，比如 "cs.CL"。
    - 可用简单规则，如包含 'NLP','language model' 等词 → 'cs.CL'
    - 暂时可以返回 None，后续再完善。
    """
    # TODO: 按需实现，初期可直接 return None
    pass


def choose_strategy(latency_budget_ms: int) -> Dict:
    """
    根据延迟预算选择策略组合。
    返回示例：
    {
      "mode": "bm25" | "dense" | "hybrid",
      "enable_rerank": bool,
      "enable_expansion": bool
    }
    """
    # TODO:
    # if latency_budget_ms < 500:
    #     return {"mode": "bm25", "enable_rerank": False, "enable_expansion": False}
    # elif latency_budget_ms < 1200:
    #     return {"mode": "hybrid", "enable_rerank": False, "enable_expansion": True}
    # else:
    #     return {"mode": "hybrid", "enable_rerank": True, "enable_expansion": True}
    pass
```

---

## 7. `src/retriever.py` —— 基础 + v4 增强检索入口

> 建议先实现基础 `retrieve`（v3），再在同文件中增加 `_apply_category_logic` + `retrieve_enhanced`（v4）。

### 7.1 伪代码骨架（重点放在增强部分）

```python
#!/usr/bin/env python3
"""
retriever.py

功能：
- 基础检索接口 retrieve：支持 bm25 / dense / hybrid。
- v4 增强接口 retrieve_enhanced：支持 dynamic alpha / expansion / rerank / category / SLA。
"""

import argparse
from typing import List, Dict, Optional

# import yaml
# from .index_bm25 import search_bm25
# from .index_dense import search_dense
# from .heuristics import classify_query, predict_query_category, choose_strategy
# from .expansion import expand_query
# from .reranker import rerank as rerank_docs


def _load_config(path: str) -> Dict:
    """
    从 YAML 文件加载配置。
    """
    # TODO:
    # with open(path, "r", encoding="utf-8") as f:
    #     return yaml.safe_load(f)
    pass


def _norm(scores: List[float]) -> List[float]:
    """
    min-max 归一化到 0~1。
    """
    # TODO:
    # if not scores: return []
    # mn, mx = min(scores), max(scores)
    # if mx == mn: return [0.0 for _ in scores]
    # return [(s-mn)/(mx-mn) for s in scores]
    pass


def retrieve(
    query: str,
    topk: int = 5,
    mode: str = "bm25",
    alpha: float = 0.5,
    config_path: str = "configs/config.yaml",
) -> List[Dict]:
    """
    基础检索入口（v3 同款）。
    - mode: "bm25" / "dense" / "hybrid"
    - hybrid: alpha * dense + (1-alpha) * bm25
    """
    # TODO:
    # cfg = _load_config(config_path)
    # bm25_path = cfg["bm25"]["index_path"]
    # dense_cfg = cfg["dense"]
    # if mode == "bm25": return search_bm25(bm25_path, query, topk)
    # if mode == "dense": return search_dense(dense_cfg["db"], query, topk, dense_cfg["model"], dense_cfg["collection"])
    # if mode == "hybrid":
    #   bm25_res = search_bm25(...)
    #   dense_res = search_dense(...)
    #   # 合并 & score 归一化 & 融合
    pass


def _apply_category_logic(docs: List[Dict], query_cat: Optional[str], cat_cfg: Dict) -> List[Dict]:
    """
    根据类别配置进行过滤/加权。
    """
    # TODO:
    # def primary_cat(d):
    #   cats = d.get("categories") or ""
    #   return cats.split()[0] if cats else ""
    #
    # if cat_cfg.get("enable_filter"):
    #   prefixes = cat_cfg.get("allowed_prefixes") or []
    #   docs = [d for d in docs if any(primary_cat(d).startswith(p) for p in prefixes)]
    #
    # if cat_cfg.get("enable_boost") and query_cat:
    #   bf = cat_cfg.get("boost_factor", 0.0)
    #   for d in docs:
    #       if primary_cat(d).startswith(query_cat.split(".")[0]):  # 简单前缀匹配
+#
+    #       d["score"] = d.get("score", 0.0) * (1.0 + bf)
    #
    # return docs
    pass


def retrieve_enhanced(
    query: str,
    topk: int = 5,
    mode: str = "hybrid",
    alpha: Optional[float] = None,
    config_path: str = "configs/config.yaml",
    use_dynamic_alpha: bool = True,
    enable_expansion: bool = False,
    enable_rerank: bool = False,
    latency_budget_ms: Optional[int] = None,
) -> List[Dict]:
    """
    增强版检索入口。
    """
    # TODO 大致流程：
    # 1. cfg = _load_config(config_path)
    # 2. 如果 latency_budget_ms 不为空：strategy = choose_strategy(latency_budget_ms)
    #    用 strategy 覆盖 mode/enable_rerank/enable_expansion
    # 3. 如果 use_dynamic_alpha 且 alpha is None:
    #       info = classify_query(query)
    #       alpha = info["alpha"]
    # 4. queries = [query]
    #    如果 enable_expansion：queries = expand_query(query, client=None, ...)
    # 5. 对每个 q' in queries:
    #       调用基础检索 retrieve(q', topN, mode, alpha) （topN 可 > topk）
    #       合并到 {id: doc} 映射，注意 score 的累加或取 max
    # 6. 把映射转成列表 docs_all
    # 7. query_cat = predict_query_category(query)
    #    docs_all = _apply_category_logic(docs_all, query_cat, cfg["category"])
    # 8. 如果 enable_rerank:
    #       topN = cfg["rerank"]["topn"]
    #       取 docs_all 的前 topN 调 rerank_docs(query, topN_docs, model_name, topk)
    #       返回重排结果
    #    否则：
    #       按 doc["score"] 排序，取前 topk
    pass


def main():
    """
    命令行简单测试：
    python src/retriever.py "contrastive learning" --mode hybrid --topk 5 --enhanced
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--enhanced", action="store_true")
    args = parser.parse_args()

    # TODO:
    # if args.enhanced:
    #     res = retrieve_enhanced(args.query, args.topk, mode=args.mode)
    # else:
    #     res = retrieve(args.query, args.topk, mode=args.mode)
    # print(res)
    pass


if __name__ == "__main__":
    main()
```

---

## 8. `src/reranker.py` —— Cross-Encoder 精排

```python
#!/usr/bin/env python3
"""
reranker.py

功能：使用 Cross-Encoder 对初排文档列表进行精排。
"""

from typing import List, Dict

# from sentence_transformers import CrossEncoder

_RERANKER_MODEL = None
_RERANKER_NAME = None


def load_reranker(model_name: str):
    """
    初始化或从缓存中获取 CrossEncoder 模型。
    """
    global _RERANKER_MODEL, _RERANKER_NAME
    # TODO:
    # if _RERANKER_MODEL is None or _RERANKER_NAME != model_name:
    #     _RERANKER_MODEL = CrossEncoder(model_name)
    #     _RERANKER_NAME = model_name
    # return _RERANKER_MODEL
    pass


def rerank(
    query: str,
    docs: List[Dict],
    model_name: str,
    topk: int,
    batch_size: int = 16,
) -> List[Dict]:
    """
    对初排 docs 使用 Cross-Encoder 精排，返回 Top-k docs。
    - docs 每个元素至少包含字段 "text"（或 title+abstract 拼接）。
    """
    # TODO:
    # model = load_reranker(model_name)
    # pairs = [(query, d["text"]) for d in docs]
    # scores = model.predict(pairs, batch_size=batch_size)
    # for d, s in zip(docs, scores):
    #     d["rerank_score"] = float(s)
    # docs_sorted = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)
    # return docs_sorted[:topk]
    pass
```

---

## 9. `src/expansion.py` —— 查询扩展（PRF + LLM 改写）

```python
#!/usr/bin/env python3
"""
expansion.py

功能：
- PRF：基于初排结果的伪相关反馈，提取高频关键词。
- LLM 改写：调用 LLMClient 将 query 改写为多个等价问法。
- expand_query：综合上述两种方式，输出一组 query 变体列表。
"""

from typing import List, Dict, Optional

# from .retriever import retrieve
# from .rag import LLMClient


def prf_terms(
    query: str,
    retriever_cfg: Dict,
    m_docs: int = 5,
    top_terms: int = 10,
) -> List[str]:
    """
    使用当前检索后端，对 query 做一次初排，从 top-m 文档统计关键词。
    """
    # TODO:
    # 1. 调基础 retrieve(query, topk=m_docs, mode=retriever_cfg["mode"], alpha=...)
    # 2. 将这些文档的 text 拼在一起，做简单 tokenizer
    # 3. 统计词频，去掉停用词/过短词，按频率排序
    # 4. 返回前 top_terms 个词
    pass


def llm_expand(
    query: str,
    client: "LLMClient",
    n_variants: int = 3,
) -> List[str]:
    """
    使用 LLM 将 query 改写为 n 个等价或更具体的问法。
    """
    # TODO:
    # system_prompt = "你是一个检索查询改写助手..."
    # user_prompt = f"请基于以下查询生成 {n_variants} 个意思相近的检索问题，每行一个：\n\n{query}"
    # text = client.generate(system_prompt, user_prompt)
    # lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # return lines[:n_variants]
    pass


def expand_query(
    query: str,
    client: Optional["LLMClient"],
    use_prf: bool = True,
    use_llm: bool = True,
    retriever_cfg: Optional[Dict] = None,
    prf_cfg: Optional[Dict] = None,
    llm_cfg: Optional[Dict] = None,
) -> List[str]:
    """
    综合 PRF 和 LLM 改写，返回若干 query 变体。
    """
    # TODO:
    # variants = [query]
    # if use_prf and retriever_cfg is not None:
    #     terms = prf_terms(query, retriever_cfg, **(prf_cfg or {}))
    #     if terms:
    #         expanded_q = query + " " + " ".join(terms[:3])
    #         variants.append(expanded_q)
    # if use_llm and client is not None:
    #     llm_qs = llm_expand(query, client, **(llm_cfg or {}))
    #     variants.extend(llm_qs)
    # 去重：variants = list(dict.fromkeys(variants))
    # return variants
    pass
```

---

## 10. `src/snippets.py` —— 句级证据抽取

```python
#!/usr/bin/env python3
"""
snippets.py

功能：
- 将文档文本切分为句子。
- 对句子进行相关性打分，并选出作为 evidence 的 Top 句。
"""

import re
from typing import List, Dict


def sentence_split(text: str) -> List[str]:
    """
    简单分句：
    - 按句号/问号/感叹号等 split
    - 去掉空句
    """
    # TODO:
    # parts = re.split(r"[。！？.!?]", text)
    # return [p.strip() for p in parts if p.strip()]
    pass


def score_sentences(
    query: str,
    doc: Dict,
    max_sentences: int = 3,
    method: str = "bm25",
) -> List[Dict]:
    """
    对单个文档中的句子进行打分，返回若干句子及其分数。
    """
    # TODO:
    # text = doc.get("text") or (doc.get("title","") + " " + doc.get("abstract",""))
    # sentences = sentence_split(text)
    # scored = []
    # if method == "bm25":
    #   q_tokens = set(query.lower().split())
    #   for s in sentences:
    #       s_tokens = s.lower().split()
    #       overlap = len(q_tokens.intersection(s_tokens))
    #       if overlap > 0:
    #           scored.append({"sentence": s, "score": float(overlap)})
    # elif method == "embedding":
    #   # 可选：使用 embedding 模型计算相似度
    #   pass
    # scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    # return scored_sorted[:max_sentences]
    pass


def select_evidence_for_docs(
    query: str,
    docs: List[Dict],
    per_doc: int = 2,
    max_total: int = 10,
) -> List[Dict]:
    """
    针对多个文档选取句级 evidence。
    返回示例：
    [{"id": doc_id, "title": title, "sentence": s, "score": 0.95}, ...]
    """
    evidences: List[Dict] = []
    # TODO:
    # for d in docs:
    #     ss = score_sentences(query, d, max_sentences=per_doc)
    #     for item in ss:
    #         evidences.append({
    #             "id": d.get("id"),
    #             "title": d.get("title"),
    #             "sentence": item["sentence"],
    #             "score": item["score"],
    #         })
    # evidences_sorted = sorted(evidences, key=lambda x: x["score"], reverse=True)
    # return evidences_sorted[:max_total]
    pass
```

---

## 11. `src/rag.py` —— RAG 基础 & v4 增强

> 这里只给 RAG 关键骨架，假设你会复用 code3 中的 `LLMClient` 和基础 `answer`，在此基础上新增 `build_context_with_evidence` 与 `enhanced_answer`。

```python
#!/usr/bin/env python3
"""
rag.py

功能：
- 基础 RAG 流程：检索 + 构造上下文 + 调用 LLM。
- v4 增强版：使用 retrieve_enhanced + 句级 evidence 构造上下文，并返回 evidence。
"""

import argparse
from typing import List, Dict

# import yaml
# from openai import OpenAI
# from .retriever import retrieve, retrieve_enhanced
# from .snippets import select_evidence_for_docs


class LLMClient:
    """
    大模型调用封装：
    - provider="mock"：返回伪造答案；
    - provider="openai"/"ollama"：使用 OpenAI 兼容接口。
    """

    def __init__(self, provider: str, model: str, base_url: str = None, api_key: str = None, max_tokens: int = 512):
        # TODO: 保存参数，并在非 mock 时初始化 OpenAI 客户端
        pass

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用模型生成回答。
        """
        # TODO:
        # if self.provider == "mock":
        #   return user_prompt[:100] + " ... [Mock Answer]"
        # else:
        #   调 self.client.chat.completions.create(...)
        pass


def _load_config(path: str) -> Dict:
    # TODO: yaml.safe_load
    pass


def build_context(docs: List[Dict], max_chars: int) -> str:
    """
    基础版：使用文档的 title+abstract 拼接上下文。
    """
    # TODO: 与 code3 类似
    pass


def build_context_with_evidence(
    evidences: List[Dict],
    max_chars: int = 4000,
) -> str:
    """
    使用句级 evidence 构造上下文。
    """
    # TODO:
    # evidences_sorted = sorted(evidences, key=lambda x: x["score"], reverse=True)
    # context = ""
    # for e in evidences_sorted:
    #   chunk = f"[{e['id']}] {e['title']}\n{e['sentence']}\n\n---\n\n"
    #   if len(context) + len(chunk) > max_chars:
    #       break
    #   context += chunk
    # return context
    pass


def answer(
    query: str,
    mode: str = "hybrid",
    topk: int = 5,
    config_path: str = "configs/config.yaml",
) -> Dict:
    """
    基础版 RAG：使用 retrieve + build_context。
    """
    # TODO:
    # cfg = _load_config(config_path)
    # docs = retrieve(query, topk, mode, alpha=cfg["retrieval"]["alpha"], config_path=config_path)
    # context = build_context(docs, cfg["runtime"]["max_context_chars"])
    # system_prompt / user_prompt 根据 context+query 设计
    # client = LLMClient(...)
    # ans = client.generate(system_prompt, user_prompt)
    # citations = [{"id": d["id"], "title": d["title"]} for d in docs]
    # return {"answer": ans, "citations": citations}
    pass


def enhanced_answer(
    query: str,
    mode: str = "hybrid",
    topk: int = 5,
    config_path: str = "configs/config.yaml",
) -> Dict:
    """
    v4 增强版 RAG：
    - 使用 retrieve_enhanced 获取 docs
    - 句级 evidence 抽取
    - 基于 evidence 构造上下文（或 fallback 到全文）
    """
    # TODO:
    # cfg = _load_config(config_path)
    # docs = retrieve_enhanced(query, topk, mode, config_path=config_path)
    # ev_cfg = cfg["rag"]["evidence"]
    # evidences = select_evidence_for_docs(query, docs, ev_cfg["per_doc"], ev_cfg["max_total"])
    # if cfg["rag"]["use_evidence_snippets"] and evidences:
    #   context = build_context_with_evidence(evidences, cfg["runtime"]["max_context_chars"])
    # else:
    #   context = build_context(docs, cfg["runtime"]["max_context_chars"])
    # 构造 prompt + 调用 LLMClient（同 answer）
    # 返回 {"answer": ..., "citations": ..., "evidence": evidences}
    pass


def main():
    """
    命令行：
    python src/rag.py "What is contrastive learning?" --mode hybrid --topk 5 --enhanced
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--enhanced", action="store_true")
    args = parser.parse_args()

    # TODO:
    # if args.enhanced:
    #   res = enhanced_answer(args.query, args.mode, args.topk)
    # else:
    #   res = answer(args.query, args.mode, args.topk)
    # print(res)
    pass


if __name__ == "__main__":
    main()
```

---

## 12. `src/synth_qa.py` —— 合成问答数据生成

> 与 project3 一致，这里只给简要骨架（与 v4 兼容）。

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

# from .rag import LLMClient, _load_config


def _iter_clean_docs(clean_path: str):
    """
    生成器：逐行读取 clean.jsonl，yield 单篇文档字典。
    """
    # TODO: yield json.loads(line)
    pass


def _primary_category(categories: str) -> Optional[str]:
    """
    从 categories 字符串中取出第一个标签，比如 "cs.CL stat.ML" -> "cs.CL"。
    """
    # TODO: return categories.split()[0] if categories else None
    pass


def generate_questions_for_doc(
    title: str,
    abstract: str,
    n_q: int,
    client: "LLMClient",
) -> List[str]:
    """
    输入单篇论文的 title + abstract，调用 LLM 生成 n_q 个问题。
    初期可用 mock 固定模板。
    """
    # TODO:
    # 如果使用真实 LLM：
    #   system_prompt = ...
    #   user_prompt = ...
    #   text = client.generate(system_prompt, user_prompt)
    #   解析成若干问题列表
    # 否则（mock）：
    #   return [f"What is the main idea of paper: {title}?" for _ in range(n_q)]
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
    抽样 sample_size 篇论文，对每篇生成 questions_per_doc 个问题，并写入 out_path。
    返回生成的 QA 条数。
    """
    # TODO:
    # cfg = _load_config(config_path)
    # client = LLMClient(...)
    # 收集 docs 列表（可先全部加载或流式+随机采样）
    # 对满足 category_filter 的 docs 进行抽样
    # 对每篇调用 generate_questions_for_doc(...)
    # 写入 {"q": q, "gold_ids": [id], "source": "synthetic_llm", "category": primary_category} 到 out_path
    pass


def main():
    """
    命令行：
    python src/synth_qa.py --in ./data/clean.jsonl --out ./data/synth_qa.jsonl --sample_size 500 --questions_per_doc 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--questions_per_doc", type=int, default=2)
    parser.add_argument("--category", dest="category_filter", default=None)
    args = parser.parse_args()

    # n = generate_synthetic_qa(...)
    # print(f"Generated {n} QA pairs -> {args.out_path}")
    pass


if __name__ == "__main__":
    main()
```

---

## 13. `src/eval.py` —— 离线评测（含 v4 扩展）

```python
#!/usr/bin/env python3
"""
eval.py

功能：
- 在 QA 集上评测 bm25/dense/hybrid（基础 / enhanced）模式：
  - Recall@k、MRR
  - 平均检索时间 search_ms
  - 可选生成时间 gen_ms & 端到端时间 end2end_ms
"""

import argparse
import csv
import json
import time
from statistics import mean
from typing import List, Dict

# from .retriever import retrieve, retrieve_enhanced
# from .rag import answer, enhanced_answer


def _read_qa(path: str) -> List[Dict]:
    items: List[Dict] = []
    # TODO: 逐行读取 jsonl，append 到 items
    pass


def _recall_at_k(retrieved: List[str], gold: List[str]) -> float:
    # TODO: return 1.0 if set(retrieved) ∩ set(gold) else 0.0
    pass


def _mrr_at_k(retrieved: List[str], gold: List[str]) -> float:
    # TODO:
    # for i, rid in enumerate(retrieved):
    #   if rid in gold: return 1.0 / (i+1)
    # return 0.0
    pass


def evaluate(
    qa_path: str,
    modes: List[str],
    out_csv: str,
    k: int = 5,
    include_gen: bool = False,
    use_enhanced: bool = False,
) -> None:
    """
    核心评测函数。
    """
    # TODO:
    # qa_items = _read_qa(qa_path)
    # rows = []
    # for mode in modes:
    #   recalls, mrrs, search_times, gen_times, end2end_times = [], [], [], [], []
    #   for item in qa_items:
    #       q = item["q"]; gold = item["gold_ids"]
    #       t0 = time.perf_counter()
    #       if use_enhanced:
    #           docs = retrieve_enhanced(q, k, mode)
    #       else:
    #           docs = retrieve(q, k, mode)
    #       t1 = time.perf_counter()
    #       ids = [d["id"] for d in docs]
    #       recalls.append(_recall_at_k(ids, gold))
    #       mrrs.append(_mrr_at_k(ids, gold))
    #       search_times.append((t1 - t0) * 1000)
    #       if include_gen:
    #           t2 = time.perf_counter()
    #           if use_enhanced:
    #               _ = enhanced_answer(q, mode, k)
    #           else:
    #               _ = answer(q, mode, k)
    #           t3 = time.perf_counter()
    #           gen_times.append((t3 - t2) * 1000)
    #           end2end_times.append((t3 - t0) * 1000)
    #   row = {
    #     "mode": mode,
    #     "k": k,
    #     "recall": mean(recalls) if recalls else 0.0,
    #     "mrr": mean(mrrs) if mrrs else 0.0,
    #     "search_ms": mean(search_times) if search_times else 0.0,
    #     "gen_ms": mean(gen_times) if gen_times else 0.0,
    #     "end2end_ms": mean(end2end_times) if end2end_times else 0.0,
    #   }
    #   rows.append(row)
    #
    # with open(out_csv, "w", newline="", encoding="utf-8") as f:
    #   writer = csv.DictWriter(f, fieldnames=["mode","k","recall","mrr","search_ms","gen_ms","end2end_ms"])
    #   writer.writeheader()
    #   for r in rows: writer.writerow(r)
    pass


def main():
    """
    命令行：
    python src/eval.py --qa ./data/synth_qa.jsonl --modes bm25 dense hybrid --out ./logs/metrics.csv --k 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa", dest="qa", required=True)
    parser.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    parser.add_argument("--out", dest="out_csv", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--include_gen", action="store_true")
    parser.add_argument("--use_enhanced", action="store_true")
    args = parser.parse_args()

    # evaluate(args.qa, args.modes, args.out_csv, args.k, include_gen=args.include_gen, use_enhanced=args.use_enhanced)
    # print(f"Metrics written to {args.out_csv}")
    pass


if __name__ == "__main__":
    main()
```

---

## 14. 从 0 开始的测试流程 v4（基础 + 创新点消融）

> 完整测试建议拆两层：  
> - 基础功能：确保所有文件“跑得通”；  
> - 创新模块：每个开关的 ablation 实验。

### 14.1 基础测试（类似 test3，但直接写在这里）

1. **预处理 mini 数据**

```bash
python src/ingest.py \
  --in ./data/mini_raw.jsonl \
  --out ./data/clean.jsonl \
  --max_rows 100
head -n 3 ./data/clean.jsonl
```

2. **BM25 索引构建 + 简单查询**

```bash
python src/index_bm25.py \
  --in ./data/clean.jsonl \
  --index ./data/bm25.idx

python -c "from src.index_bm25 import search_bm25; \
print(search_bm25('./data/bm25.idx', 'contrastive learning', 2))"
```

3. **稠密向量索引 + 查询**

```bash
python src/index_dense.py \
  --in ./data/clean.jsonl \
  --db ./data/chroma \
  --model bge-small-en-v1.5

python -c "from src.index_dense import search_dense; \
print(search_dense('./data/chroma', 'graph neural networks', 2, 'bge-small-en-v1.5'))"
```

4. **基础检索 `retrieve`**

```bash
python -c "from src.retriever import retrieve; \
print(retrieve('contrastive learning', 3, mode='bm25')); \
print(retrieve('graph neural networks', 3, mode='dense')); \
print(retrieve('transformer architectures', 3, mode='hybrid'))"
```

5. **RAG 基础版 `answer`（mock）**

配置中：

```yaml
generation:
  provider: mock
```

命令：

```bash
python -c "from src.rag import answer; \
print(answer('What is contrastive learning?', mode='hybrid', topk=3))"
```

预期返回 dict，有 `answer` 和 `citations`。

6. **合成 QA**

```bash
python src/synth_qa.py \
  --in ./data/clean.jsonl \
  --out ./data/synth_qa.jsonl \
  --sample_size 3 \
  --questions_per_doc 1

head -n 5 ./data/synth_qa.jsonl
```

7. **评测基础版（v3 风格）**

```bash
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes bm25 dense hybrid \
  --out ./logs/metrics_v3_baseline.csv \
  --k 5

cat ./logs/metrics_v3_baseline.csv
```

### 14.2 v4 新模块单独测试（见上一条回答中的所有 `python -c` / `python - << EOF` 片段）

- `heuristics.classify_query / choose_strategy`；
- `reranker.rerank`；
- `expansion.expand_query`；
- `snippets.sentence_split / score_sentences / select_evidence_for_docs`；
- `retrieve_enhanced`；
- `enhanced_answer`。

### 14.3 v4 增强端到端 + 消融实验（核心命令一览）

假设：

- 将所有评测结果分别输出到不同 CSV 中，便于对比。

1. **动态 alpha：off vs on**

```bash
# dynamic_alpha = false
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_dynamic_alpha_off.csv \
  --k 5

# dynamic_alpha = true
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_dynamic_alpha_on.csv \
  --k 5 \
  --use_enhanced
```

2. **Rerank：off vs on**

```bash
# rerank.enable = false
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_rerank_off.csv \
  --k 5 \
  --use_enhanced

# rerank.enable = true
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_rerank_on.csv \
  --k 5 \
  --use_enhanced
```

3. **Query Expansion：off / PRF / PRF+LLM**

```bash
# expansion.enable = false
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_expansion_off.csv \
  --k 10 \
  --use_enhanced

# expansion.enable = true, llm.enable = false
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_expansion_prf.csv \
  --k 10 \
  --use_enhanced

# expansion.enable = true, llm.enable = true
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_expansion_prf_llm.csv \
  --k 10 \
  --use_enhanced
```

4. **Evidence 高亮：off vs on**

```bash
# rag.use_evidence_snippets = false
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_evidence_off.csv \
  --k 5 \
  --include_gen \
  --use_enhanced

# rag.use_evidence_snippets = true
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_evidence_on.csv \
  --k 5 \
  --include_gen \
  --use_enhanced
```

5. **类别感知：off vs on**

```bash
# category.enable_filter = false, enable_boost = false
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_category_off.csv \
  --k 5 \
  --use_enhanced

# category.enable_filter = true, enable_boost = true
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_category_on.csv \
  --k 5 \
  --use_enhanced
```

6. **SLA：不同延迟预算**

```bash
# runtime.latency_budget_ms = 300 / 800 / 1500 各跑一次
python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes hybrid \
  --out ./logs/ablation_sla_300ms.csv \
  --k 5 \
  --include_gen \
  --use_enhanced
```

通过对比这些 CSV 中的 `recall/mrr/search_ms/gen_ms/end2end_ms`，以及手动打印部分 `enhanced_answer` 的 `evidence` 列表，你就能完整展示：

- 从 0 搭建的基础 RAG 系统是可用的；
- 每个 v4 创新点（动态 alpha / rerank / expansion / evidence / category / SLA）都能被独立打开/关闭；
- 并且可以用“消融实验”的方式，观察它们对检索效果与延迟的具体影响。

```
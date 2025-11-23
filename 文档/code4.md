```markdown
# 从零实现指南（code4 文档，完整版）  
—— 从 0 搭建 + 实现“多策略增强 RAG 学术问答系统”及完整测试流程

> 使用方式：  
> - 把本文件当成“从零实现 + 自测 + 消融实验”的总说明。  
> - 每个模块都有两部分：  
>   - 先用中文说明“要干什么”；  
>   - 再给一个可以直接复制到代码文件里的 **Python 伪代码骨架**（函数签名 + 注释 + `pass` / `TODO`）。  
> - 全流程覆盖：基础 RAG 系统 + 所有增强模块 + 完整测试与消融实验命令。

---

## 0. 总体实现顺序（建议）

1. **准备项目结构与环境**
   - 新建目录 [5020-RAG/](cci:7://file:///mnt/sdb/dongpeijie/workspace_sxy/5020-RAG:0:0-0:0)；
   - 建立虚拟环境，安装 `requirements.txt` 依赖。
2. **准备配置文件 `configs/config.yaml`**
   - 统一管理数据路径、索引路径、模型名、检索策略和运行参数。
3. **实现基础数据与索引模块**
   - `src/ingest.py`：从原始 arXiv JSONL 抽取核心字段 → `clean.jsonl`；
   - `src/index_bm25.py`：构建 BM25 索引；
   - `src/index_dense.py`：构建稠密向量索引（Chroma）。
4. **实现基础检索与 RAG**
   - `src/retriever.py`：基础检索接口 `retrieve`（bm25/dense/hybrid）；
   - `src/rag.py`：RAG 主流程（`answer` + `LLMClient`）。
5. **实现合成 QA 与评测**
   - `src/synth_qa.py`：合成问答数据生成；
   - `src/eval.py`：离线评测（Recall/MRR/Latency）。
6. **实现增强模块**
   - `src/heuristics.py`：查询启发式、动态 alpha、SLA 策略；
   - `src/reranker.py`：Cross-Encoder 重排；
   - `src/expansion.py`：PRF + LLM 查询扩展；
   - `src/snippets.py`：句级 evidence 抽取与高亮；
   - 扩展 `retriever.py` 增加 `retrieve_enhanced`；
   - 扩展 `rag.py` 增加 `enhanced_answer`。
7. **实现并执行完整测试与消融实验**
   - 用 mini 数据和合成 QA 做基础功能测试；
   - 针对每个增强点做“开/关”消融实验，并用 `logs/*.csv` 对比指标。

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

在项目根目录创建/编辑 `requirements.txt`：

```text
pyyaml
tqdm
rank-bm25
chromadb
sentence-transformers
openai
python-dotenv   # 用于从 .env 自动加载 OPENAI_API_KEY / OPENAI_BASE_URL
# 如需交互式测试，可加：ipython / jupyter（可选）
```

安装依赖：

```bash
pip install -U -r requirements.txt
```

---

## 2. 配置文件 `configs/config.yaml`

### 2.1 要做什么

- 集中管理数据路径、索引路径、模型配置；
- 控制检索模式和增强模块开关；
- 配置 RAG 生成、类别策略和延迟预算等参数。

在 `configs/` 中创建 `config.yaml`，示例骨架：

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
  alpha: 0.5           # 初始固定 alpha
  dynamic_alpha: true  # 是否启用自适应 alpha
  expansion:
    enable: false      # 查询扩展总开关
    prf:
      enable: true
      m_docs: 5
      top_terms: 10
    llm:
      enable: false    # 初期可关掉 LLM 改写
      n_variants: 3

rerank:
  enable: false
  model: bge-reranker-base
  topn: 50
  batch_size: 16

rag:
  use_evidence_snippets: true
  evidence:
    per_doc: 2
    max_total: 10
    method: bm25       # 句级打分方式：bm25 | embedding

generation:
  provider: mock           # mock | openai | ollama（或其他 OpenAI 兼容服务）
  model: gpt-4.1-mini      # 示例：OpenAI 官方模型名，可按需要修改
  max_tokens: 512

  # 使用真实 LLM 时的说明（代码里会从环境变量读取）：
  # - OPENAI_API_KEY  : 必填，你的 API Key
  # - OPENAI_BASE_URL : 选填，OpenAI 兼容服务地址
  #   - 官方 OpenAI 可设为: https://api.openai.com/v1
  #   - 其他厂商按其文档填写

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

- 从超大的 arXiv JSONL 流式读取；
- 解析 JSON，抽取核心字段；
- 输出精简版 `clean.jsonl`，便于后续索引与检索。

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
    # # 如需，可做一些正则清洗
    # return text.strip()
    pass


def stream_clean_arxiv(input_path: str, output_path: str, max_rows: Optional[int] = None) -> None:
    """
    核心函数：流式读取 input_path，写出精简字段到 output_path。
    - max_rows 用于调试，只处理前 N 行。
    """
    # 1. 打开输入/输出文件（encoding="utf-8"）
    # 2. for i, line in enumerate(f): 逐行读取
    # 3. 对每一行：
    #    - strip 去掉首尾空白
    #    - 空行跳过
    #    - try/except json.loads 防止坏行
    #    - 抽取 id/title/abstract/categories/versions
    #    - created = _parse_created(versions)
    #    - abstract = _clean_abstract(abstract)
    #    - new_rec = {...}
    #    - 用 json.dump(new_rec, out_file, ensure_ascii=False) + 换行
    # 4. 如果 max_rows 非空，处理到 N 行后 break
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
- 构建 BM25 文本索引；
- 序列化保存到 `bm25.idx`；
- 提供 `search_bm25` 查询接口。

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

# from rank_bm25 import BM25Okapi  # 实际使用时导入


def _tokenize(text: str) -> List[str]:
    """
    简单英文分词：
    - 转小写
    - 用正则提取连续字母/数字
    """
    # TODO:
    # text = text.lower()
    # return re.findall(r"[a-z0-9]+", text)
    pass


def build_bm25_index(clean_path: str, index_path: str) -> int:
    """
    从 clean.jsonl 构建 BM25 索引，并保存到 index_path。
    返回文档数量。
    """
    docs: List[Dict] = []
    tokens: List[List[str]] = []

    # TODO:
    # with open(clean_path, "r", encoding="utf-8") as f:
    #     for line in f:
    #         rec = json.loads(line)
    #         text = (rec.get("title","") + "\n" + rec.get("abstract","")).strip()
    #         docs.append({"id": rec["id"], "title": rec["title"], "abstract": rec["abstract"], "text": text})
    #         tokens.append(_tokenize(text))
    # bm25 = BM25Okapi(tokens)
    # with open(index_path, "wb") as f_out:
    #     pickle.dump({"docs": docs, "tokens": tokens}, f_out)
    # return len(docs)
    pass


def search_bm25(index_path: str, query: str, topk: int = 5) -> List[Dict]:
    """
    用 BM25 索引搜索 query，返回 Top-k 结果列表。
    每条包含：id/title/abstract/text/score。
    """
    # TODO:
    # with open(index_path, "rb") as f:
    #     data = pickle.load(f)
    # docs, tokens = data["docs"], data["tokens"]
    # bm25 = BM25Okapi(tokens)
    # q_tokens = _tokenize(query)
    # scores = bm25.get_scores(q_tokens)
    # 根据 scores 取前 topk 的索引，组装结果列表
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

    # n = build_bm25_index(args.in_path, args.index_path)
    # print(f"Built BM25 index with {n} docs -> {args.index_path}")
    pass


if __name__ == "__main__":
    main()
```

---

## 5. `src/index_dense.py` —— 稠密向量索引（Chroma）

### 5.1 要做什么

- 从 `clean.jsonl` 加载文档；
- 使用 `SentenceTransformer` 编码为向量；
- 写入 Chroma 持久化向量库；
- 提供 `search_dense` 接口。

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
    #         text = (rec.get("title","") + "\n" + rec.get("abstract","")).strip()
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
    # client = chromadb.PersistentClient(path=db_path)
    # 如果 collection 已存在可删除再重建
    # docs = _load_docs(clean_path)
    # model = SentenceTransformer(model_name)
    # 按 batch_size 遍历 docs:
    #   texts = [d["text"] for d in batch]
    #   emb = model.encode(texts, normalize_embeddings=True)
    #   ids = [d["id"] for d in batch]
    #   metas = [{"title": d["title"]} for d in batch]
    #   collection.add(ids=ids, documents=texts, metadatas=metas, embeddings=emb)
    # return len(docs)
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
    每条包含：id/title/text/score（可用 1 - 距离 作为 score）。
    """
    # TODO:
    # client = chromadb.PersistentClient(path=db_path)
    # col = client.get_collection(collection)
    # model = SentenceTransformer(model_name)
    # q_emb = model.encode([query], normalize_embeddings=True)
    # result = col.query(query_embeddings=q_emb, n_results=topk, include=["metadatas","documents","distances"])
    # 把 ids/documents/metadatas/distances 整理成统一 dict 列表
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

    # n = embed_and_build_vector_db(args.in_path, args.db_path, args.model, args.collection)
    # print(f"Built dense DB with {n} docs -> {args.db_path} [{args.collection}]")
    pass


if __name__ == "__main__":
    main()
```

---

## 6. `src/heuristics.py` —— 查询/类别/SLA 启发式

### 6.1 要做什么

- 对 query 提取简单特征并分类（术语型/语义型/混合型），给出建议 `alpha`；
- 可选：根据 query 粗略预测一个类别前缀（如 `cs.*`）；
- 根据延迟预算 `latency_budget_ms` 返回一组策略开关。

### 6.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
heuristics.py

功能：
- 对 query 做简单特征分析，输出查询类型和建议 alpha。
- 可选：对 query 做类别预测。
- 根据延迟预算给出检索/重排/扩展策略配置。
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
    # return {"len_chars": len_chars, "len_tokens": len(tokens), "num_digits": num_digits, "num_punct": num_punct}
    pass


def classify_query(q: str) -> Dict:
    """
    将 query 粗分为 'term' / 'semantic' / 'mixed' 三类，并给出建议 alpha。
    返回示例：
    {
      "type": "term",
      "alpha": 0.2,
      "features": {...}
    }
    """
    # TODO:
    # feat = _basic_query_features(q)
    # 结合长度/数字符号比例等判断类型：
    # if feat["len_tokens"] <= 4 and (feat["num_digits"]+feat["num_punct"]) 较高 → term
    # elif feat["len_tokens"] >= 8 → semantic
    # else → mixed
    # 根据类型设定 alpha
    pass


def predict_query_category(q: str) -> Optional[str]:
    """
    可选：根据 query 里的关键词，粗略预测一个类别，比如 "cs.CL"。
    初期可以简单规则或返回 None。
    """
    # TODO: 可先 return None
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

## 7. `src/retriever.py` —— 基础 + 增强检索入口

### 7.1 要做什么

- 实现基础 `retrieve`：支持 `bm25 / dense / hybrid` 三种模式；
- 实现 `_apply_category_logic`：按类别过滤/加权；
- 实现 `retrieve_enhanced`：结合 dynamic alpha、query expansion、rerank、类别和 SLA 策略。

### 7.2 伪代码骨架（重点放在增强部分）

```python
#!/usr/bin/env python3
"""
retriever.py

功能：
- 基础检索接口 retrieve。
- 增强检索接口 retrieve_enhanced：支持动态 alpha / 查询扩展 / rerank / 类别感知 / SLA。
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
    """从 YAML 文件加载配置。"""
    # TODO:
    # with open(path, "r", encoding="utf-8") as f:
    #     return yaml.safe_load(f)
    pass


def _norm(scores: List[float]) -> List[float]:
    """min-max 归一化到 0~1。"""
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
    基础检索入口。
    - mode: "bm25" / "dense" / "hybrid"
    - hybrid: final_score = alpha * dense_score + (1-alpha) * bm25_score
    """
    # TODO:
    # cfg = _load_config(config_path)
    # bm25_path = cfg["bm25"]["index_path"]
    # dcfg = cfg["dense"]
    # if mode == "bm25": return search_bm25(bm25_path, query, topk)
    # if mode == "dense": return search_dense(dcfg["db"], query, topk, dcfg["model"], dcfg["collection"])
    # if mode == "hybrid":
    #   bm25_res = search_bm25(...)
    #   dense_res = search_dense(...)
    #   # 归一化两边分数，按 alpha 融合，去重合并
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
    #       if primary_cat(d).startswith(query_cat.split(".")[0]):
    #           d["score"] = d.get("score", 0.0) * (1.0 + bf)
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
    # TODO:
    # 1. cfg = _load_config(config_path)
    # 2. 若 latency_budget_ms 不为空：
    #       strategy = choose_strategy(latency_budget_ms)
    #       用 strategy 覆盖 mode / enable_rerank / enable_expansion
    # 3. 若 use_dynamic_alpha 且 alpha is None:
    #       info = classify_query(query); alpha = info["alpha"]
    # 4. queries = [query]
    #    若 enable_expansion: queries = expand_query(query, client=None, ...)
    # 5. 对每个 q' in queries:
    #       使用基础 retrieve(q', topN, mode, alpha) 做初排
    #       将结果合并到 {id: doc} 映射（score 累加或取最大）
    # 6. 将映射转为列表 docs_all
    #    query_cat = predict_query_category(query)
    #    docs_all = _apply_category_logic(docs_all, query_cat, cfg["category"])
    # 7. 若 enable_rerank:
    #       topN = cfg["rerank"]["topn"]
    #       取前 topN 调用 rerank_docs(query, topN_docs, model_name, topk)
    #       返回 rerank 后结果
    #    否则：
    #       按 doc["score"] 排序，取前 topk 返回
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

### 8.1 要做什么

- 加载 CrossEncoder 模型；
- 对初排结果中的文档与 query 两两组合进行打分；
- 按新分数重排，产出更精确的 Top-k。

### 8.2 伪代码骨架

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
    - docs 每个元素至少应有 "text" 字段（或 title+abstract 拼成 text）。
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

## 9. `src/expansion.py` —— 查询扩展（PRF + LLM）

### 9.1 要做什么

- 基于 PRF，从初排 Top-M 文档中抽取高频关键词；
- 使用 LLM 改写 query，生成若干等价问法；
- 综合构造多种 query 变体，与基础检索结合。

### 9.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
expansion.py

功能：
- PRF：从初排结果中做伪相关反馈，提取高频关键词。
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
    使用当前检索后端，对 query 做一次初排，
    从 top-m 文档中统计关键词，返回若干扩展词。
    """
    # TODO:
    # 1. 用基础 retrieve(query, topk=m_docs, mode=retriever_cfg["mode"], ...) 获取 top-m 文档
    # 2. 拼接这些文档的 text，做简单分词
    # 3. 统计词频，去掉停用词/过短词，按频率排序
    # 4. 返回前 top_terms 个词
    pass


def llm_expand(
    query: str,
    client: "LLMClient",
    n_variants: int = 3,
) -> List[str]:
    """
    使用 LLM 把 query 改写为 n 个等价或更具体的问法。
    """
    # TODO:
    # system_prompt = "你是检索查询改写助手..."
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
    返回列表至少包含原始 query 本身。
    """
    # TODO:
    # variants = [query]
    # if use_prf and retriever_cfg is not None:
    #   terms = prf_terms(query, retriever_cfg, **(prf_cfg or {}))
    #   if terms:
    #       expanded_q = query + " " + " ".join(terms[:3])
    #       variants.append(expanded_q)
    # if use_llm and client is not None:
    #   llm_qs = llm_expand(query, client, **(llm_cfg or {}))
    #   variants.extend(llm_qs)
    # 去重：
    # variants = list(dict.fromkeys(variants))
    # return variants
    pass
```

---

## 10. `src/snippets.py` —— 句级证据抽取

### 10.1 要做什么

- 将文档文本分句；
- 对每句与 query 计算相关性得分；
- 为每篇文档选出若干 Top 句；
- 汇总得到 evidence 列表供 RAG 使用和展示。

### 10.2 伪代码骨架

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
    简单的分句函数：
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
    返回结构示例：
    [
      {"sentence": "...", "score": 0.9},
      ...
    ]
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
    #   # 可扩展为基于向量相似度的句级打分
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
    返回结构示例：
    [
      {"id": doc_id, "title": title, "sentence": s, "score": 0.95},
      ...
    ]
    """
    evidences: List[Dict] = []
    # TODO:
    # for d in docs:
    #   ss = score_sentences(query, d, max_sentences=per_doc)
    #   for item in ss:
    #       evidences.append({
    #         "id": d.get("id"),
    #         "title": d.get("title"),
    #         "sentence": item["sentence"],
    #         "score": item["score"],
    #       })
    # evidences_sorted = sorted(evidences, key=lambda x: x["score"], reverse=True)
    # return evidences_sorted[:max_total]
    pass
```

---

## 11. `src/rag.py` —— RAG 基础与增强

### 11.1 要做什么

- 提供基础 RAG 接口 `answer`：`retrieve + build_context + LLMClient.generate`；
- 提供增强版 `enhanced_answer`：使用 `retrieve_enhanced` + 句级 evidence 构造上下文，并返回 evidence 列表。

### 11.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
rag.py

功能：
- 基础 RAG 流程：检索 + 拼接上下文 + 调用 LLM。
- 增强版：结合 retrieve_enhanced + 句级 evidence 构造上下文，返回 evidence 信息。
"""

import argparse
from typing import List, Dict

# import yaml
# from openai import OpenAI
# from dotenv import load_dotenv
# from .retriever import retrieve, retrieve_enhanced
# from .snippets import select_evidence_for_docs

# 加载项目根目录 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL
# 建议在真实实现中取消注释：
# load_dotenv()

class LLMClient:
    """
    大模型调用封装：
    - provider="mock"：返回伪造答案；
    - provider="openai"/"ollama"：使用 OpenAI 兼容接口。
    """

    def __init__(self, provider: str, model: str, base_url: str = None, api_key: str = None, max_tokens: int = 512):
        # TODO: 保存 provider/model/max_tokens，必要时初始化 OpenAI 客户端
        pass

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        调用模型生成回答。
        """
        # TODO:
        # if self.provider == "mock":
        #   return user_prompt[:100] + " ... [Mock Answer]"
        # else:
        #   resp = self.client.chat.completions.create(...)
        #   return resp.choices[0].message.content
        pass


def _load_config(path: str) -> Dict:
    # TODO: yaml.safe_load
    pass


def build_context(docs: List[Dict], max_chars: int) -> str:
    """
    基础版：使用文档的 title+abstract 拼接上下文。
    """
    # TODO:
    # context = ""
    # for d in docs:
    #   chunk = f"[{d['id']}] {d['title']}\n{d.get('abstract','')}\n\n---\n\n"
    #   if len(context) + len(chunk) > max_chars:
    #       break
    #   context += chunk
    # return context
    pass


def build_context_with_evidence(
    evidences: List[Dict],
    max_chars: int = 4000,
) -> str:
    """
    使用句级 evidence 构造上下文。
    """
    # TODO:
    # ev_sorted = sorted(evidences, key=lambda x: x["score"], reverse=True)
    # context = ""
    # for e in ev_sorted:
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
    基础 RAG：使用 retrieve + build_context。
    """
    # TODO:
    # cfg = _load_config(config_path)
    # docs = retrieve(query, topk, mode, alpha=cfg["retrieval"]["alpha"], config_path=config_path)
    # context = build_context(docs, cfg["runtime"]["max_context_chars"])
    # system_prompt = "你是一个学术问答助手..."
    # user_prompt = f"问题：{query}\n\n参考文献摘要：\n{context}\n\n请基于以上内容回答问题，并使用自然语言总结。"
    # gen_cfg = cfg["generation"]
    # client = LLMClient(gen_cfg["provider"], gen_cfg["model"], max_tokens=gen_cfg["max_tokens"])
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
    增强版 RAG：
    - 使用 retrieve_enhanced 获取 docs
    - 句级 evidence 抽取
    - 基于 evidence 或全文构造上下文
    - 返回 answer + citations + evidence
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
    # system_prompt = "你是一个学术问答助手..."
    # user_prompt = f"问题：{query}\n\n以下是相关论文的关键信息：\n{context}\n\n请回答问题，并尽量提及相关论文。"
    # gen_cfg = cfg["generation"]
    # client = LLMClient(gen_cfg["provider"], gen_cfg["model"], max_tokens=gen_cfg["max_tokens"])
    # ans = client.generate(system_prompt, user_prompt)
    # citations = [{"id": d["id"], "title": d["title"]} for d in docs]
    # return {"answer": ans, "citations": citations, "evidence": evidences}
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

### 12.1 要做什么

- 从 `clean.jsonl` 抽样论文；
- 利用 LLM 或模板为每篇论文生成若干问题；
- 写入 `synth_qa.jsonl`，用于检索评测。

### 12.2 伪代码骨架

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
    # TODO:
    # with open(clean_path, "r", encoding="utf-8") as f:
    #   for line in f:
    #       yield json.loads(line)
    pass


def _primary_category(categories: str) -> Optional[str]:
    """
    从 categories 字符串中取出第一个标签，比如 "cs.CL stat.ML" -> "cs.CL"。
    """
    # TODO:
    # return categories.split()[0] if categories else None
    pass


def generate_questions_for_doc(
    title: str,
    abstract: str,
    n_q: int,
    client: "LLMClient",
) -> List[str]:
    """
    输入单篇论文的 title + abstract，调用 LLM 生成 n_q 个问题。
    初期可用简单 mock 实现。
    """
    # TODO:
    # 若使用真实 LLM：
    #   system_prompt = "你是一个研究助教，请根据论文标题和摘要生成若干可被摘要回答的问题..."
    #   user_prompt = f"标题: {title}\n摘要: {abstract}\n请生成 {n_q} 个问题，每行一个。"
    #   text = client.generate(system_prompt, user_prompt)
    #   questions = [ln.strip() for ln in text.splitlines() if ln.strip()]
    #   return questions[:n_q]
    # 若暂时不用 LLM，可先 mock：
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
    # gen_cfg = cfg["generation"]
    # client = LLMClient(gen_cfg["provider"], gen_cfg["model"], max_tokens=gen_cfg["max_tokens"])
    # docs = list(_iter_clean_docs(clean_path))
    # 如果 category_filter 不为空，只保留主类别匹配的文档
    # random.shuffle(docs)
    # 取前 sample_size 篇
    # count = 0
    # with open(out_path, "w", encoding="utf-8") as f_out:
    #   for rec in sampled_docs:
    #       cat = _primary_category(rec.get("categories",""))
    #       qs = generate_questions_for_doc(rec["title"], rec["abstract"], questions_per_doc, client)
    #       for q in qs:
    #           item = {
    #             "q": q,
    #             "gold_ids": [rec["id"]],
    #             "source": "synthetic_llm",
    #             "category": cat
    #           }
    #           json.dump(item, f_out, ensure_ascii=False)
    #           f_out.write("\n")
    #           count += 1
    # return count
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

    # n = generate_synthetic_qa(args.in_path, args.out_path, args.sample_size, args.questions_per_doc, args.category_filter)
    # print(f"Generated {n} QA pairs -> {args.out_path}")
    pass


if __name__ == "__main__":
    main()
```

---

## 13. `src/eval.py` —— 离线评测（含检索/生成/端到端延迟）

### 13.1 要做什么

- 在给定 QA 集上评测不同检索模式或策略组合；
- 计算 Recall@k、MRR、平均检索时间、生成时间和端到端时间；
- 将结果写入 CSV，方便后续制表和画图。

### 13.2 伪代码骨架

```python
#!/usr/bin/env python3
"""
eval.py

功能：
- 在 QA 集上评测 bm25/dense/hybrid 以及增强检索模式：
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
    """
    读取 QA JSONL 文件，返回列表。
    """
    items: List[Dict] = []
    # TODO:
    # with open(path, "r", encoding="utf-8") as f:
    #   for line in f:
    #       line = line.strip()
    #       if not line: continue
    #       items.append(json.loads(line))
    # return items
    pass


def _recall_at_k(retrieved: List[str], gold: List[str]) -> float:
    """
    Recall@k 的单样本版本：命中则为 1，否则为 0。
    """
    # TODO:
    # return 1.0 if set(retrieved) & set(gold) else 0.0
    pass


def _mrr_at_k(retrieved: List[str], gold: List[str]) -> float:
    """
    MRR 的单样本版本：第一个命中的倒数排名。
    """
    # TODO:
    # for i, rid in enumerate(retrieved):
    #   if rid in gold:
    #       return 1.0 / (i+1)
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
    #       q = item.get("q") or ""
    #       gold = item.get("gold_ids") or []
    #       t0 = time.perf_counter()
    #       if use_enhanced:
    #           docs = retrieve_enhanced(q, k, mode)
    #       else:
    #           docs = retrieve(q, k, mode)
    #       t1 = time.perf_counter()
    #       ids = [d.get("id") for d in docs]
    #       recalls.append(_recall_at_k(ids, gold))
    #       mrrs.append(_mrr_at_k(ids, gold))
    #       search_ms = (t1 - t0) * 1000.0
    #       search_times.append(search_ms)
    #       gen_ms = 0.0
    #       if include_gen:
    #           g0 = time.perf_counter()
    #           if use_enhanced:
    #               _ = enhanced_answer(q, mode, k)
    #           else:
    #               _ = answer(q, mode, k)
    #           g1 = time.perf_counter()
    #           gen_ms = (g1 - g0) * 1000.0
    #       gen_times.append(gen_ms)
    #       end2end_times.append(search_ms + gen_ms)
    #   row = {
    #     "mode": mode,
    #     "k": k,
    #     "recall": round(mean(recalls), 4) if recalls else 0.0,
    #     "mrr": round(mean(mrrs), 4) if mrrs else 0.0,
    #     "search_ms": round(mean(search_times), 2) if search_times else 0.0,
    #     "gen_ms": round(mean(gen_times), 2) if gen_times else 0.0,
    #     "end2end_ms": round(mean(end2end_times), 2) if end2end_times else 0.0,
    #   }
    #   rows.append(row)
    #
    # with open(out_csv, "w", newline="", encoding="utf-8") as f:
    #   writer = csv.DictWriter(f, fieldnames=["mode","k","recall","mrr","search_ms","gen_ms","end2end_ms"])
    #   writer.writeheader()
    #   for r in rows:
    #       writer.writerow(r)
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

## 14. 从 0 开始的测试与消融流程（概要）

> 这一节在 `test4` 文档中已经详细展开，这里简要列出关键命令与思路，方便你确认实现是否完整。

### 14.1 基础功能测试（mini 数据）

1. **预处理 mini 数据**

```bash
python src/ingest.py \
  --in ./data/mini_raw.jsonl \
  --out ./data/clean.jsonl \
  --max_rows 100

head -n 3 ./data/clean.jsonl
```

2. **BM25 索引 + 查询**

```bash
python src/index_bm25.py \
  --in ./data/clean.jsonl \
  --index ./data/bm25.idx

python -c "from src.index_bm25 import search_bm25; \
print(search_bm25('./data/bm25.idx', 'contrastive learning', 2))"
```

3. **稠密索引 + 查询**

```bash
python src/index_dense.py \
  --in ./data/clean.jsonl \
  --db ./data/chroma \
  --model bge-small-en-v1.5

python -c "from src.index_dense import search_dense; \
print(search_dense('./data/chroma', 'graph neural networks', 2, 'bge-small-en-v1.5'))"
```

4. **基础检索接口**

```bash
python -c "from src.retriever import retrieve; \
print(retrieve('contrastive learning', 3, mode='bm25')); \
print(retrieve('graph neural networks', 3, mode='dense')); \
print(retrieve('transformer architectures', 3, mode='hybrid'))"
```

5. **基础 RAG 答案（mock）**

```bash
python -c "from src.rag import answer; \
print(answer('What is contrastive learning?', mode='hybrid', topk=3))"
```

> 如果此时 `configs/config.yaml` 中 `generation.provider: mock`，
> 上面的命令会走 mock 模式，不会发真实网络请求，只用于验证检索和上下文构建逻辑。

#### 14.1 补充：切换为真实 OpenAI / 兼容 API（使用 .env 管理 Key）

1. **在 `configs/config.yaml` 中修改 `generation` 配置**

   ```yaml
   generation:
     provider: openai           # 从 mock 改成 openai
     model: gpt-4.1-mini        # 或你的 OpenAI 兼容模型名
     max_tokens: 512

2. 在项目根目录 5020-RAG/ 创建 .env 文件 文件路径示例：
    5020-RAG/
    ├── .env
    ├── configs/
    ├── src/
    └── ...
    .env 内容示例：
    OPENAI_API_KEY=sk-xxxxxx_your_key
    OPENAI_BASE_URL=https://api.openai.com/v1
    OPENAI_API_KEY：必填，你的 API Key
    OPENAI_BASE_URL：可改成你使用的 OpenAI 兼容服务地址
    确保代码会加载 .env
    在 requirements.txt 中已经包含 python-dotenv；
    在 src/rag.py 的真实实现里，按照本文件第 11 节伪代码，在文件顶部调用：

    from dotenv import load_dotenv
    load_dotenv()
    这样程序启动时会自动把 .env 中的 OPENAI_API_KEY / OPENAI_BASE_URL 注入到环境变量，供 LLMClient 通过 os.getenv 读取。
    再次运行基础 RAG 测试命令
    bash
    python -c "from src.rag import answer; \
    print(answer('What is contrastive learning?', mode='hybrid', topk=3))"
    此时：
    检索模块使用本地 BM25 / 稠密索引找到相关摘要；
    LLMClient 使用 .env 中的 Key 调用真实的 OpenAI / 兼容 LLM；
    终端输出的是“真实模型生成的答案 + 引用论文列表”。
### 14.2 合成 QA 与基础评测

```bash
python src/synth_qa.py \
  --in ./data/clean.jsonl \
  --out ./data/synth_qa.jsonl \
  --sample_size 50 \
  --questions_per_doc 1

python src/eval.py \
  --qa ./data/synth_qa.jsonl \
  --modes bm25 dense hybrid \
  --out ./logs/metrics_baseline.csv \
  --k 5

cat ./logs/metrics_baseline.csv
```

### 14.3 增强模块与消融实验（示例命令）

- 混合检索：比较 `mode=bm25` vs `mode=hybrid` 的影响；
- Rerank：比较 `rerank.enable=false` vs `true`；
- Query Expansion：比较 `expansion.enable=false` / `PRF-only` / `PRF+LLM`；
- Evidence：比较 `rag.use_evidence_snippets=false` vs `true`；
- Category：比较类别过滤开关；
- SLA：设置不同的 `latency_budget_ms` 并比较 `end2end_ms` 与 `recall/mrr`。

每个消融只需：

1. 修改 `config.yaml` 中对应开关；  
2. 运行一次 `eval.py` 输出到不同的 CSV；  
3. 对比各 CSV 中的指标，即可在报告/展示中引用。

---

这样，这个 `code4` 文档就从 0 描述了整个项目的实现步骤、各模块伪代码骨架以及测试/消融思路，不依赖任何“v3 基础”叙述，作为一个独立项目的实现指南即可。
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag.py

功能概述：
- 提供基础 RAG 接口 `answer`：检索 -> 构造上下文 -> 调用 LLM 生成答案。
- 提供增强版 `enhanced_answer`：使用增强检索 + 句级 evidence 构造上下文，返回 evidence。
- 封装 `LLMClient`：支持 provider="mock" 或 OpenAI 兼容接口（通过 .env 加载 Key）。

实现思路：
- 配置统一从 configs/config.yaml 读取（路径、模型、开关与预算）。
- `LLMClient`：
  - mock：直接返回带有占位提示的文本；
  - openai/兼容：使用 `openai` 包的 `OpenAI` 客户端调用 `/v1/chat/completions` 接口。
- 上下文构造：
  - 基础：拼接 id + title + abstract 片段；
  - 增强：优先用句级 evidence 列表，按分数降序拼接；
- 返回结构：{"answer": str, "citations": [{id,title}], ["evidence": [...] ]}。

主要函数/类：
- `LLMClient(provider, model, base_url, api_key, max_tokens)`
- `_load_config(path) -> Dict`
- `build_context(docs, max_chars) -> str`
- `build_context_with_evidence(evidences, max_chars) -> str`
- `answer(query, mode, topk, config_path) -> Dict`
- `enhanced_answer(query, mode, topk, config_path) -> Dict`

测试/使用：
- 基础（mock）：
  python -c "from src.rag import answer; print(answer('What is contrastive learning?', topk=3))"
- 增强（mock）：
  python -c "from src.rag import enhanced_answer; print(enhanced_answer('GNN for text classification', topk=3))"
- 命令行：
  python src/rag.py "What is contrastive learning?" --mode hybrid --topk 5 --enhanced

注意事项：
- 若要使用真实 LLM，请在项目根目录创建 .env，包含 OPENAI_API_KEY 与可选的 OPENAI_BASE_URL；
- 并在 configs/config.yaml 中将 generation.provider 改为 openai 或兼容服务名。
"""

import argparse
import os
from typing import List, Dict

import yaml
from dotenv import load_dotenv

# 允许脚本直接运行时导入兄弟模块
try:  # 优先包内相对导入
    from .retriever import retrieve, retrieve_enhanced
    from .snippets import select_evidence_for_docs
except Exception:  # 当以脚本运行时回退到绝对导入
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.retriever import retrieve, retrieve_enhanced  # type: ignore
    from src.snippets import select_evidence_for_docs  # type: ignore


# 可选加载 .env（便于 openai 兼容接口读取 Key）
load_dotenv()

try:
    from openai import OpenAI  # type: ignore
except Exception:  # openai 可能未安装，但 mock 模式不需要
    OpenAI = None  # type: ignore


class LLMClient:
    """大模型调用封装。

    - provider="mock"：直接返回用户提示的片段，便于离线调试；
    - provider="openai" 或其他 OpenAI 兼容：使用 OpenAI 官方 SDK。
    """

    def __init__(self, provider: str, model: str, base_url: str = None, api_key: str = None, max_tokens: int = 512):
        self.provider = (provider or "mock").lower()
        self.model = model
        self.max_tokens = int(max_tokens or 512)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.provider != "mock" and OpenAI is not None:
            try:
                # OpenAI SDK 将自动从环境变量读取 Key；也可显式传入
                if self.base_url or self.api_key:
                    self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                else:
                    self.client = OpenAI()
            except Exception:
                self.client = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """调用模型生成回答。"""
        if self.provider == "mock" or self.client is None:
            text = (user_prompt or "")
            if len(text) > 200:
                text = text[:200] + "..."
            return f"[MOCK ANSWER based on provided context]\n{text}"

        # OpenAI 兼容接口
        try:
            resp = self.client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content  # type: ignore
        except Exception as e:
            return f"[LLM ERROR] {e}"


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_context(docs: List[Dict], max_chars: int) -> str:
    """使用文档的 id + title + abstract 拼接上下文。"""
    context = ""
    for d in docs:
        chunk = f"[{d.get('id')}] {d.get('title','')}\n{d.get('abstract') or d.get('text','')}\n\n---\n\n"
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk
    return context


def build_context_with_evidence(evidences: List[Dict], max_chars: int = 4000) -> str:
    """使用句级 evidence 构造上下文，按分数降序拼接。"""
    ev_sorted = sorted(evidences or [], key=lambda x: x.get("score", 0.0), reverse=True)
    context = ""
    for e in ev_sorted:
        chunk = f"[{e.get('id')}] {e.get('title','')}\n{e.get('sentence','')}\n\n---\n\n"
        if len(context) + len(chunk) > max_chars:
            break
        context += chunk
    return context


def answer(query: str, mode: str = "hybrid", topk: int = 5, config_path: str = "configs/config.yaml") -> Dict:
    """基础 RAG：检索 + 构造上下文 + 调用 LLM。"""
    cfg = _load_config(config_path)
    docs = retrieve(query, topk, mode, alpha=cfg["retrieval"].get("alpha", 0.5), config_path=config_path)
    context = build_context(docs, int(cfg["runtime"].get("max_context_chars", 6000)))
    system_prompt = "你是一个学术问答助手，请基于给定的参考内容进行严谨回答，尽量引用论文编号。"
    user_prompt = f"问题：{query}\n\n参考文献摘要：\n{context}\n请基于以上内容回答问题，若资料不足请明确说明。"

    gen_cfg = cfg.get("generation", {})
    client = LLMClient(gen_cfg.get("provider", "mock"), gen_cfg.get("model", "gpt-4.1-mini"), max_tokens=int(gen_cfg.get("max_tokens", 512)))
    ans = client.generate(system_prompt, user_prompt)
    citations = [{"id": d.get("id"), "title": d.get("title")} for d in docs]
    return {"answer": ans, "citations": citations}


def enhanced_answer(query: str, mode: str = "hybrid", topk: int = 5, config_path: str = "configs/config.yaml") -> Dict:
    """增强版 RAG：增强检索 + 句级 evidence 构造上下文。"""
    cfg = _load_config(config_path)
    docs = retrieve_enhanced(query, topk, mode, config_path=config_path)
    ev_cfg = cfg.get("rag", {}).get("evidence", {})
    evidences = select_evidence_for_docs(
        query,
        docs,
        int(ev_cfg.get("per_doc", 2)),
        int(ev_cfg.get("max_total", 10)),
    )
    if cfg.get("rag", {}).get("use_evidence_snippets", True) and evidences:
        context = build_context_with_evidence(evidences, int(cfg["runtime"].get("max_context_chars", 6000)))
    else:
        context = build_context(docs, int(cfg["runtime"].get("max_context_chars", 6000)))

    system_prompt = "你是一个学术问答助手，请基于给定的证据片段进行严谨回答，尽量引用论文编号。"
    user_prompt = f"问题：{query}\n\n相关论文的关键信息：\n{context}\n请回答问题，并在可能时提及来源论文编号。"

    gen_cfg = cfg.get("generation", {})
    client = LLMClient(gen_cfg.get("provider", "mock"), gen_cfg.get("model", "gpt-4.1-mini"), max_tokens=int(gen_cfg.get("max_tokens", 512)))
    ans = client.generate(system_prompt, user_prompt)
    citations = [{"id": d.get("id"), "title": d.get("title")} for d in docs]
    return {"answer": ans, "citations": citations, "evidence": evidences}


def main():
    """命令行：
    python src/rag.py "What is contrastive learning?" --mode hybrid --topk 5 --enhanced
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--enhanced", action="store_true")
    args = parser.parse_args()

    if args.enhanced:
        res = enhanced_answer(args.query, args.mode, args.topk)
    else:
        res = answer(args.query, args.mode, args.topk)
    print(res)


if __name__ == "__main__":
    main()

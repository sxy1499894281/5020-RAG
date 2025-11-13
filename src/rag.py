#!/usr/bin/env python3
import argparse
import os
from typing import List, Dict

import yaml
from openai import OpenAI

try:
    from .retriever import retrieve as _retrieve
except ImportError:  # allow running as a script
    from retriever import retrieve as _retrieve


def _load_config(path: str) -> Dict:
    if os.path.exists(path):
        import yaml as _y
        with open(path, "r", encoding="utf-8") as f:
            return _y.safe_load(f) or {}
    return {}


def build_context(docs: List[Dict], max_chars: int = 6000, sep: str = "\n\n---\n\n") -> str:
    parts = []
    for d in docs:
        title = d.get("title") or ""
        text = d.get("abstract") or d.get("text") or ""
        parts.append(f"Title: {title}\n{text}")
    ctx = sep.join(parts)
    if len(ctx) > max_chars:
        ctx = ctx[:max_chars]
    return ctx


class LLMClient:
    def __init__(self, provider: str, model: str, base_url: str = None, api_key: str = None, max_tokens: int = 512):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        if provider == "openai":
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"), base_url=base_url or os.environ.get("OPENAI_BASE_URL"))
        elif provider == "ollama":
            self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY", "ollama"), base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"))
        else:
            self.client = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider in ("openai", "ollama"):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        return (user_prompt[:2000] + "\n\n[Mock Answer]").strip()


def answer(query: str, mode: str = "hybrid", topk: int = 5, config_path: str = "configs/config.yaml") -> Dict:
    cfg = _load_config(config_path)
    max_chars = ((cfg.get("runtime") or {}).get("max_context_chars")) or 6000
    gen_cfg = (cfg.get("generation") or {})
    provider = gen_cfg.get("provider") or "mock"
    model = gen_cfg.get("model") or "gpt-4o-mini"
    max_tokens = int(gen_cfg.get("max_tokens") or 512)

    docs = _retrieve(query, topk, mode, float(((cfg.get("retrieval") or {}).get("alpha")) or 0.5), config_path)
    ctx = build_context(docs, max_chars=max_chars)

    system_prompt = "You are a helpful academic assistant. Use the provided context to answer the question concisely and cite papers. If unsure, say you don't know."
    user_prompt = f"Question: {query}\n\nContext:\n{ctx}\n\nInstructions: Provide a short answer, then list 2-3 citations by id and title."

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    client = LLMClient(provider=provider, model=model, base_url=base_url, api_key=api_key, max_tokens=max_tokens)
    ans = client.generate(system_prompt, user_prompt)

    cits = []
    for d in docs:
        cits.append({"id": d.get("id"), "title": d.get("title")})
    return {"answer": ans, "citations": cits}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query", type=str)
    p.add_argument("--mode", default="hybrid")
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()
    print(answer(args.query, args.mode, args.topk))

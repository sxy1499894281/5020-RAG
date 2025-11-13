#!/usr/bin/env python3
import json
import re
import argparse
import os
from typing import Optional

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def stream_clean_arxiv(in_path: str, out_path: str, max_rows: Optional[int] = None) -> int:
    count = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for _i, line in enumerate(fin):
            if max_rows is not None and count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            id_ = rec.get("id") or ""
            title = _clean_text(rec.get("title") or "")
            abstract = _clean_text(rec.get("abstract") or "")
            categories = (rec.get("categories") or "").strip()
            versions = rec.get("versions") or []
            created = versions[0].get("created") if versions and isinstance(versions[0], dict) else None
            out = {
                "id": id_,
                "title": title,
                "abstract": abstract,
                "categories": categories,
                "created": created,
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    args = parser.parse_args()
    n = stream_clean_arxiv(args.in_path, args.out_path, args.max_rows)
    print(f"Wrote {n} cleaned rows to {args.out_path}")

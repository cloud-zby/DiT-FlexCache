#!/usr/bin/env python3
from __future__ import annotations

"""调试小工具。

用途：
- 查看集合中已写入的点（id、saved_steps、num_frames、prompt）
- 按 prompt 关键字过滤
- 按 original_prompt 分组，检查是否存在重复点（同一 prompt 多个 id）

环境变量：
- QDRANT_URL（必填）
- QDRANT_API_KEY（可选）

示例：
export QDRANT_URL=http://127.0.0.1:6333
python scripts/qdrant_inspect.py --base wan_video_cache --limit 200 --group
"""

import os
import sys
import argparse
import json
from typing import Dict, List


def _connect():
    url = os.getenv("QDRANT_URL")
    if not url:
        print("ERROR: QDRANT_URL is not set", file=sys.stderr)
        sys.exit(2)
    api_key = os.getenv("QDRANT_API_KEY") or None
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except Exception:
        print("ERROR: qdrant_client is not installed. pip install qdrant-client", file=sys.stderr)
        sys.exit(2)
    return QdrantClient(url=url, api_key=api_key)


def _scroll_all(client, collection: str, limit: int, contains: str | None):
    """简化版分页滚动，最多读取 limit 条。"""
    out = []
    next_o = None
    read = 0
    while True:
        kwargs = dict(collection_name=collection, limit=min(256, max(1, limit - read)), with_payload=True, with_vectors=False)
        if next_o is not None:
            kwargs["offset"] = next_o
        page, next_o = client.scroll(**kwargs)  # type: ignore
        if not page:
            break
        for p in page:
            if contains:
                payload = getattr(p, "payload", {}) or {}
                text = payload.get("original_prompt") or payload.get("prompt") or ""
                if contains.lower() not in str(text).lower():
                    continue
            out.append(p)
            read += 1
            if read >= limit:
                break
        if read >= limit or next_o is None:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="wan_video_cache", help="集合前缀：<base>_whole/background/object")
    ap.add_argument("--limit", type=int, default=200, help="每个集合最多读取条数")
    ap.add_argument("--contains", type=str, default=None, help="按 prompt 关键字过滤")
    ap.add_argument("--group", action="store_true", help="按 original_prompt 分组输出（仅 whole 集合）")
    args = ap.parse_args()

    client = _connect()
    colls = [f"{args.base}_whole", f"{args.base}_background", f"{args.base}_object"]

    for name in colls:
        pts = _scroll_all(client, name, args.limit, args.contains)
        print(f"\nCollection: {name} (showing {len(pts)} points)")
        for p in pts:
            payload = getattr(p, "payload", {}) or {}
            data = {
                "id": getattr(p, "id", None),
                "saved_steps": payload.get("saved_steps"),
                "num_frames": payload.get("num_frames"),
                "prompt": payload.get("original_prompt") or payload.get("prompt"),
            }
            print(json.dumps(data, ensure_ascii=False))

    if args.group:
        name = f"{args.base}_whole"
        pts = _scroll_all(client, name, args.limit, args.contains)
        groups: Dict[str, List] = {}
        for p in pts:
            payload = getattr(p, "payload", {}) or {}
            prompt = payload.get("original_prompt") or payload.get("prompt") or ""
            groups.setdefault(prompt, []).append(p)
        dup = {k: v for k, v in groups.items() if len(v) > 1}
        if dup:
            print(f"\n[Group by prompt] Duplicates in {name} ({len(dup)} groups):")
            for prompt, items in dup.items():
                print(f"- prompt={prompt!r} count={len(items)}")
                for p in items:
                    payload = getattr(p, "payload", {}) or {}
                    print(f"  id={getattr(p,'id',None)} steps={payload.get('saved_steps')} frames={payload.get('num_frames')}")
        else:
            print(f"\n[Group by prompt] No duplicates found in {name}")


if __name__ == "__main__":
    main()


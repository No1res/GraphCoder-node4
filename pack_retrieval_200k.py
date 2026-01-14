#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from utils.utils import CodexTokenizer

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def tok_len(tok: CodexTokenizer, s: str) -> int:
    return len(tok.tokenize(s))

def truncate_by_tokens(tok: CodexTokenizer, s: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    ids = tok.tokenize(s)
    if len(ids) <= max_tokens:
        return s
    return tok.decode(ids[:max_tokens])

def item_to_path_and_val(item: Any) -> Tuple[str, str, float]:
    """
    item is expected: [val, statement, key_forward_context, fpath_tuple, sim]
    """
    if not isinstance(item, (list, tuple)) or len(item) < 5:
        return "", "", 0.0
    val = item[0] if isinstance(item[0], str) else ""
    fpath_tuple = item[3]
    sim = item[4]
    if isinstance(fpath_tuple, (list, tuple)):
        path = "/".join(map(str, fpath_tuple))
    else:
        path = str(fpath_tuple) if fpath_tuple is not None else ""
    try:
        sim = float(sim)
    except Exception:
        sim = 0.0
    return path, val, sim

def pack_one_query(
    tok: CodexTokenizer,
    top_k_context: List[Any],
    budget_tokens: int,
    separator: str,
    include_sim_line: bool,
) -> Tuple[str, int, int]:
    """
    返回：
      packed_text, packed_tokens, used_items
    严格按论文检索输出：path + snippet（按现有顺序；你已确认是升序）
    """
    pieces: List[str] = []
    used_tokens = 0
    used_items = 0

    sep_tokens = tok_len(tok, separator) if separator else 0

    for item in top_k_context:
        path, val, sim = item_to_path_and_val(item)
        if not path and not val:
            continue

        # 每个 snippet 的“输出单元”：path + val（可选加 sim 行，但默认不加）
        if include_sim_line:
            block = f"{path}\nSIM={sim}\n{val}"
        else:
            block = f"{path}\n{val}"

        # 防止粘连：块之间加 separator（这是纯粹的拼接符，不是提示词模板）
        add_sep = (len(pieces) > 0 and separator)
        extra = sep_tokens if add_sep else 0
        bt = tok_len(tok, block)

        if used_tokens + extra + bt <= budget_tokens:
            if add_sep:
                pieces.append(separator)
                used_tokens += sep_tokens
            pieces.append(block)
            used_tokens += bt
            used_items += 1
            continue

        # 放不下完整 block：截断最后一个 block 塞满剩余预算
        remaining = budget_tokens - used_tokens - (sep_tokens if add_sep else 0)
        if remaining > 0:
            if add_sep:
                pieces.append(separator)
                used_tokens += sep_tokens
            truncated = truncate_by_tokens(tok, block, remaining)
            if truncated:
                pieces.append(truncated)
                used_tokens += tok_len(tok, truncated)
                used_items += 1
        break

    return "".join(pieces), used_tokens, used_items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="search_results/*.search_res.jsonl")
    ap.add_argument("--output", required=True, help="output jsonl")
    ap.add_argument("--budget_tokens", type=int, default=200000)
    ap.add_argument("--separator", type=str, default="\n\n")  # 仅做分隔符
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include_sim_line", action="store_true",
                    help="如果你希望把相似度也当作检索输出的一部分，可打开；默认关闭（更贴近论文 snippet+path）")
    args = ap.parse_args()

    tok = CodexTokenizer()

    def gen():
        for i, row in enumerate(iter_jsonl(args.input)):
            if args.limit is not None and i >= args.limit:
                break
            topk = row.get("top_k_context", [])
            packed, packed_tokens, used_items = pack_one_query(
                tok=tok,
                top_k_context=topk,  # 已确认是 sim 升序
                budget_tokens=args.budget_tokens,
                separator=args.separator,
                include_sim_line=args.include_sim_line,
            )
            yield {
                "metadata": row.get("metadata", {}),
                "packed_retrieval_output": packed,
                "packed_tokens": packed_tokens,
                "used_items": used_items,
            }

    write_jsonl(args.output, gen())
    print(f"[OK] wrote: {args.output}")

if __name__ == "__main__":
    main()

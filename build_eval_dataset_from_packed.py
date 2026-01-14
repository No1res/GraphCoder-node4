#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 识别每个 snippet block 的“路径行”
# packed 的每个 block 形如：
#   repo/.../file.py
#   <code...>
PATH_LINE_RE = re.compile(r"^[^\s].*/.*\.(py|java)$")

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)

def split_packed_to_blocks(packed_text: str) -> List[str]:
    """
    将 packed_retrieval_output 拆成 context blocks：
    每个 block 从一个“路径行”开始，直到下一个路径行。
    这样不会被代码里的空行干扰，并且与论文的 snippets+path 对齐。
    """
    if not packed_text:
        return []
    lines = packed_text.splitlines()
    blocks: List[str] = []
    cur: List[str] = []

    def flush():
        nonlocal cur
        if cur:
            while cur and cur[-1].strip() == "":
                cur.pop()
            if cur:
                blocks.append("\n".join(cur))
        cur = []

    for ln in lines:
        if PATH_LINE_RE.match(ln):
            flush()
            cur.append(ln)
        else:
            if not cur:
                # packed 理论上第一行就是 path；如果不是，忽略前导噪声
                continue
            cur.append(ln)

    flush()
    return blocks

def build_packed_map(packed_path: Path) -> Dict[str, str]:
    """
    short_id -> packed_retrieval_output
    short_id = metadata.task_id.split('/')[-1]
    """
    mp: Dict[str, str] = {}
    dup = 0
    for row in iter_jsonl(packed_path):
        md = row.get("metadata", {}) or {}
        task_id = md.get("task_id", "")
        if not isinstance(task_id, str) or "/" not in task_id:
            continue
        short_id = task_id.split("/")[-1]
        packed = row.get("packed_retrieval_output") or ""
        if short_id in mp:
            dup += 1
            continue
        mp[short_id] = packed
    if dup:
        print(f"[WARN] duplicate short_id in packed: {dup} (kept first)")
    print(f"[INFO] packed_map size: {len(mp)}")
    return mp

def build_input_map(ce_input_path: Path) -> Dict[str, str]:
    """
    question_id -> input
    文件：/workspace/Projects/CoderEval/CoderEval-Input4Models/CEPythonHumanLabel.jsonl
    """
    mp: Dict[str, str] = {}
    dup = 0
    for row in iter_jsonl(ce_input_path):
        qid = row.get("question_id")
        inp = row.get("input")
        if qid is None:
            continue
        qid = str(qid)
        inp = "" if inp is None else str(inp)
        if qid in mp:
            dup += 1
            continue
        mp[qid] = inp
    if dup:
        print(f"[WARN] duplicate question_id in CE input file: {dup} (kept first)")
    print(f"[INFO] input_map size: {len(mp)}")
    return mp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True, help="packed_context jsonl (contains metadata.task_id and packed_retrieval_output)")
    ap.add_argument("--ce-input", required=True, help="CEPythonHumanLabel.jsonl (contains question_id and input)")
    ap.add_argument("--out", required=True, help="output dataset jsonl: each line has _id, model_input, context(list)")
    ap.add_argument("--context-mode", choices=["blocks", "single"], default="blocks",
                    help="blocks: split packed into snippet blocks (recommended for budgets); single: context=[packed_text]")
    ap.add_argument("--limit", type=int, default=None, help="only process first N packed rows")
    args = ap.parse_args()

    packed_path = Path(args.packed).expanduser().resolve()
    ce_input_path = Path(args.ce_input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    packed_map = build_packed_map(packed_path)
    input_map = build_input_map(ce_input_path)

    n = 0
    missing_input = 0
    missing_packed = 0

    # 以 packed 为主表：只为 packed 里出现的样本生成 dataset 行
    with out_path.open("w", encoding="utf-8") as fw:
        for i, (short_id, packed_text) in enumerate(packed_map.items()):
            if args.limit is not None and i >= args.limit:
                break

            model_input = input_map.get(short_id, "")
            if not model_input:
                missing_input += 1

            if args.context_mode == "single":
                ctx_list = [packed_text] if packed_text else []
            else:
                ctx_list = split_packed_to_blocks(packed_text)

            if not packed_text:
                missing_packed += 1

            out = {
                "_id": short_id,
                "model_input": model_input,
                "context": ctx_list,
            }
            fw.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] wrote: {out_path}")
    print(f"[INFO] rows: {n}")
    print(f"[INFO] missing model_input (question_id not found or empty): {missing_input}")
    print(f"[INFO] missing packed_retrieval_output: {missing_packed}")
    if missing_input > 0:
        print("[HINT] 如果 missing_input 很多，说明 packed 的 short_id 与 CEPythonHumanLabel 的 question_id 不一致或文件版本不匹配。")

if __name__ == "__main__":
    main()

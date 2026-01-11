import os
import json
from tqdm import tqdm

import networkx as nx

from utils.utils import CONSTANTS, CodexTokenizer, make_needed_dir, dump_jsonl, graph_to_json
from utils.ccg import create_graph
from utils.slicing import Slicing


CE_QUERY_JSONL = "/workspace/Projects/CoderEval/CoderEval-Input4Models/CEPythonHumanLabel.jsonl"
RAW_DATASET_JSON = "/workspace/Projects/CoderEval/CoderEval4Python.json"
OUTPUT_NAME = "CErepos_graphcoder_query.jsonl"


def project_to_repo_id(project: str) -> str:
    # 你的本地 repo 目录是 owner---repo
    return (project or "").strip().replace("/", "---")


def build_stub(signature: str, human_label: str) -> str:
    sig = (signature or "").strip()
    if not sig.startswith("def "):
        sig = "def " + sig
    if not sig.endswith(":"):
        sig += ":"
    doc = (human_label or "").replace('"""', "'''").strip()
    return f"{sig}\n" f'    """{doc}"""\n' f"    pass\n"


def pick_anchor_node_id(ccg: nx.MultiDiGraph, signature: str):
    """
    Prefer the node whose sourceLines contains 'def <name>'.
    Fallback: node with max startRow.
    """
    if ccg is None or ccg.number_of_nodes() == 0:
        return None

    sig = (signature or "").strip()
    head = None
    if sig.startswith("def "):
        head = sig.split("(")[0].strip()  # e.g. "def hydrate_time"

    if head:
        for v in ccg.nodes:
            src = "".join(ccg.nodes[v].get("sourceLines", []))
            if head in src:
                return v

    # fallback: max startRow
    last_node_id = None
    max_line = -1
    for v in ccg.nodes:
        sr = ccg.nodes[v].get("startRow", -1)
        if sr is not None and sr > max_line:
            max_line = sr
            last_node_id = v
    return last_node_id


def load_raw_records(raw_json_path: str):
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("RECORDS", [])
    id2rec = {}
    for r in records:
        rid = r.get("_id", None)
        if rid is None:
            continue
        id2rec[str(rid)] = r
    return id2rec


def _should_use_sliced_context(stub: str, human_label: str, qctx: str) -> bool:
    """
    Decide whether to replace stub with sliced context.
    We must ensure query_forward_context remains derived from signature + human_label,
    so only use qctx if it still contains the human_label (or at least most of it).
    """
    if qctx is None:
        return False
    qctx_str = qctx.strip()
    if qctx_str == "":
        return False

    # if human_label exists, qctx should contain it; otherwise qctx might drop it and become too weak
    hl = (human_label or "").strip()
    if hl:
        # case-insensitive containment check
        if hl.lower() not in qctx_str.lower():
            return False

    # additionally: avoid replacing with extremely short contexts
    # (e.g., only "def ...:\n")
    if len(qctx_str) < min(40, len(stub.strip())):
        return False

    return True


def main(
    ce_query_jsonl: str = CE_QUERY_JSONL,
    raw_dataset_json: str = RAW_DATASET_JSON,
    output_name: str = OUTPUT_NAME,
    include_query_graph: bool = True,
):
    id2rec = load_raw_records(raw_dataset_json)

    tokenizer = CodexTokenizer()
    slicer = Slicing()

    out = []

    # stats
    n_total = 0
    n_id_miss = 0
    n_repo_db_miss = 0
    n_graph_none = 0
    n_graph_empty = 0
    n_anchor_none = 0
    n_slice_fail = 0
    n_ctx_fallback = 0
    n_ok = 0

    with open(ce_query_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, total=len(lines), desc="Build query-cases"):
        line = line.strip()
        if not line:
            continue
        n_total += 1
        q = json.loads(line)

        qid = str(q.get("question_id", "")).strip()
        if not qid:
            n_id_miss += 1
            continue

        raw_rec = id2rec.get(qid, None)
        if raw_rec is None:
            n_id_miss += 1
            continue

        project = raw_rec.get("project", "")
        repo_id = project_to_repo_id(project)

        # 只对本地存在的图数据库 repo 生成 query-case
        repo_db_path = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_id}.jsonl")
        if not os.path.exists(repo_db_path):
            n_repo_db_miss += 1
            continue

        signature = (q.get("signature", "") or "").strip()
        human_label = (q.get("docstring", "") or "").strip()

        # query_forward_context：只由 signature + human_label 构造（作为保底）
        stub = build_stub(signature, human_label)
        stub_lines = stub.splitlines(keepends=True)

        query_forward_context = stub  # ✅ 默认一定含 human_label
        query_forward_graph = None
        forward_context_line_list = []

        if include_query_graph:
            try:
                ccg = create_graph(stub_lines, repo_id)
            except Exception:
                ccg = None

            if ccg is None:
                n_graph_none += 1
            elif ccg.number_of_nodes() == 0:
                n_graph_empty += 1
            else:
                anchor = pick_anchor_node_id(ccg, signature)
                if anchor is None:
                    n_anchor_none += 1
                else:
                    try:
                        qctx, qlines, qgraph = slicer.forward_dependency_slicing(
                            anchor, ccg, contain_node=True
                        )

                        # ✅ 无论如何，只要图有效就保存图，用于 coarse2fine
                        if qgraph is not None and qgraph.number_of_nodes() > 0:
                            query_forward_graph = graph_to_json(qgraph)
                            forward_context_line_list = qlines if qlines is not None else []

                        # ✅ 只有当 sliced context 仍然包含 human_label 且不太短时，才用 qctx 覆盖 stub
                        if qctx and _should_use_sliced_context(stub, human_label, qctx):
                            query_forward_context = qctx
                        else:
                            # fallback to stub to ensure human_label is present
                            n_ctx_fallback += 1
                            query_forward_context = stub

                    except Exception:
                        n_slice_fail += 1
                        # slicing失败也不影响：query_forward_context 仍是 stub（含 human_label）
                        query_forward_context = stub

        # 必须字段：query_forward_context / query_forward_encoding
        qcase = {
            "query_forward_context": query_forward_context,
            "query_forward_encoding": tokenizer.tokenize(query_forward_context),
            # 可选字段：query_forward_graph（没有就留 None）
            "query_forward_graph": query_forward_graph,
            "metadata": {
                # 用于 search_code.py 根据 repo 选择库：task_id.split('/')[0]
                "task_id": f"{repo_id}/{qid}",
                "repo_id": repo_id,
                "project": project,

                # 只用于过滤/评测（不进入模型输入）
                "file_path": raw_rec.get("file_path", ""),
                "lineno": raw_rec.get("lineno", ""),
                "end_lineno": raw_rec.get("end_lineno", ""),
                "code": raw_rec.get("code", ""),

                # debug
                "question_id": qid,
                "signature": signature,
                "human_label": human_label,
                "forward_context_line_list": forward_context_line_list,
            },
        }

        out.append(qcase)
        n_ok += 1

    save_path = os.path.join(CONSTANTS.query_graph_save_dir, output_name)
    make_needed_dir(save_path)
    dump_jsonl(out, save_path)

    print("\n[SUMMARY]")
    print(f"input_lines={len(lines)} parsed_total={n_total}")
    print(f"raw_id_miss={n_id_miss} repo_db_miss={n_repo_db_miss}")
    print(
        f"graph_none={n_graph_none} graph_empty={n_graph_empty} "
        f"anchor_none={n_anchor_none} slice_fail={n_slice_fail} ctx_fallback={n_ctx_fallback}"
    )
    print(f"saved={n_ok} -> {save_path}")


if __name__ == "__main__":
    # include_query_graph=True：会产出 query_forward_graph，后面可用于 coarse2fine
    # 如果你想先跑最稳的 coarse 检索，把 include_query_graph 改成 False
    main(include_query_graph=True)

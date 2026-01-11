import os
import json
import argparse
from tqdm import tqdm
import networkx as nx

from utils.utils import CONSTANTS, CodexTokenizer, make_needed_dir, dump_jsonl, graph_to_json
from utils.ccg import create_graph
from utils.slicing import Slicing


DEFAULT_CE_QUERY_JSONL = "/workspace/Projects/CoderEval/CoderEval-Input4Models/CEPythonHumanLabel.jsonl"
DEFAULT_RAW_DATASET_JSON = "/workspace/Projects/CoderEval/CoderEval4Python.json"
DEFAULT_OUTPUT_NAME = "CErepos_graphcoder_query.jsonl"


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

    # 保证 query 文本只来自 signature + human_label
    return (
        f"{sig}\n"
        f'    """{doc}"""\n'
        f"    pass\n"
    )


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
    MUST ensure query_forward_context remains derived from signature + human_label,
    so only use qctx if it still contains the human_label and isn't too short.
    """
    if qctx is None:
        return False
    qctx_str = qctx.strip()
    if qctx_str == "":
        return False

    hl = (human_label or "").strip()
    if hl and (hl.lower() not in qctx_str.lower()):
        return False

    # avoid replacing with extremely short contexts (e.g., only def line)
    if len(qctx_str) < min(40, len(stub.strip())):
        return False

    return True


def iter_jsonl(path: str):
    """Yield parsed JSON objects from a jsonl file; skip empty/bad lines."""
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), None
            except Exception as e:
                yield None, e


def main(
    ce_query_jsonl: str,
    raw_dataset_json: str,
    output_name: str,
    include_query_graph: bool = True,
):
    # sanity checks
    if not os.path.exists(ce_query_jsonl):
        raise FileNotFoundError(f"CE query jsonl not found: {ce_query_jsonl}")
    if not os.path.exists(raw_dataset_json):
        raise FileNotFoundError(f"Raw dataset json not found: {raw_dataset_json}")

    id2rec = load_raw_records(raw_dataset_json)

    tokenizer = CodexTokenizer()
    slicer = Slicing()
    out = []

    # stats
    n_total = 0
    n_bad_json = 0
    n_id_miss = 0
    n_repo_db_miss = 0
    n_graph_none = 0
    n_graph_empty = 0
    n_anchor_none = 0
    n_slice_fail = 0
    n_ctx_fallback = 0
    n_ok = 0

    # We don't know total lines easily without a pass; tqdm without total is OK
    pbar = tqdm(desc="Build query-cases")

    for obj, err in iter_jsonl(ce_query_jsonl):
        pbar.update(1)

        if err is not None:
            n_bad_json += 1
            continue

        n_total += 1
        q = obj

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

        # Only generate query-case if repo db exists
        repo_db_path = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_id}.jsonl")
        if not os.path.exists(repo_db_path):
            n_repo_db_miss += 1
            continue

        signature = (q.get("signature", "") or "").strip()
        human_label = (q.get("docstring", "") or "").strip()

        # base stub context (guaranteed to contain human_label)
        stub = build_stub(signature, human_label)
        stub_lines = stub.splitlines(keepends=True)

        query_forward_context = stub
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

                        # save graph if valid
                        if qgraph is not None and qgraph.number_of_nodes() > 0:
                            query_forward_graph = graph_to_json(qgraph)
                            forward_context_line_list = qlines if qlines is not None else []

                        # IMPORTANT: keep query_forward_context derived from signature+human_label
                        if qctx and _should_use_sliced_context(stub, human_label, qctx):
                            query_forward_context = qctx
                        else:
                            n_ctx_fallback += 1
                            query_forward_context = stub

                    except Exception:
                        n_slice_fail += 1
                        query_forward_context = stub

        qcase = {
            "query_forward_context": query_forward_context,
            "query_forward_encoding": tokenizer.tokenize(query_forward_context),
            "query_forward_graph": query_forward_graph,
            "metadata": {
                "task_id": f"{repo_id}/{qid}",
                "repo_id": repo_id,
                "project": project,

                # for leakage filtering / evaluation (not used as query)
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

    pbar.close()

    save_path = os.path.join(CONSTANTS.query_graph_save_dir, output_name)
    make_needed_dir(save_path)
    dump_jsonl(out, save_path)

    print("\n[SUMMARY]")
    print(f"ce_query_jsonl: {ce_query_jsonl}")
    print(f"raw_dataset_json: {raw_dataset_json}")
    print(f"output: {save_path}")
    print(f"parsed_total={n_total} bad_json_lines={n_bad_json}")
    print(f"raw_id_miss={n_id_miss} repo_db_miss={n_repo_db_miss}")
    print(
        f"graph_none={n_graph_none} graph_empty={n_graph_empty} "
        f"anchor_none={n_anchor_none} slice_fail={n_slice_fail} ctx_fallback={n_ctx_fallback}"
    )
    print(f"saved={n_ok}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ce_query_jsonl", type=str, default=DEFAULT_CE_QUERY_JSONL)
    parser.add_argument("--raw_dataset_json", type=str, default=DEFAULT_RAW_DATASET_JSON)
    parser.add_argument("--output_name", type=str, default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--include_query_graph", action="store_true", help="include query_forward_graph via create_graph+slicing")
    parser.add_argument("--no_query_graph", action="store_true", help="force disable graph building (coarse only)")

    args = parser.parse_args()

    include_graph = True
    if args.no_query_graph:
        include_graph = False
    elif args.include_query_graph:
        include_graph = True
    # default: include graph (与你之前一致)

    main(
        ce_query_jsonl=args.ce_query_jsonl,
        raw_dataset_json=args.raw_dataset_json,
        output_name=args.output_name,
        include_query_graph=include_graph,
    )
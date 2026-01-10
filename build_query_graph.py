import os
import copy
from tqdm import tqdm
import networkx as nx
from utils.ccg import create_graph
from utils.slicing import Slicing
from utils.utils import load_jsonl, make_needed_dir, graph_to_json, CONSTANTS, dump_jsonl, CodexTokenizer

import json
from utils.utils import (
    CONSTANTS,
    CodexTokenizer,
    make_needed_dir,
    dump_jsonl,
    graph_to_json,
)





def resolve_repo_file_path(repo_base_dir: str, repo_id: str, fpath_tuple):
    """
    Resolve real file path under repo_base_dir for different repo folder layouts.

    fpath_tuple can start with:
      A) [repo_id, ...]
      B) [owner, repo, ...] where f"{owner}_{repo}" == repo_id

    Actual repo root may be:
      1) repo_id/
      2) repo_id/repo_id/
      3) repo_id/<suffix>/               (suffix = part after '_' in repo_id, or repo name in owner/repo)
      4) repo_id/<suffix>/<suffix>/      (double nesting)
    """
    repo_base_dir = os.path.abspath(repo_base_dir)
    parts = list(fpath_tuple) if fpath_tuple else []
    if not parts:
        return None

    # Decide how many leading components to strip from fpath_tuple
    strip = 0
    if parts[0] == repo_id:
        strip = 1
    elif len(parts) >= 2 and f"{parts[0]}_{parts[1]}" == repo_id:
        strip = 2

    rel_parts = parts[strip:]
    rel_path = os.path.join(*rel_parts) if rel_parts else ""

    # Determine possible inner root name (suffix)
    suffix = None
    if "_" in repo_id:
        suffix = repo_id.split("_", 1)[1]
    # If owner/repo form, the repo name is parts[1]
    if len(parts) >= 2 and f"{parts[0]}_{parts[1]}" == repo_id:
        suffix = parts[1]

    # Candidate repo roots
    roots = []
    roots.append(os.path.join(repo_base_dir, repo_id))                    # ./repositories/<repo_id>/
    roots.append(os.path.join(repo_base_dir, repo_id, repo_id))           # ./repositories/<repo_id>/<repo_id>/

    if suffix:
        roots.append(os.path.join(repo_base_dir, repo_id, suffix))        # ./repositories/<repo_id>/<suffix>/
        roots.append(os.path.join(repo_base_dir, repo_id, suffix, suffix))# ./repositories/<repo_id>/<suffix>/<suffix>/

    # Try candidates
    for r in roots:
        cand = os.path.join(r, rel_path) if rel_path else r
        if os.path.isfile(cand):
            return cand

    return None



def last_n_context_lines_graph(graph: nx.MultiDiGraph):
    max_line = 0
    last_node_id = 0
    slicer = Slicing()
    for v in graph.nodes:
        if graph.nodes[v]['startRow'] > max_line:
            max_line = graph.nodes[v]['startRow']
            last_node_id = v
    return slicer.forward_dependency_slicing(last_node_id, graph, contain_node=True)


def build_query_subgraph(task_name):
    test_cases = load_jsonl(os.path.join(CONSTANTS.dataset_dir, task_name))
    graph_test_cases = []
    tokenizer = CodexTokenizer()
    with tqdm(total=len(test_cases)) as pbar:
        for case in test_cases:
            # read full query context
            fpath_tuple = case['metadata']['fpath_tuple']
            line_no = case['metadata']['line_no']
            case_id = case['metadata']['task_id'].split('/')[0]

            # if case_id not in CONSTANTS.repos:
            #     continue
            # with open(os.path.join(CONSTANTS.repo_base_dir, case_path), 'r') as f:
            #     src_lines = f.readlines()
            if case_id not in CONSTANTS.repos:
                pbar.update(1)
                continue

            real_path = resolve_repo_file_path(CONSTANTS.repo_base_dir, case_id, fpath_tuple)
            if real_path is None:
                # 不中断：直接跳过并打印，方便你之后补规则
                print(f"[FILE NOT FOUND] repo={case_id} fpath_tuple={fpath_tuple}")
                pbar.update(1)
                continue

            with open(real_path, "r", encoding="utf-8", errors="replace") as f:
                src_lines = f.readlines()



            query_context = src_lines[:line_no]
            ccg = create_graph(query_context, case_id)
            query_ctx, query_line_list, query_graph = last_n_context_lines_graph(ccg)
            graph_case = dict()
            graph_case['query_forward_graph'] = graph_to_json(query_graph)
            graph_case['query_forward_context'] = query_ctx
            graph_case['query_forward_encoding'] = tokenizer.tokenize(query_ctx)
            context_lines = case['prompt'].splitlines(keepends=True)
            graph_case['context'] = context_lines
            graph_case['metadata'] = copy.deepcopy(case['metadata'])
            graph_case['metadata']['forward_context_line_list'] = query_line_list
            graph_test_cases.append(copy.deepcopy(graph_case))
            pbar.update(1)

    save_path = os.path.join(CONSTANTS.query_graph_save_dir, task_name)
    make_needed_dir(save_path)
    dump_jsonl(graph_test_cases, save_path)
    return

def _project_to_repo_id(project: str) -> str:
    # 默认规则：owner/repo -> owner_repo
    # 如果你本地 repo 命名还替换了 '-'，在这里加：.replace('-', '_')
    # return (project or "").replace("/", "_").strip()
    return (project or "").strip().replace("/", "---")


def _make_python_query_stub(signature: str, human_label: str) -> str:
    sig = (signature or "").strip()
    if not sig.startswith("def "):
        sig = "def " + sig
    if not sig.endswith(":"):
        sig = sig + ":"
    doc = (human_label or "").replace('"""', "'''").strip()
    return f"{sig}\n" f'    """{doc}"""\n' f"    pass\n"


def _pick_anchor_node_id(ccg: nx.MultiDiGraph, signature: str):
    if ccg is None or ccg.number_of_nodes() == 0:
        return None

    sig = (signature or "").strip()
    head = None
    if sig.startswith("def "):
        # e.g. "def hydrate_time(nanoseconds, tz=None):" -> "def hydrate_time"
        head = sig.split("(")[0].strip()

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


def build_query_subgraph_from_signature_label_jsonl(
    query_jsonl_path: str="/workspace/Projects/CoderEval/CoderEval-Input4Models/CEPythonHumanLabel.jsonl",
    raw_dataset_json_path: str = "/workspace/Projects/CoderEval/CoderEval4Python.json",
    save_name: str = None,
):
    """
    Build query cases for GraphCoder search from:
      - CEPythonHumanLabel.jsonl (has question_id, signature, docstring)
      - CoderEval4Python.json (has RECORDS with _id, project, file_path, lineno, end_lineno, code)

    Query text uses ONLY: signature + docstring (human_label).
    Repo selection & leakage filtering metadata comes from raw dataset (allowed per your rule).
    """
    # 1) load raw dataset, build id -> record map
    with open(raw_dataset_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("RECORDS", [])
    id2rec = {str(r.get("_id")): r for r in records if r.get("_id") is not None}

    # 2) load query jsonl
    queries = []
    with open(query_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))

    tokenizer = CodexTokenizer()
    slicer = Slicing()
    out_cases = []

    with tqdm(total=len(queries)) as pbar:
        for q in queries:
            pbar.update(1)

            qid = str(q.get("question_id", "")).strip()
            if not qid:
                continue

            # signature & human label strictly from query jsonl
            signature = q.get("signature", "") or ""
            human_label = q.get("docstring", "") or ""

            # look up raw record to get repo/project info
            raw_rec = id2rec.get(qid, None)
            if raw_rec is None:
                # 你也可以 print 一下方便排查
                # print(f"[MISS RAW] question_id={qid}")
                continue

            project = raw_rec.get("project", "")
            repo_id = _project_to_repo_id(project)
            if not repo_id:
                continue

            # optional: if you want to ensure it's a known repo
            # if hasattr(CONSTANTS, "repos") and repo_id not in CONSTANTS.repos:
            #     continue

            # Build query stub (python)
            stub = _make_python_query_stub(signature, human_label)
            stub_lines = stub.splitlines(keepends=True)

            # create graph
            try:
                ccg = create_graph(stub_lines, repo_id)
            except Exception:
                continue
            if ccg is None or ccg.number_of_nodes() == 0:
                continue

            anchor = _pick_anchor_node_id(ccg, signature)
            if anchor is None:
                continue

            # slicing
            try:
                query_ctx, query_line_list, query_graph = slicer.forward_dependency_slicing(
                    anchor, ccg, contain_node=True
                )
            except Exception:
                continue
            if not query_ctx or query_graph is None:
                continue

            task_id = f"{repo_id}/{qid}"

            out_cases.append(
                {
                    "query_forward_graph": graph_to_json(query_graph),
                    "query_forward_context": query_ctx,
                    "query_forward_encoding": tokenizer.tokenize(query_ctx),
                    "metadata": {
                        "task_id": task_id,
                        "question_id": qid,
                        "repo_id": repo_id,
                        "project": project,
                        # leakage filtering fields (allowed)
                        "file_path": raw_rec.get("file_path", ""),
                        "lineno": raw_rec.get("lineno", ""),
                        "end_lineno": raw_rec.get("end_lineno", ""),
                        "code": raw_rec.get("code", ""),
                        # for debug
                        "signature": signature,
                        "human_label": human_label,
                        "forward_context_line_list": query_line_list,  # 先存着，后面search_code可能会用
                    },
                }
            )

    # 3) save
    if save_name is None:
        base = os.path.basename(query_jsonl_path)
        save_name = base.rsplit(".", 1)[0] + ".test.jsonl"  # 让 search_code.py 默认习惯也能吃

    save_path = os.path.join(CONSTANTS.query_graph_save_dir, save_name)
    make_needed_dir(save_path)  # 你项目里这个通常传 file path
    dump_jsonl(out_cases, save_path)

    print(f"[build_query_subgraph_from_signature_label_jsonl] saved {len(out_cases)} -> {save_path}")
    return save_path

if __name__ == "__main__":
    tasks_name = ["api_level.java.test", 
                  "line_level.java.test",
                  "api_level.python.test",
                  "line_level.python.test"]
    for task_name in tasks_name:
        build_query_subgraph(task_name)




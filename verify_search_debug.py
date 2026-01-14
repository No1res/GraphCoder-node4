
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_search_debug.py

用途：
  - 验证 query_forward_encoding / query_forward_graph 是否正常
  - 验证 context_database/<repo>.jsonl 的 key_forward_encoding / key_forward_graph 是否正常
  - 快速判断 coarse 是否有区分度、fine 是否全部变 0 的根因
  - 检查 MultiDiGraph 的 edge 迭代格式（2-tuple 还是 3-tuple）、edge key 是否为 CFG/DDG/CDG
  - 检查 root 选取：max(node_id) vs max(startRow)

运行示例：
  python3 verify_search_debug.py \
    --query_file ./graph_based_query/CErepos_graphcoder_query.jsonl \
    --db_dir ./context_database \
    --query_idx 0 \
    --repo_sample 20000 \
    --fine_topn 200 \
    --gamma 0.1
"""

import argparse
import os
import random
import statistics
from typing import Any, Dict, List, Tuple, Optional

import networkx as nx

# 依赖你项目已有的 utils
from utils.utils import load_jsonl, json_to_graph


def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def jaccard_tokens(a: List[int], b: List[int]) -> float:
    sa, sb = set(a or []), set(b or [])
    union = len(sa | sb)
    if union == 0:
        return 0.0
    return len(sa & sb) / union


def pick_root_by_max_id(G: nx.MultiDiGraph):
    if len(G.nodes) == 0:
        return None
    return max(G.nodes)


def pick_root_by_startrow(G: nx.MultiDiGraph):
    if len(G.nodes) == 0:
        return None

    def sr(n):
        v = G.nodes[n].get("startRow", None)
        return v if isinstance(v, int) else -1

    return max(G.nodes, key=sr)


def node_preview(G: nx.MultiDiGraph, n) -> str:
    if n is None or n not in G.nodes:
        return "<None>"
    start = G.nodes[n].get("startRow", None)
    end = G.nodes[n].get("endRow", None)
    src = G.nodes[n].get("sourceLines", [])
    head = "".join(src[:1]).strip().replace("\t", "    ")
    if len(head) > 120:
        head = head[:120] + "..."
    return f"id={n} startRow={start} endRow={end} head={head!r}"


def show_edge_samples(G: nx.MultiDiGraph, title: str, k: int = 5):
    print(f"\n[{title}] edges sample (keys=True, data=True) first {k}:")
    edges = list(G.edges(keys=True, data=True))
    if not edges:
        print("  (no edges)")
        return
    for e in edges[:k]:
        u, v, key, data = e
        etype = None
        if isinstance(data, dict):
            etype = data.get("type", None)
        print(f"  u={u} v={v} key={key!r} data.type={etype!r} data.keys={list(data.keys())[:6] if isinstance(data, dict) else None}")

    # 也看一下默认迭代 G.edges 时 tuple 长度（很多 bug 就在这里）
    print(f"\n[{title}] default iteration tuple example:")
    try:
        one = next(iter(G.edges))
        if isinstance(one, tuple):
            print(f"  next(iter(G.edges)) = {one}, len={len(one)}")
        else:
            print(f"  next(iter(G.edges)) = {one}")
    except StopIteration:
        print("  (no edges)")


def edge_key_is_cfg_like(G: nx.MultiDiGraph, sample_n: int = 200) -> Tuple[int, int]:
    """统计 sample_n 条边里，edge key 是否是 'CFG'/'DDG'/'CDG' 这类字符串。"""
    edges = list(G.edges(keys=True, data=True))
    if not edges:
        return (0, 0)
    sampled = edges[:sample_n]
    good = 0
    for _, _, key, data in sampled:
        if key in ("CFG", "DDG", "CDG"):
            good += 1
    return (good, len(sampled))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_file", required=True, type=str, help="your query jsonl")
    ap.add_argument("--db_dir", default="./context_database", type=str, help="context database dir")
    ap.add_argument("--query_idx", default=0, type=int, help="which query line to inspect")
    ap.add_argument("--repo_name", default=None, type=str, help="override repo name (otherwise from metadata.task_id)")
    ap.add_argument("--repo_sample", default=20000, type=int, help="how many repo_cases to sample for coarse stats")
    ap.add_argument("--fine_topn", default=200, type=int, help="how many top coarse candidates to inspect for fine viability")
    ap.add_argument("--gamma", default=0.1, type=float)
    ap.add_argument("--seed", default=1234, type=int)
    args = ap.parse_args()

    random.seed(args.seed)

    # 1) Load query
    if not os.path.exists(args.query_file):
        raise FileNotFoundError(args.query_file)

    queries = load_jsonl(args.query_file)
    if not queries:
        raise RuntimeError("query_file is empty")

    if args.query_idx < 0 or args.query_idx >= len(queries):
        raise IndexError(f"query_idx out of range: {args.query_idx}, total={len(queries)}")

    q = queries[args.query_idx]
    task_id = safe_get(q, ["metadata", "task_id"], "")
    if args.repo_name is not None:
        repo_name = args.repo_name
    else:
        repo_name = task_id.split("/")[0] if isinstance(task_id, str) and "/" in task_id else task_id

    print("=" * 80)
    print("[Query basic]")
    print(f"query_file: {args.query_file}")
    print(f"query_idx : {args.query_idx}")
    print(f"task_id   : {task_id!r}")
    print(f"repo_name : {repo_name!r}")
    print(f"db_dir    : {args.db_dir}")
    print("=" * 80)

    q_enc = q.get("query_forward_encoding", [])
    print("\n[Query encoding]")
    print(f"query_forward_encoding len = {len(q_enc)}")
    if len(q_enc) == 0:
        print("  !!! WARNING: query_forward_encoding is empty -> coarse will be all 0 if repo encodings are empty too")

    q_graph_str = q.get("query_forward_graph", "")
    print("\n[Query graph presence]")
    print(f"query_forward_graph is empty? {not bool(q_graph_str)}")
    if not q_graph_str:
        print("  !!! WARNING: query_forward_graph missing -> fine scores will be 0 by design in your wrapper")

    # 2) Parse query graph
    Gq = None
    if q_graph_str:
        try:
            Gq = json_to_graph(q_graph_str)
        except Exception as e:
            print(f"  !!! ERROR: json_to_graph(query_forward_graph) failed: {repr(e)}")
            Gq = None

    if Gq is not None:
        print("\n[Query graph stats]")
        print(f"nodes={len(Gq.nodes)} edges={len(Gq.edges)} type={type(Gq)}")
        if len(Gq.nodes) == 0:
            print("  !!! WARNING: query graph has 0 nodes -> fine will be 0")
        else:
            root_id = pick_root_by_max_id(Gq)
            root_sr = pick_root_by_startrow(Gq)
            print("\n[Query root check]")
            print(f"root_by_max_id      : {node_preview(Gq, root_id)}")
            print(f"root_by_max_startRow: {node_preview(Gq, root_sr)}")
            if root_id != root_sr:
                print("  !!! NOTE: root differs (max_id vs max_startRow). If your similarity uses max(node_id), root may be wrong.")
            show_edge_samples(Gq, "QUERY", k=5)
            good, total = edge_key_is_cfg_like(Gq)
            print(f"\n[QUERY edge key check] key in {{'CFG','DDG','CDG'}}: {good}/{total}")
            if total > 0 and good == 0:
                print("  !!! WARNING: query edge keys are not CFG/DDG/CDG -> if fine uses graph.has_edge(v,u,t) with t as key, edge_sim may be always 0")
    else:
        print("\n[Query graph stats] (no graph parsed)")

    # 3) Load repo DB
    if not repo_name:
        raise RuntimeError("repo_name is empty; check metadata.task_id in your query")

    repo_db_path = os.path.join(args.db_dir, f"{repo_name}.jsonl")
    print("\n[Repo DB]")
    print(f"repo_db_path: {repo_db_path}")
    if not os.path.exists(repo_db_path):
        print("  !!! ERROR: repo db not found. Your search result would be empty or error.")
        return

    repo_cases = load_jsonl(repo_db_path)
    print(f"repo_cases total: {len(repo_cases)}")
    if len(repo_cases) == 0:
        print("  !!! ERROR: repo db is empty.")
        return

    # 4) Sample repo cases for coarse stats
    sample_n = min(args.repo_sample, len(repo_cases))
    sampled_indices = random.sample(range(len(repo_cases)), sample_n) if sample_n < len(repo_cases) else list(range(sample_n))
    sampled = [repo_cases[i] for i in sampled_indices]

    # Basic sanity: how many repo encodings empty?
    empty_repo_enc = sum(1 for r in sampled if len(r.get("key_forward_encoding", [])) == 0)
    print("\n[Repo encoding sanity on sample]")
    print(f"sample_n={sample_n}, empty key_forward_encoding in sample: {empty_repo_enc}")
    if empty_repo_enc == sample_n and len(q_enc) == 0:
        print("  !!! CRITICAL: both query encoding and repo encodings empty -> coarse similarity will be 0 for all items")

    # Coarse similarity distribution
    coarse_scores = []
    for r in sampled:
        coarse_scores.append(jaccard_tokens(q_enc, r.get("key_forward_encoding", [])))

    nz = sum(1 for s in coarse_scores if s > 0)
    mx = max(coarse_scores) if coarse_scores else 0.0
    print("\n[Coarse score stats on sample]")
    print(f"nonzero count: {nz}/{len(coarse_scores)}")
    print(f"max score    : {mx:.6f}")
    print(f"mean         : {statistics.mean(coarse_scores):.6f}")
    print(f"median       : {statistics.median(coarse_scores):.6f}")
    # show top 5 coarse (within sample)
    top_idx = sorted(range(len(sampled)), key=lambda i: coarse_scores[i], reverse=True)[:5]
    print("\n[Top-5 coarse candidates in sample]")
    for rank, i in enumerate(top_idx, start=1):
        r = sampled[i]
        sim = coarse_scores[i]
        fpath = "/".join(r.get("fpath_tuple", []))
        stmt = (r.get("statement", "") or "").strip().replace("\t", "    ")
        if len(stmt) > 100:
            stmt = stmt[:100] + "..."
        print(f"  #{rank} sim={sim:.6f} file={fpath!r} stmt={stmt!r}")

    # 5) Fine viability checks on top coarse candidates (no full fine scoring here; we just sanity-check graphs)
    #    We only parse graphs and inspect edge key formats to see why fine could become all 0.
    topn = min(args.fine_topn, len(sampled))
    top_for_fine = sorted(range(len(sampled)), key=lambda i: coarse_scores[i], reverse=True)[:topn]

    empty_repo_graph = 0
    edgekey_mismatch = 0
    parsed_repo_graphs = 0
    repo_graph_example = None

    for i in top_for_fine:
        r = sampled[i]
        rg_str = r.get("key_forward_graph", "")
        if not rg_str:
            empty_repo_graph += 1
            continue
        try:
            Gr = json_to_graph(rg_str)
        except Exception:
            empty_repo_graph += 1
            continue
        parsed_repo_graphs += 1
        if repo_graph_example is None and len(Gr.nodes) > 0 and len(Gr.edges) > 0:
            repo_graph_example = Gr
        good, total = edge_key_is_cfg_like(Gr)
        if total > 0 and good == 0:
            edgekey_mismatch += 1

    print("\n[Fine viability on top coarse candidates]")
    print(f"checked topn={topn}")
    print(f"parsed repo graphs: {parsed_repo_graphs}/{topn}")
    print(f"missing/empty/unparseable repo graphs: {empty_repo_graph}/{topn}")
    print(f"edge-key not CFG/DDG/CDG among parsed graphs: {edgekey_mismatch}/{parsed_repo_graphs if parsed_repo_graphs else 1}")

    if repo_graph_example is not None:
        print("\n[One repo graph example stats]")
        print(f"nodes={len(repo_graph_example.nodes)} edges={len(repo_graph_example.edges)}")
        rid = pick_root_by_max_id(repo_graph_example)
        rsr = pick_root_by_startrow(repo_graph_example)
        print("[Repo root check]")
        print(f"root_by_max_id      : {node_preview(repo_graph_example, rid)}")
        print(f"root_by_max_startRow: {node_preview(repo_graph_example, rsr)}")
        show_edge_samples(repo_graph_example, "REPO_EXAMPLE", k=5)
        good, total = edge_key_is_cfg_like(repo_graph_example)
        print(f"\n[REPO edge key check] key in {{'CFG','DDG','CDG'}}: {good}/{total}")
        if total > 0 and good == 0:
            print("  !!! WARNING: repo edge keys are not CFG/DDG/CDG -> edge_sim based on graph.has_edge(v,u,t) likely always 0")
    else:
        print("\n[Repo graph example] No non-empty repo graph found in top candidates. Fine will almost surely be 0.")

    print("\nDONE.")
    print("=" * 80)
    print("Interpretation quick guide:")
    print("  - If query graph nodes=0 or missing -> fine=0 (fix query graph build)")
    print("  - If edge keys are NOT CFG/DDG/CDG (but data has type), then graph.has_edge(v,u,t) with t as key likely fails -> edge_sim=0")
    print("  - If root_by_max_id differs a lot from root_by_max_startRow, using max(node_id) may pick wrong root -> node_sim tiny -> fine ~ 0")
    print("  - If both query/repo encodings empty -> coarse=0 (fix encoding build)")
    print("=" * 80)


if __name__ == "__main__":
    main()

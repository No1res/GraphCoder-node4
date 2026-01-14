#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, random
from statistics import mean

from utils.utils import load_jsonl, json_to_graph, CONSTANTS
from search_code_CE import SimilarityScore  # 直接复用你当前 search 脚本里的相似度实现

def jaccard(a, b):
    A, B = set(a or []), set(b or [])
    u = len(A | B)
    return 0.0 if u == 0 else len(A & B) / u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query_file", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--topn", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    qs = load_jsonl(args.query_file)
    q = qs[args.idx]
    task_id = q["metadata"]["task_id"]
    repo = task_id.split("/")[0]
    db_path = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo}.jsonl")
    repo_cases = load_jsonl(db_path)

    # query graph stats
    Gq = json_to_graph(q["query_forward_graph"])
    print("==== QUERY ====")
    print("idx:", args.idx)
    print("task_id:", task_id)
    print("repo:", repo)
    print("query nodes:", len(Gq.nodes), "edges:", len(Gq.edges))
    print("query_forward_encoding len:", len(q.get("query_forward_encoding", [])))

    # coarse ranking
    qenc = q.get("query_forward_encoding", [])
    coarse = []
    for rc in repo_cases:
        sim = jaccard(qenc, rc.get("key_forward_encoding", []))
        coarse.append((sim, rc))
    coarse.sort(key=lambda x: x[0], reverse=True)
    top = coarse[:args.topn]

    # fine scoring on topn
    fine_scores = []
    for csim, rc in top:
        Gr = json_to_graph(rc["key_forward_graph"])
        fsim = SimilarityScore.subgraph_edit_similarity(Gq, Gr, gamma=args.gamma)
        fine_scores.append((fsim, csim, rc))

    fine_scores.sort(key=lambda x: x[0], reverse=True)

    nz = sum(1 for fsim,_,_ in fine_scores if fsim > 0)
    mx = fine_scores[0][0] if fine_scores else 0.0
    av = mean([x[0] for x in fine_scores]) if fine_scores else 0.0

    print("\n==== FINE ON COARSE TOPN ====")
    print("topn:", args.topn, "gamma:", args.gamma)
    print("fine>0 count:", nz, "/", len(fine_scores))
    print("fine max:", mx)
    print("fine mean:", av)

    print("\nTop-10 by FINE (fsim, csim, file, statement head):")
    for fsim, csim, rc in fine_scores[:10]:
        fpath = "/".join(rc.get("fpath_tuple", []))
        stmt = (rc.get("statement", "") or "").strip().replace("\t", "    ")
        if len(stmt) > 120:
            stmt = stmt[:120] + "..."
        print(f"  fsim={fsim:.6f}  csim={csim:.6f}  file={fpath}  stmt={stmt}")

    print("\nTop-10 by COARSE (csim, file, statement head):")
    for csim, rc in top[:10]:
        fpath = "/".join(rc.get("fpath_tuple", []))
        stmt = (rc.get("statement", "") or "").strip().replace("\t", "    ")
        if len(stmt) > 120:
            stmt = stmt[:120] + "..."
        print(f"  csim={csim:.6f}  file={fpath}  stmt={stmt}")

if __name__ == "__main__":
    main()

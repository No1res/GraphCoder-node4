from concurrent.futures import ThreadPoolExecutor
import os
import copy
import queue
import time
import argparse
import networkx as nx
import Levenshtein
from functools import partial

from utils.utils import (
    CONSTANTS, dump_jsonl, json_to_graph, CodexTokenizer,
    load_jsonl, make_needed_dir
)
from utils.metrics import hit

# 原 RepoEval builder（如果不提供 --query_file，仍然可以走原流程）
from build_query_graph import build_query_subgraph


class SimilarityScore:
    @staticmethod
    def text_edit_similarity(str1: str, str2: str):
        """Bugfix: avoid division by zero."""
        denom = max(len(str1), len(str2))
        if denom == 0:
            return 0.0
        return 1 - Levenshtein.distance(str1, str2) / denom

    @staticmethod
    def text_jaccard_similarity(list1, list2):
        """Bugfix: avoid union=0."""
        set1 = set(list1)
        set2 = set(list2)
        union = len(set1.union(set2))
        if union == 0:
            return 0.0
        intersection = len(set1.intersection(set2))
        return float(intersection) / union

    @staticmethod
    def subgraph_edit_similarity(query_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph, gamma=0.1):
        """
        Keep the original heuristic matching logic.
        Bugfixes:
          - safer removals from v_neighbors/u_neighbors (avoid KeyError)
          - tokenization handles empty
        """
        # NOTE: This assumes node ids are comparable and the "root" is max node id.
        query_root = max(query_graph.nodes) if len(query_graph.nodes) > 0 else None
        root = max(graph.nodes) if len(graph.nodes) > 0 else None
        if query_root is None or root is None:
            return 0.0

        tokenizer = CodexTokenizer()

        query_graph_node_embedding = tokenizer.tokenize("".join(query_graph.nodes[query_root].get('sourceLines', [])))
        graph_node_embedding = tokenizer.tokenize("".join(graph.nodes[root].get('sourceLines', [])))
        node_sim = SimilarityScore.text_jaccard_similarity(query_graph_node_embedding, graph_node_embedding)

        node_match = dict()
        match_queue = queue.Queue()
        match_queue.put((query_root, root, 0))
        node_match[query_root] = (root, 0)

        query_graph_visited = {query_root}
        graph_visited = {root}

        graph_nodes = set(graph.nodes)

        while not match_queue.empty():
            v, u, hop = match_queue.get()
            v_neighbors = (set(query_graph.neighbors(v)) | set(query_graph.predecessors(v))) - set(query_graph_visited)
            u_neighbors = graph_nodes - set(graph_visited)

            sim_score = []
            for vn in v_neighbors:
                for un in u_neighbors:
                    q_emb = tokenizer.tokenize("".join(query_graph.nodes[vn].get('sourceLines', [])))
                    g_emb = tokenizer.tokenize("".join(graph.nodes[un].get('sourceLines', [])))
                    sim = SimilarityScore.text_jaccard_similarity(q_emb, g_emb)
                    sim_score.append((sim, vn, un))

            sim_score.sort(key=lambda x: -x[0])

            for sim, vn, un in sim_score:
                if vn not in query_graph_visited and un not in graph_visited:
                    match_queue.put((vn, un, hop + 1))
                    node_match[vn] = (un, hop + 1)
                    query_graph_visited.add(vn)
                    graph_visited.add(un)

                    # Bugfix: safe remove (original code could error if already removed)
                    if vn in v_neighbors:
                        v_neighbors.remove(vn)
                    if un in u_neighbors:
                        u_neighbors.remove(un)

                    node_sim += (gamma ** (hop + 1)) * sim

                if len(v_neighbors) == 0 or len(u_neighbors) == 0:
                    break

            if len(v_neighbors) != 0:
                for vn in v_neighbors:
                    node_match[vn] = None
                    query_graph_visited.add(vn)

        edge_sim = 0.0
        for v in query_graph.nodes:
            if v not in node_match:
                node_match[v] = None

        # NOTE: graph.has_edge(v, u, t) assumes edge "key" equals t in MultiDiGraph.
        for v_query, u_query, t in query_graph.edges:
            if node_match[v_query] is not None and node_match[u_query] is not None:
                v, hop_v = node_match[v_query]
                u, hop_u = node_match[u_query]
                if graph.has_edge(v, u, t):
                    edge_sim += (gamma ** hop_v)

        return node_sim + edge_sim


class CodeSearchWorker:
    def __init__(
        self,
        query_cases,
        output_path,
        mode,
        gamma=None,
        max_top_k=CONSTANTS.max_search_top_k,
        remove_threshold=0.0,
        disable_hole_filter=False,  # keep your switch; default False = original behavior
    ):
        self.query_cases = query_cases
        self.output_path = output_path
        self.max_top_k = max_top_k
        self.remove_threshold = remove_threshold
        self.mode = mode
        self.gamma = gamma
        self.disable_hole_filter = disable_hole_filter

    @staticmethod
    def _safe_has_hole_fields(query_case):
        """
        Bugfix/robustness: if custom query format lacks fields, just skip hole filter.
        This does NOT change original behavior when fields exist.
        """
        try:
            md = query_case.get("metadata", {})
            return ("fpath_tuple" in md) and ("forward_context_line_list" in md)
        except Exception:
            return False

    def _is_context_after_hole(self, query_case, repo_case):
        """
        Same intention as original: avoid retrieving code after the hole in same file.
        Bugfixes: tolerate missing fields.
        """
        if self.disable_hole_filter:
            return False

        if not self._safe_has_hole_fields(query_case):
            return False

        if "fpath_tuple" not in repo_case or "max_line_no" not in repo_case:
            return False

        hole_fpath_str = "/".join(query_case['metadata']['fpath_tuple'])
        repo_fpath_str = "/".join(repo_case['fpath_tuple'])
        if hole_fpath_str != repo_fpath_str:
            return False

        try:
            query_case_line = max(query_case['metadata']['forward_context_line_list'])
        except Exception:
            return False

        repo_case_last_line = repo_case['max_line_no']
        return repo_case_last_line >= query_case_line

    # ----------------------------
    # Similarity wrappers
    # ----------------------------
    def _text_jaccard_similarity_wrapper(self, query_case, repo_case):
        if self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0
        sim = SimilarityScore.text_jaccard_similarity(
            query_case.get('query_forward_encoding', []),
            repo_case.get('key_forward_encoding', [])
        )
        return repo_case, sim

    def _graph_node_prior_similarity_wrapper(self, query_case, repo_case):
        # Bugfix: if query graph missing, return 0 rather than crash
        qjg = query_case.get('query_forward_graph', None)
        rjg = repo_case.get('key_forward_graph', None)
        if not qjg or not rjg:
            return repo_case, 0.0

        if self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0

        query_graph = json_to_graph(qjg)
        repo_graph = json_to_graph(rjg)

        if len(repo_graph.nodes) == 0:
            return repo_case, 0.0

        sim = SimilarityScore.subgraph_edit_similarity(query_graph, repo_graph, gamma=self.gamma)
        return repo_case, sim

    # ----------------------------
    # Retrieval (same flow as original)
    # ----------------------------
    def _find_top_k_context_one_phase(self, query_case):
        start_time = time.time()
        repo_name = query_case['metadata']['task_id'].split('/')[0]
        search_res = copy.deepcopy(query_case)

        repo_db_path = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl")
        # Bugfix: avoid crash if db missing
        if not os.path.exists(repo_db_path):
            search_res['top_k_context'] = []
            search_res['text_runtime'] = 0.0
            search_res['graph_runtime'] = 0.0
            search_res['error'] = f"repo_db_not_found: {repo_db_path}"
            return search_res

        repo_cases = load_jsonl(repo_db_path)

        with ThreadPoolExecutor(max_workers=32) as executor:
            if self.mode == 'coarse':
                compute_sim = partial(self._text_jaccard_similarity_wrapper, query_case)
            else:
                compute_sim = partial(self._graph_node_prior_similarity_wrapper, query_case)
            futures = executor.map(compute_sim, repo_cases)
            scored = list(futures)

        top_k_context_filtered = []
        for repo_case, sim in scored:
            if sim >= self.remove_threshold:
                top_k_context_filtered.append((
                    repo_case.get('val', ''),
                    repo_case.get('statement', ''),
                    repo_case.get('key_forward_context', ''),
                    repo_case.get('fpath_tuple', []),
                    sim
                ))

        # Original flow: ascending sort, take tail as top-k
        top_k_context_filtered = sorted(top_k_context_filtered, key=lambda x: x[-1], reverse=False)
        search_res['top_k_context'] = top_k_context_filtered[-self.max_top_k:]

        case_id = query_case['metadata']['task_id']
        print(f'case {case_id} finished')

        end_time = time.time()
        if self.mode == 'coarse':
            search_res['text_runtime'] = end_time - start_time
            search_res['graph_runtime'] = 0.0
        else:
            search_res['text_runtime'] = 0.0
            search_res['graph_runtime'] = end_time - start_time

        return search_res

    def _find_top_k_context_two_phase(self, query_case):
        repo_name = query_case['metadata']['task_id'].split('/')[0]
        repo_db_path = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl")
        if not os.path.exists(repo_db_path):
            res = copy.deepcopy(query_case)
            res['top_k_context'] = []
            res['text_runtime'] = 0.0
            res['graph_runtime'] = 0.0
            res['error'] = f"repo_db_not_found: {repo_db_path}"
            return res

        repo_cases = load_jsonl(repo_db_path)

        # Phase 1: text coarse over ALL repo_cases
        text_runtime_start = time.time()
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._text_jaccard_similarity_wrapper, query_case)
            futures = executor.map(compute_sim, repo_cases)
            phase1_scored = list(futures)

        # Original flow: take top-k tail by text similarity
        phase1_scored = sorted(phase1_scored, key=lambda x: x[1], reverse=False)[-self.max_top_k:]
        text_runtime_end = time.time()

        # Phase 2: graph fine ONLY on those top-k cases
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._graph_node_prior_similarity_wrapper, query_case)
            top_k_cases = [case for case, _ in phase1_scored]
            futures = executor.map(compute_sim, top_k_cases)
            phase2_scored = list(futures)

        top_k_context_filtered = []
        for repo_case, sim in phase2_scored:
            if sim >= self.remove_threshold:
                top_k_context_filtered.append((
                    repo_case.get('val', ''),
                    repo_case.get('statement', ''),
                    repo_case.get('key_forward_context', ''),
                    repo_case.get('fpath_tuple', []),
                    sim
                ))

        # Original flow: ascending sort, take tail
        top_k_context_filtered = sorted(top_k_context_filtered, key=lambda x: x[-1], reverse=False)
        query_case['top_k_context'] = top_k_context_filtered[-self.max_top_k:]

        graph_runtime_end = time.time()

        case_id = query_case['metadata']['task_id']
        print(f'case {case_id} finished')
        query_case['text_runtime'] = text_runtime_end - text_runtime_start
        query_case['graph_runtime'] = graph_runtime_end - text_runtime_end
        return copy.deepcopy(query_case)

    def run(self):
        out = []
        if self.mode in ('coarse', 'fine'):
            for query_case in self.query_cases:
                res = self._find_top_k_context_one_phase(query_case)
                out.append(copy.deepcopy(res))
        else:
            for query_case in self.query_cases:
                res = self._find_top_k_context_two_phase(query_case)
                out.append(copy.deepcopy(res))

        dump_jsonl(out, self.output_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()

    # 原 RepoEval 参数：保留
    args_parser.add_argument('--query_cases', default="api_level", type=str)

    # ✅ 自定义：直接指定 query jsonl 文件（优先级高于 --query_cases）
    args_parser.add_argument('--query_file', default=None, type=str)

    # ✅ 自定义：只跑前 N 条（调试用）
    args_parser.add_argument('--limit', default=None, type=int)

    args_parser.add_argument('--mode', type=str, default='coarse2fine')
    args_parser.add_argument('--gamma', default=0.1, type=float)

    # topK / threshold
    args_parser.add_argument('--max_top_k', default=CONSTANTS.max_search_top_k, type=int)
    args_parser.add_argument('--remove_threshold', default=0.0, type=float)

    # keep switch
    args_parser.add_argument('--disable_hole_filter', action='store_true')

    args = args_parser.parse_args()

    # 1) load query_cases
    if args.query_file is not None:
        query_path = args.query_file
        query_cases = load_jsonl(query_path)
        query_name = os.path.basename(query_path)
    else:
        # 原 RepoEval 流程：自动构建并读取
        build_query_subgraph(f"{args.query_cases}.test.jsonl")
        query_path = os.path.join(CONSTANTS.query_graph_save_dir, f"{args.query_cases}.test.jsonl")
        query_cases = load_jsonl(query_path)
        query_name = f"{args.query_cases}.test.jsonl"

    # 2) limit
    if args.limit is not None:
        query_cases = query_cases[:args.limit]

    # 3) output path（保留你的命名风格）
    safe_tag = query_name.replace("/", "_")
    save_path = os.path.join(f"./search_results/{safe_tag}.{args.mode}.{args.gamma*100}.search_res.jsonl")
    make_needed_dir(save_path)

    all_start_time = time.time()

    searcher = CodeSearchWorker(
        query_cases,
        save_path,
        args.mode,
        gamma=args.gamma,
        max_top_k=args.max_top_k,
        remove_threshold=args.remove_threshold,
        disable_hole_filter=args.disable_hole_filter,
    )
    searcher.run()

    all_end_time = time.time()
    running_time = all_end_time - all_start_time

    print('-' * 20 + "Parameters" + '-' * 20)
    print(f"query_source: {query_path}")
    print(f"mode: {args.mode}")
    print(f"gamma: {args.gamma}")
    print(f"limit: {args.limit}")
    print(f"max_top_k: {args.max_top_k}")
    print(f"remove_threshold: {args.remove_threshold}")
    print(f"disable_hole_filter: {args.disable_hole_filter}")
    print('-' * 20 + "Results" + '-' * 20)
    print(f"save_path: {save_path}")
    print('runtime %.4f' % running_time)
from concurrent.futures import ThreadPoolExecutor
import os
from utils.utils import CONSTANTS, dump_jsonl, json_to_graph, CodexTokenizer, load_jsonl, make_needed_dir
import copy
import networkx as nx
import queue
import Levenshtein
import argparse
import time
from utils.metrics import hit
from functools import partial

# 原 RepoEval 流程用的 builder（我们保留，但当 --query_file 指定时不会调用）
from build_query_graph import build_query_subgraph


class SimilarityScore:
    @staticmethod
    def text_edit_similarity(str1: str, str2: str):
        denom = max(len(str1), len(str2))
        if denom == 0:
            return 0.0
        return 1 - Levenshtein.distance(str1, str2) / denom

    @staticmethod
    def text_jaccard_similarity(list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        union = len(set1.union(set2))
        if union == 0:
            return 0.0
        intersection = len(set1.intersection(set2))
        return float(intersection) / union

    @staticmethod
    def subgraph_edit_similarity(query_graph: nx.MultiDiGraph, graph: nx.MultiDiGraph, gamma=0.1):
        # SED -> subgraph similarity
        query_root = max(query_graph.nodes)
        root = max(graph.nodes)
        tokenizer = CodexTokenizer()
        query_graph_node_embedding = tokenizer.tokenize("".join(query_graph.nodes[query_root]['sourceLines']))
        graph_node_embedding = tokenizer.tokenize("".join(graph.nodes[root]['sourceLines']))
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
                    query_graph_node_embedding = tokenizer.tokenize("".join(query_graph.nodes[vn]['sourceLines']))
                    graph_node_embedding = tokenizer.tokenize("".join(graph.nodes[un]['sourceLines']))
                    sim = SimilarityScore.text_jaccard_similarity(query_graph_node_embedding, graph_node_embedding)
                    sim_score.append((sim, vn, un))
            sim_score.sort(key=lambda x: -x[0])

            for sim, vn, un in sim_score:
                if vn not in query_graph_visited and un not in graph_visited:
                    match_queue.put((vn, un, hop + 1))
                    node_match[vn] = (un, hop + 1)
                    query_graph_visited.add(vn)
                    graph_visited.add(un)
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

        edge_sim = 0
        for v in query_graph.nodes:
            if v not in node_match.keys():
                node_match[v] = None

        for v_query, u_query, t in query_graph.edges:
            if node_match[v_query] is not None and node_match[u_query] is not None:
                v, hop_v = node_match[v_query]
                u, hop_u = node_match[u_query]
                if graph.has_edge(v, u, t):
                    edge_sim += (gamma ** hop_v)

        graph_sim = node_sim + edge_sim
        return graph_sim


class CodeSearchWorker:
    def __init__(
        self,
        query_cases,
        output_path,
        mode,
        gamma=None,
        max_top_k=CONSTANTS.max_search_top_k,
        remove_threshold=0,
        disable_hole_filter=False,   # CE 任务建议 True
        enable_leakage_filter=True,  # ✅ 新增：泄露过滤（默认开启）
        leakage_min_consecutive=3,   # ✅ 新增：连续行匹配阈值
        pool_multiplier=10,          # ✅ 新增：先取更大候选池再过滤，避免过滤后不够 topK
    ):
        self.query_cases = query_cases
        self.output_path = output_path
        self.max_top_k = max_top_k
        self.remove_threshold = remove_threshold
        self.mode = mode
        self.gamma = gamma
        self.disable_hole_filter = disable_hole_filter

        self.enable_leakage_filter = enable_leakage_filter
        self.leakage_min_consecutive = leakage_min_consecutive
        self.pool_multiplier = pool_multiplier

    @staticmethod
    def _safe_has_hole_fields(query_case):
        try:
            md = query_case.get("metadata", {})
            return ("fpath_tuple" in md) and ("forward_context_line_list" in md)
        except Exception:
            return False

    def _is_context_after_hole(self, query_case, repo_case):
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
    # Leakage filtering helpers
    # ----------------------------
    @staticmethod
    def _normalize_text(s: str) -> str:
        if s is None:
            return ""
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        # strip trailing spaces per line to make matching robust
        lines = [ln.rstrip() for ln in s.split("\n")]
        return "\n".join(lines).strip()

    @staticmethod
    def _has_consecutive_overlap(candidate_text: str, gold_code: str, min_consecutive: int = 3) -> bool:
        """
        Return True if candidate contains any >=min_consecutive consecutive lines from gold_code,
        or contains the whole gold_code.
        """
        cand = CodeSearchWorker._normalize_text(candidate_text)
        gold = CodeSearchWorker._normalize_text(gold_code)

        if not cand or not gold:
            return False

        # strongest: full containment
        if gold in cand:
            return True

        cand_lines = [ln for ln in cand.split("\n") if ln.strip() != ""]
        gold_lines = [ln for ln in gold.split("\n") if ln.strip() != ""]

        if len(cand_lines) < min_consecutive or len(gold_lines) < min_consecutive:
            return False

        cand_joined = "\n".join(cand_lines)
        for i in range(0, len(gold_lines) - min_consecutive + 1):
            snippet = "\n".join(gold_lines[i:i + min_consecutive])
            if snippet and snippet in cand_joined:
                return True
        return False

    def _apply_leakage_filter(self, query_case: dict, tuples_list: list) -> list:
        """
        tuples_list: list of (val, statement, key_forward_context, fpath_tuple, sim)
        """
        if not self.enable_leakage_filter:
            return tuples_list

        gold_code = query_case.get("metadata", {}).get("code", "")
        if not gold_code:
            return tuples_list

        out = []
        for (val, statement, key_ctx, fpath_tuple, sim) in tuples_list:
            cand_text = f"{val}\n{statement}\n{key_ctx}"
            if self._has_consecutive_overlap(cand_text, gold_code, min_consecutive=self.leakage_min_consecutive):
                continue
            out.append((val, statement, key_ctx, fpath_tuple, sim))
        return out

    # ----------------------------
    # Similarity wrappers
    # ----------------------------
    def _text_jaccard_similarity_wrapper(self, query_case, repo_case):
        if self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0
        sim = SimilarityScore.text_jaccard_similarity(
            query_case['query_forward_encoding'],
            repo_case['key_forward_encoding']
        )
        return repo_case, sim

    def _graph_node_prior_similarity_wrapper(self, query_case, repo_case):
        if query_case.get('query_forward_graph', None) in (None, ""):
            return repo_case, 0.0

        query_graph = json_to_graph(query_case['query_forward_graph'])
        repo_graph = json_to_graph(repo_case['key_forward_graph'])

        if len(repo_graph.nodes) == 0 or self._is_context_after_hole(query_case, repo_case):
            return repo_case, 0.0

        sim = SimilarityScore.subgraph_edit_similarity(query_graph, repo_graph, gamma=self.gamma)
        return repo_case, sim

    # ----------------------------
    # Retrieval
    # ----------------------------
    def _find_top_k_context_one_phase(self, query_case):
        start_time = time.time()
        repo_name = query_case['metadata']['task_id'].split('/')[0]
        search_res = copy.deepcopy(query_case)

        repo_db_path = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl")
        if not os.path.exists(repo_db_path):
            search_res['top_k_context'] = []
            search_res['text_runtime'] = 0
            search_res['graph_runtime'] = 0
            search_res['error'] = f"repo_db_not_found: {repo_db_path}"
            return search_res

        repo_cases = load_jsonl(repo_db_path)

        with ThreadPoolExecutor(max_workers=32) as executor:
            if self.mode == 'coarse':
                compute_sim = partial(self._text_jaccard_similarity_wrapper, query_case)
            else:
                compute_sim = partial(self._graph_node_prior_similarity_wrapper, query_case)
            futures = executor.map(compute_sim, repo_cases)
            top_k_context = list(futures)

        # collect candidates (no leakage filter yet)
        top_k_context_filtered = []
        for repo_case, sim in top_k_context:
            if sim >= self.remove_threshold:
                top_k_context_filtered.append((
                    repo_case.get('val', ''),
                    repo_case.get('statement', ''),
                    repo_case.get('key_forward_context', ''),
                    repo_case.get('fpath_tuple', []),
                    sim
                ))

        # sort ascending to take tail as high-score pool
        top_k_context_filtered = sorted(top_k_context_filtered, key=lambda x: x[-1], reverse=False)

        # take larger pool before leakage filter
        pool_k = max(self.max_top_k * self.pool_multiplier, self.max_top_k)
        candidate_pool = top_k_context_filtered[-pool_k:]

        # ✅ leakage filter
        candidate_pool = self._apply_leakage_filter(query_case, candidate_pool)

        # ✅ final: sort DESC and take topK
        candidate_pool = sorted(candidate_pool, key=lambda x: x[-1], reverse=True)
        search_res['top_k_context'] = candidate_pool[:self.max_top_k]

        case_id = query_case['metadata']['task_id']
        print(f'case {case_id} finished')

        end_time = time.time()
        if self.mode == 'coarse':
            search_res['text_runtime'] = end_time - start_time
            search_res['graph_runtime'] = 0
        else:
            search_res['text_runtime'] = 0
            search_res['graph_runtime'] = end_time - start_time
        return search_res

    def _find_top_k_context_two_phase(self, query_case):
        repo_name = query_case['metadata']['task_id'].split('/')[0]
        repo_db_path = os.path.join(CONSTANTS.graph_database_save_dir, f"{repo_name}.jsonl")
        if not os.path.exists(repo_db_path):
            res = copy.deepcopy(query_case)
            res['top_k_context'] = []
            res['text_runtime'] = 0
            res['graph_runtime'] = 0
            res['error'] = f"repo_db_not_found: {repo_db_path}"
            return res

        repo_cases = load_jsonl(repo_db_path)

        # phase 1: text coarse
        text_runtime_start = time.time()
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._text_jaccard_similarity_wrapper, query_case)
            futures = executor.map(compute_sim, repo_cases)
            top_k_context_phase1 = list(futures)

        # IMPORTANT: take larger pool for phase2 + leakage filtering
        phase1_k = max(self.max_top_k * self.pool_multiplier, self.max_top_k)
        top_k_context_phase1 = sorted(top_k_context_phase1, key=lambda x: x[1], reverse=False)[-phase1_k:]
        text_runtime_end = time.time()

        # phase 2: graph fine
        with ThreadPoolExecutor(max_workers=32) as executor:
            compute_sim = partial(self._graph_node_prior_similarity_wrapper, query_case)
            top_k_cases = [case for case, _ in top_k_context_phase1]
            futures = executor.map(compute_sim, top_k_cases)
            top_k_context_phase2 = list(futures)

        top_k_context_filtered = []
        for repo_case, sim in top_k_context_phase2:
            if sim >= self.remove_threshold:
                top_k_context_filtered.append((
                    repo_case.get('val', ''),
                    repo_case.get('statement', ''),
                    repo_case.get('key_forward_context', ''),
                    repo_case.get('fpath_tuple', []),
                    sim
                ))

        # ✅ leakage filter
        top_k_context_filtered = self._apply_leakage_filter(query_case, top_k_context_filtered)

        # ✅ final: sort DESC and take topK
        top_k_context_filtered = sorted(top_k_context_filtered, key=lambda x: x[-1], reverse=True)
        query_case['top_k_context'] = top_k_context_filtered[:self.max_top_k]

        graph_runtime_end = time.time()

        case_id = query_case['metadata']['task_id']
        print(f'case {case_id} finished')
        query_case['text_runtime'] = text_runtime_end - text_runtime_start
        query_case['graph_runtime'] = graph_runtime_end - text_runtime_end
        return copy.deepcopy(query_case)

    def run(self):
        query_lines_with_retrieved_results = []
        if self.mode == 'coarse' or self.mode == 'fine':
            for query_case in self.query_cases:
                res = self._find_top_k_context_one_phase(query_case)
                query_lines_with_retrieved_results.append(copy.deepcopy(res))
        else:
            for query_case in self.query_cases:
                res = self._find_top_k_context_two_phase(query_case)
                query_lines_with_retrieved_results.append(copy.deepcopy(res))
        dump_jsonl(query_lines_with_retrieved_results, self.output_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()

    # 原 RepoEval 参数：仍保留
    args_parser.add_argument('--query_cases', default="api_level", type=str)

    # 直接指定 query jsonl 文件（优先级高于 --query_cases）
    args_parser.add_argument('--query_file', default=None, type=str)

    # 只跑前 N 条（调试用）
    args_parser.add_argument('--limit', default=None, type=int)

    args_parser.add_argument('--mode', type=str, default='coarse2fine')
    args_parser.add_argument('--gamma', default=0.1, type=float)

    # topK / 阈值可控
    args_parser.add_argument('--max_top_k', default=CONSTANTS.max_search_top_k, type=int)
    args_parser.add_argument('--remove_threshold', default=0.0, type=float)

    # 禁用 hole-filter（CE 任务用）
    args_parser.add_argument('--disable_hole_filter', action='store_true')

    # ✅ 新增：泄露过滤相关参数
    args_parser.add_argument('--disable_leakage_filter', action='store_true')
    args_parser.add_argument('--leakage_min_consecutive', default=3, type=int)
    args_parser.add_argument('--pool_multiplier', default=10, type=int)

    args = args_parser.parse_args()

    # 1) load query_cases
    if args.query_file is not None:
        query_path = args.query_file
        query_cases = load_jsonl(query_path)
        query_name = os.path.basename(query_path)
    else:
        # RepoEval 旧流程：自动构建并读取
        build_query_subgraph(f"{args.query_cases}.test.jsonl")
        query_path = os.path.join(CONSTANTS.query_graph_save_dir, f"{args.query_cases}.test.jsonl")
        query_cases = load_jsonl(query_path)
        query_name = f"{args.query_cases}.test.jsonl"

    # 2) limit
    if args.limit is not None:
        query_cases = query_cases[:args.limit]

    # 3) output path
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
        enable_leakage_filter=(not args.disable_leakage_filter),
        leakage_min_consecutive=args.leakage_min_consecutive,
        pool_multiplier=args.pool_multiplier,
    )
    searcher.run()

    all_end_time = time.time()
    running_time = all_end_time - all_start_time

    print('-'*20 + "Parameters" + '-'*20)
    print(f"query_source: {query_path}")
    print(f"mode: {args.mode}")
    print(f"gamma: {args.gamma}")
    print(f"limit: {args.limit}")
    print(f"max_top_k: {args.max_top_k}")
    print(f"remove_threshold: {args.remove_threshold}")
    print(f"disable_hole_filter: {args.disable_hole_filter}")
    print(f"disable_leakage_filter: {args.disable_leakage_filter}")
    print(f"leakage_min_consecutive: {args.leakage_min_consecutive}")
    print(f"pool_multiplier: {args.pool_multiplier}")
    print('-' * 20 + "Results" + '-' * 20)
    print(f"save_path: {save_path}")
    print('runtime %.4f' % running_time)

    # hit() 对 CE 数据集可能不适配；能算就算，算不了就跳过
    try:
        search_cases = load_jsonl(save_path)
        hit1, hit5, hit10 = hit(search_cases, hits=[1, 5, 10])
        print('hit1 %.4f' % hit1)
        print('hit5 %.4f' % hit5)
        print('hit10 %.4f' % hit10)
    except Exception as e:
        print(f"[INFO] hit() skipped (dataset format mismatch): {repr(e)}")
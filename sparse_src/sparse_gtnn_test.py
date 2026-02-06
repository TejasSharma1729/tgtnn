#!/usr/bin/env python3
import sys
import os
import time
from argparse import ArgumentParser
import csv
from typing import Literal, List, Tuple, Dict, Optional, Any, Union
from dataclasses import dataclass, field

import math
import random

import sparse_gtnn
from sparse_gtnn import KNNIndexDataset, ThresholdIndexDataset, ThresholdIndexRandomized
from sparse_gtnn import SparseElem

# Type alias syntax for Python 3.12+ (compatible with 3.13)
type SparseVec = List[SparseElem]
type SparseMat = List[SparseVec]

CUR_PATH: str = os.path.dirname(os.path.abspath(__file__))
# Assumes sparse_src is one level deep.
REPO_ROOT_PATH: str = os.path.dirname(CUR_PATH)

DATASETS: List[str] = ["sparse1M", "sparseFull", "movielens", "kddb", "avazu", "random_test"]
ALGORITHMS: List[str] = ["knns", "threshold-nns", "threshold-nns-randomized"]

@dataclass
class Results:
    avg_time_ms: float = 0.0
    avg_verification_time_ms: float = 0.0
    avg_recall: float = 0.0
    avg_precision: float = 0.0
    num_queries: int = 0

    def update(self, time_ms: float, verification_time_ms: float, recall: float, precision: float):
        if self.num_queries == 0:
            self.avg_time_ms = time_ms
            self.avg_verification_time_ms = verification_time_ms
            self.avg_recall = recall
            self.avg_precision = precision
        else:
            # Iterative mean update
            self.avg_time_ms = (time_ms + self.avg_time_ms * self.num_queries) / (self.num_queries + 1)
            self.avg_verification_time_ms = (verification_time_ms + self.avg_verification_time_ms * self.num_queries) / (self.num_queries + 1)
            self.avg_recall = (recall + self.avg_recall * self.num_queries) / (self.num_queries + 1)
            self.avg_precision = (precision + self.avg_precision * self.num_queries) / (self.num_queries + 1)
        self.num_queries += 1


def make_random_test_dataset(
        dim: int = 1024, 
        data_size: int = 1000, 
        nnz: int = 20,
        num_queries: int = 10, 
        seed: int = 12345
) -> Tuple[SparseMat, SparseMat, int]:
    """
    Generates a random dataset and queries for testing purposes.
    Returns (features, queries, num_queries).
    """
    rng = random.Random(seed)

    def make_random_sparse(dim: int, nnz: int, rng: random.Random) -> SparseVec:
        idxs = rng.sample(range(dim), nnz)
        vals = [rng.random() for _ in range(nnz)]
        vec: SparseVec = []
        s = 0.0
        for i, v in zip(idxs, vals):
            new_sparse_elem = SparseElem()
            new_sparse_elem.index = i
            new_sparse_elem.value = v
            vec.append(new_sparse_elem)
            s += float(v) * float(v)
        if s > 0:
            norm = math.sqrt(s)
            for e in vec:
                e.value = float(e.value / norm)
        return vec

    features = []
    for _ in range(data_size):
        features.append(make_random_sparse(dim, nnz, rng))
    queries = []
    for _ in range(num_queries):
        queries.append(make_random_sparse(dim, nnz, rng))
    return features, queries, num_queries


def run_algorithm(
    algorithm: Literal["knns", "threshold-nns", "threshold-nns-randomized"],
    dataset: str,
    K_val: Optional[int] = None,
    threshold: Optional[float] = None,
    num_queries: Optional[int] = None,
    features_in: Optional[SparseMat] = None,
    queries_in: Optional[SparseMat] = None,
    random_dim: Optional[int] = None,
    random_size: Optional[int] = None,
    random_nnz: Optional[int] = None,
    random_num_queries: Optional[int] = None,
    random_seed: Optional[int] = None,
    double_group: bool = False,
) -> Results:
    # Prepare features and queries depending on dataset selection
    features: SparseMat
    queries: SparseMat
    available_q: int

    if dataset == "random_test":
        # use provided in-memory dataset when available (from CLI helper), otherwise generate from flags
        if features_in is not None and queries_in is not None:
            features = features_in
            queries = queries_in
            available_q = len(queries)
        else:
            rd = random_dim if random_dim is not None else 1024
            rs = random_size if random_size is not None else 1000
            rn = random_nnz if random_nnz is not None else 20
            rqq = random_num_queries if random_num_queries is not None else 10
            rseed = random_seed if random_seed is not None else 12345
            features, queries, available_q = make_random_test_dataset(
                dim=rd,
                data_size=rs,
                nnz=rn,
                num_queries=rqq,
                seed=rseed,
            )

        # Interpret num_queries: None -> use available_q; 0 -> build-only; <0 -> all queries; >available -> cap to available
        if num_queries is None:
            num_queries = available_q
        elif num_queries == 0:
            num_queries = 0
        elif num_queries < 0:
            num_queries = available_q
        else:
            num_queries = min(num_queries, available_q)
    else:
        dataset_path: str = os.path.join(REPO_ROOT_PATH, "data", dataset, "X.csr")
        features, _ = sparse_gtnn.read_sparse_matrix(dataset_path)
        query_path: str = os.path.join(REPO_ROOT_PATH, "data", dataset, "Q.csr")
        queries, _ = sparse_gtnn.read_sparse_matrix(query_path)
        total_q = len(queries)

        if num_queries is None:
            num_queries = total_q
        elif num_queries == 0:
            num_queries = 0
        elif num_queries < 0:
            num_queries = total_q
        else:
            num_queries = min(num_queries, total_q)

    # Instantiate the correct index based on algorithm
    index_dataset: Any # Union[KNNIndexDataset, ThresholdIndexDataset, ThresholdIndexRandomized]
    
    if algorithm == "knns":
        assert K_val is not None, "K_val must be provided for knns algorithm"
        index_dataset = KNNIndexDataset(dataset=features, k=K_val, use_threading=True)
    elif algorithm == "threshold-nns":
        assert threshold is not None, "threshold must be provided for threshold-nns algorithm"
        index_dataset = ThresholdIndexDataset(dataset=features, threshold=threshold, use_threading=True)
    elif algorithm == "threshold-nns-randomized":
        assert threshold is not None, "threshold must be provided for threshold-nns algorithm"
        # ThresholdIndexRandomized does not support threading argument in constructor as per .pyi
        index_dataset = ThresholdIndexRandomized(dataset=features, threshold=threshold)
    elif algorithm == "all":
         raise ValueError("algorithm='all' should be handled by the caller, not passed to run_algorithm")
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    results_obj = Results()

    # If num_queries == 0: build index only, do not perform searches.
    if num_queries == 0:
        print("Built index (num_queries=0); exiting without running queries.")
        return results_obj

    # Double-group mode: call the batched `search_multiple` on all queries at once.
    if double_group:
        if algorithm == "threshold-nns-randomized":
            raise ValueError("double-group mode is not supported for threshold-nns-randomized")

        time_start = time.time()
        # search_multiple expects list[list[SparseElem]]
        multi_res = index_dataset.search_multiple(queries[:num_queries])
        time_end = time.time()

        # Unpack results and optional dot-product count
        # Some implementations might return (results, dot_products), others just results?
        # Based on .pyi: tuple[list[list[int]], int] for KNNIndexDataset and ThresholdIndexDataset
        try:
            all_results_list, dot_products = multi_res
        except Exception:
            all_results_list = multi_res
            dot_products = None

        overall_time_ms = (time_end - time_start) * 1000.0
        
        # Calculate per-query time average
        q_time_ms = overall_time_ms / float(num_queries) if num_queries > 0 else 0.0

        for qidx in range(num_queries):
            query: SparseVec = queries[qidx]
            results: List[int] = all_results_list[qidx] # type: ignore
            metrics = index_dataset.verify_results(query, results)
            verification_time: float = metrics[0]
            recall: float = metrics[1]
            precision: float = metrics[2] if len(metrics) > 2 else 0.0

            results_obj.update(q_time_ms, verification_time, recall, precision)

        # Print a single final summary for double-group mode (average over queries)
        avg_time_ms = results_obj.avg_time_ms
        avg_ver_ms = results_obj.avg_verification_time_ms
        avg_recall = results_obj.avg_recall
        avg_precision = results_obj.avg_precision
        print(f"Double-group summary: queries={results_obj.num_queries} avg_time_ms={avg_time_ms:.2f} avg_ver_ms={avg_ver_ms:.2f} avg_recall={avg_recall:.4f} avg_precision={avg_precision:.4f}")

        if dot_products is not None:
            print(f"Total dot-products computed: {dot_products}")

        return results_obj

    # Default per-query search path
    for qidx in range(num_queries):
        query: SparseVec = queries[qidx]
        time_start: float = time.time()
        # search returns tuple[list[int], int]
        search_ret = index_dataset.search(query)
        # Handle tuple unpacking carefully
        if isinstance(search_ret, tuple):
            results = search_ret[0]
        else:
            results = search_ret

        time_end: float = time.time()
        metrics = index_dataset.verify_results(query, results)
        verification_time: float = metrics[0]
        recall: float = metrics[1]
        precision: float = metrics[2] if len(metrics) > 2 else 0.0

        q_time_ms = (time_end - time_start) * 1000.0

        results_obj.update(q_time_ms, verification_time, recall, precision)

        if algorithm == "knns":
            print(f"Query {qidx}: Found {len(results)} results, Time taken: {q_time_ms:.2f} ms, Verification Time: {verification_time:.2f} ms, Recall: {recall:.4f}")
        else:
            print(f"Query {qidx}: Found {len(results)} results, Time taken: {q_time_ms:.2f} ms, Verification Time: {verification_time:.2f} ms, Recall: {recall:.4f}, Precision: {precision:.4f}")

    return results_obj


def export_all_results_csv(all_results: List[Tuple[str, str, Results]], out_path: str) -> None:
    """Write all collected results to CSV at out_path."""
    ensure_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if ensure_header:
            writer.writerow(["dataset", "algorithm", "count", "avg_time_ms", "avg_verification_time_ms", "avg_recall", "avg_precision"])
        
        for dataset, algorithm, results in all_results:
            writer.writerow([
                dataset,
                algorithm,
                results.num_queries,
                f"{results.avg_time_ms:.4f}",
                f"{results.avg_verification_time_ms:.4f}",
                f"{results.avg_recall:.6f}",
                f"{results.avg_precision:.6f}",
            ])


if __name__ == "__main__":
    parser = ArgumentParser(description="Run GTNN algorithms on specified datasets.")
    parser.add_argument("--algorithm", choices=ALGORITHMS + ["all"], default="all", help="Algorithm to run: knns or threshold-nns or threshold-nns-randomized (default: all)")
    parser.add_argument("--double-group", choices=["False", "True", "Both"], default="Both", help="Use double grouping optimization (default: False)")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all", help="Dataset to use (default: all)")
    parser.add_argument("--K", type=int, default=10, help="Number of nearest neighbors for knns algorithm (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold for threshold-nns algorithm (default: 0.7)")
    parser.add_argument("--num-queries", type=int, default=-1, help="Number of queries to process (default: all)")
    # random_test specific options
    parser.add_argument("--random-dim", type=int, default=1024, help="Dimension for random_test dataset")
    parser.add_argument("--random-size", type=int, default=1000, help="Number of feature vectors for random_test")
    parser.add_argument("--random-nnz", type=int, default=20, help="Non-zeros per vector for random_test")
    parser.add_argument("--random-num-queries", type=int, default=10, help="Number of queries for random_test")
    parser.add_argument("--random-seed", type=int, default=12345, help="RNG seed for random_test")
    parser.add_argument("--out-csv", type=str, help="Path to CSV file to append summary results")

    args = parser.parse_args()

    # Determine which datasets to run
    datasets_to_run = []
    if args.dataset == "all":
        datasets_to_run = [d for d in DATASETS if d != "random_test"]
    else:
        datasets_to_run = [args.dataset]

    # Determine which algorithms to run
    algorithms_to_run = []
    if args.algorithm == "all":
        algorithms_to_run = ALGORITHMS
    else:
        algorithms_to_run = [args.algorithm]

    # Pre-generate random dataset if needed
    random_features = None
    random_queries = None
    if "random_test" in datasets_to_run:
        print("Generating random dataset...")
        random_features, random_queries, available_q = make_random_test_dataset(
            dim=args.random_dim,
            data_size=args.random_size,
            nnz=args.random_nnz,
            num_queries=args.random_num_queries,
            seed=args.random_seed,
        )

    # Collection list for batch export
    collected_results: List[Tuple[str, str, Results]] = []

    for ds in datasets_to_run:
        print(f"\n=== Running dataset: {ds} ===")
        for alg in algorithms_to_run:
            print(f"\n--- Algorithm: {alg} ---")
            
            # Prepare arguments
            # If random_test, pass the pre-generated data
            features_arg = None
            queries_arg = None
            if ds == "random_test":
                features_arg = random_features
                queries_arg = random_queries

            # Determine double_group modes to run
            dg_modes = []
            if args.double_group == "Both":
                dg_modes = [False, True]
            elif args.double_group == "True":
                dg_modes = [True]
            else:
                dg_modes = [False]

            for dg in dg_modes:
                try:
                    if alg == "threshold-nns-randomized" and dg:
                        print(f"Skipping {alg} with double-group (not supported)")
                        continue

                    res = run_algorithm(
                        algorithm=alg, # type: ignore
                        dataset=ds,
                        K_val=args.K,
                        threshold=args.threshold,
                        num_queries=args.num_queries,
                        random_dim=args.random_dim,
                        random_size=args.random_size,
                        random_nnz=args.random_nnz,
                        random_num_queries=args.random_num_queries,
                        random_seed=args.random_seed,
                        double_group=dg,
                        features_in=features_arg,
                        queries_in=queries_arg,
                    )
                    
                    dg_label = " (DG)" if dg else ""
                    print(f"Summary [{ds} - {alg}{dg_label}]: queries={res.num_queries} avg_time={res.avg_time_ms:.2f}ms avg_recall={res.avg_recall:.4f}")
                    
                    # Collect results
                    collected_results.append((ds, alg + dg_label, res))
                
                except Exception as e:
                    print(f"Error running {ds} - {alg}: {e}")
                    # We raise to stop strictly or we can continue. 
                    # Given strict instructions "fix bugs", raising ensures visibility.
                    raise e

    # Export all collected results at once
    if args.out_csv and collected_results:
        export_all_results_csv(collected_results, args.out_csv)

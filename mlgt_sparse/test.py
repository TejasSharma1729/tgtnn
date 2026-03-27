#!/usr/bin/env python3
import mlgt_sparse
from mlgt_sparse import BloomHashFunction, DenseSRPHasher, MinHasher, WeightedMinHasher, SparseSRPHasher
from mlgt_sparse import MLGTSaffronBloom, MLGTSaffronDenseSRP, MLGTSaffronMinHash, MLGTSaffronWeightedMinHash, MLGTSaffronSparseSRP

import numpy as np
from scipy.sparse import csr_matrix
import time
import os
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any


CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(CUR_DIR, "..", "data")
DATASETS: List[str] = ["sparse1M", "sparseFull", "movielens", "kddb", "avazu"]

def read_binary_csr(file_path: str) -> csr_matrix:
    """
    Reads a CSR matrix from the custom binary format used in this project.
    Confirmed format:
    - num_rows (uint64, 8 bytes)
    - num_cols (uint64, 8 bytes)
    - num_nonzero (uint64, 8 bytes)
    - indptr ((num_rows + 1) * uint64, 8 bytes each)
    - indices (num_nonzero * uint32, 4 bytes each)
    - values (num_nonzero * float32, 4 bytes each)
    """
    with open(file_path, 'rb') as f:
        num_vectors = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        dim = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        num_nonzero = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        
        indptr = np.frombuffer(f.read(8 * (num_vectors + 1)), dtype=np.uint64)
        indices = np.frombuffer(f.read(4 * num_nonzero), dtype=np.uint32)
        data = np.frombuffer(f.read(4 * num_nonzero), dtype=np.float32)
        
    return csr_matrix((data, indices, indptr), shape=(num_vectors, dim))


def get_dataset(dataset: str) -> Tuple[csr_matrix, csr_matrix]:
    """Loads X and Q for a given dataset name from the data directory."""
    path = os.path.join(DATA_DIR, dataset)
    x_path = os.path.join(path, "X.csr")
    q_path = os.path.join(path, "Q.csr")
    
    if not os.path.exists(x_path):
        # Try lowercase or other variations if needed, but here we assume exact match
        raise FileNotFoundError(f"Dataset X not found at {x_path}")
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"Dataset Q not found at {q_path}")
        
    return read_binary_csr(x_path), read_binary_csr(q_path)


def calculate_recall(found_indices, ground_truth_indices, k):
    """
    Calculates recall@K.
    Since the SAFFRON index returns a set of candidates (up to target_sparsity),
    we check how many of the true top-K ground truth items are present in the 
    returned candidate set.
    """
    if not found_indices or len(ground_truth_indices) == 0:
        return 0.0
    
    found_set = set(found_indices)
    gt_set = set(ground_truth_indices[:k])
    intersection = found_set.intersection(gt_set)
    return len(intersection) / k


def run_benchmarks(X, Q, args):
    """Runs recall and latency benchmarks for the specified MLGT variants."""
    k = args.num_neighbors
    num_f = args.num_features
    num_q = args.num_queries
    
    if num_f > 0:
        X = X[:num_f]
    N, D = X.shape
    num_q = min(num_q, Q.shape[0])
    
    print(f"Calculating ground truth for {num_q} queries...")
    gt_indices = []
    # Batch process for speed
    batch_size = 10
    for i in tqdm(range(0, num_q, batch_size), desc="GT computation"):
        end = min(i + batch_size, num_q)
        q_batch = Q[i:end]
        scores_batch = X.dot(q_batch.T).toarray()
        for j in range(scores_batch.shape[1]):
            scores = scores_batch[:, j]
            # Exact Top-K
            top_k = np.argsort(scores)[-k:][::-1]
            gt_indices.append(top_k)

    # Variant definitions
    all_variants = {
        "Bloom": (lambda: BloomHashFunction(dimension=D, num_hashes=args.num_hashes, num_bits=args.bits, threshold=args.threshold), MLGTSaffronBloom),
        "MinHash": (lambda: MinHasher(num_hashes=args.num_hashes, hashes_per_table=1, hash_range_pow=args.bits, seed=args.seed), MLGTSaffronMinHash),
        "WeightedMinHash": (lambda: WeightedMinHasher(num_hashes=args.num_hashes, hash_range_pow=args.bits, seed=args.seed), MLGTSaffronWeightedMinHash),
        "SparseSRP": (lambda: SparseSRPHasher(num_bits=args.bits, seed=args.seed, num_hashes=args.num_hashes), MLGTSaffronSparseSRP),
        "DenseSRP": (lambda: DenseSRPHasher(num_bits=args.bits, dimension=D, seed=args.seed, num_hashes=args.num_hashes, store=False), MLGTSaffronDenseSRP),
    }

    selected_names = args.variants.split(",")
    if "all" in selected_names:
        variants_to_run = all_variants
    else:
        variants_to_run = {name: all_variants[name] for name in selected_names if name in all_variants}

    if not variants_to_run:
        print(f"No valid variants selected from: {args.variants}")
        return {}

    results = {}

    for name, (hasher_gen, algo_class) in variants_to_run.items():
        print(f"\n--- Testing {name} ---")
        print(f"Configuration: nh={args.num_hashes}, bits={args.bits}, threshold={args.threshold}, target_sparsity={args.target_sparsity}, normalize={args.normalize}")
        
        hasher = hasher_gen()
        
        start_build = time.time()
        # Pass parameters to the engine
        index = algo_class(X.data, X.indices, X.indptr, D, hasher, 
                           num_neighbors=args.target_sparsity, threshold=args.threshold, 
                           debug=args.debug, normalize=args.normalize)
        build_time = time.time() - start_build
        print(f"Index build time: {build_time:.4f}s")

        latencies = []
        recalls = []
        
        for i in tqdm(range(num_q), desc=f"Querying {name}"):
            query = Q[i].toarray().flatten()
            
            start_search = time.time()
            found = index.search(query)
            end_search = time.time()
            
            latencies.append((end_search - start_search) * 1000) # ms
            recalls.append(calculate_recall(found, gt_indices[i], k))

        avg_latency = np.mean(latencies)
        avg_recall = np.mean(recalls)
        
        print(f"Avg Latency: {avg_latency:.2f} ms | Avg Recall@{k}: {avg_recall:.4f}")
        
        results[name] = {
            "build_time": build_time,
            "avg_latency": avg_latency,
            "avg_recall": avg_recall
        }

    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="Sophisticated Benchmark for MLGT Sparse variants.")
    # Dataset and size
    parser.add_argument("--dataset", "-d", type=str, default="sparse1M", choices=DATASETS, help="Dataset for testing (default: sparse1M)")
    parser.add_argument("--num-features", "-n", type=int, default=-1, help="N, the dataset size (-1 for full)")
    parser.add_argument("--num-queries", "-q", type=int, default=100, help="Number of queries to benchmark (default: 100)")
    parser.add_argument("--num-neighbors", "-k", type=int, default=10, help="K for Recall@K (default: 10)")
    
    # Hashing Parameters
    parser.add_argument("--num-hashes", "-H", type=int, default=50, help="Number of independent hashes (default: 50)")
    parser.add_argument("--bits", "-b", type=int, default=20, help="Bits per hash (or range power for MinHash) (default: 20)")
    parser.add_argument("--threshold", "-t", type=int, default=20, help="Match threshold for inverted index (default: 20)")
    
    # Saffron Parameters
    parser.add_argument("--target-sparsity", "-s", type=int, default=100, help="Target sparsity (num_neighbors) for recovery (default: 100)")
    parser.add_argument("--normalize", "-l", action="store_true", help="Enable L2 normalization of vectors")
    
    # Execution options
    parser.add_argument("--variants", "-V", type=str, default="all", help="Comma-separated variants to test (Bloom,MinHash,WeightedMinHash,SparseSRP,DenseSRP) or 'all'")
    parser.add_argument("--seed", "-S", type=int, default=42, help="Random seed for hashers (default: 42)")
    parser.add_argument("--debug", "-v", type=int, default=0, help="Debug verbosity level (default: 0)")
    
    args = parser.parse_args()

    print(f"=== MLGT Sparse Benchmark Tool ===")
    print(f"Dataset: {args.dataset}")
    
    try:
        X, Q = get_dataset(args.dataset)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    assert isinstance(X.shape, tuple) and isinstance(Q.shape, tuple)
    assert len(X.shape) == 2 and len(Q.shape) == 2 and X.shape[1] == Q.shape[1]

    print(f"Dataset X: {X.shape[0]} rows, {X.shape[1]} cols, {X.nnz} non-zeros")
    print(f"Queries Q: {Q.shape[0]} rows, {Q.shape[1]} cols, {Q.nnz} non-zeros")

    results = run_benchmarks(X, Q, args)

    if results:
        print("\n" + "="*70)
        n_display = args.num_features if args.num_features > 0 else X.shape[0]
        header = f"FINAL PERFORMANCE SUMMARY (Recall@{args.num_neighbors}, N={n_display})"
        print(f"{header:^70}")
        print("="*70)
        print(f"{'Variant':<15} | {'Build (s)':<10} | {'Latency (ms)':<15} | {'Recall':<10}")
        print("-" * 70)
        for name, res in results.items():
            print(f"{name:<15} | {res['build_time']:<10.2f} | {res['avg_latency']:<15.2f} | {res['avg_recall']:<10.4f}")
        print("="*70)

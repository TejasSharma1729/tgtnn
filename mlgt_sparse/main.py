#!/usr/bin/env python3
import csv
import os
import time
from argparse import ArgumentParser
from typing import Dict, Any

import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

from dataset_utils import read_binary_csr, get_dataset, calculate_recall, DATASETS

import mlgt_sparse
from mlgt_sparse import (
    BloomHashFunction, DenseSRPHasher, MinHasher, WeightedMinHasher, SparseSRPHasher,
    MLGTSaffronBloom, MLGTSaffronDenseSRP, MLGTSaffronMinHash, MLGTSaffronWeightedMinHash, MLGTSaffronSparseSRP,
    MLGTGlobalBloom, MLGTGlobalDenseSRP, MLGTGlobalMinHash, MLGTGlobalWeightedMinHash, MLGTGlobalSparseSRP
)

def run_benchmarks(X: csr_matrix, Q: csr_matrix, args) -> Dict[str, Any]:
    """Runs recall and latency benchmarks for the specified MLGT variants."""
    k = args.num_neighbors
    num_f = args.num_features
    num_q = args.num_queries
    
    if num_f > 0:
        X = X[:num_f]
    assert isinstance(X.shape, tuple) and len(X.shape) == 2
    assert isinstance(Q.shape, tuple) and len(Q.shape) == 2
    
    N, D = X.shape
    num_q = min(num_q, Q.shape[0])
    
    print(f"Calculating ground truth for {num_q} queries...")
    gt_indices = []
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

    def get_hasher(name, D):
        if name == "Bloom": return lambda: BloomHashFunction(dimension=D, num_hashes=args.num_hashes, num_bits=args.bits, threshold=args.threshold)
        if name == "MinHash": return lambda: MinHasher(num_hashes=args.num_hashes, hashes_per_table=1, hash_range_pow=args.bits, seed=args.seed)
        if name == "WeightedMinHash": return lambda: WeightedMinHasher(num_hashes=args.num_hashes, hash_range_pow=args.bits, seed=args.seed)
        if name == "SparseSRP": return lambda: SparseSRPHasher(num_bits=args.bits, seed=args.seed, num_hashes=args.num_hashes)
        if name == "DenseSRP": return lambda: DenseSRPHasher(num_bits=args.bits, dimension=D, seed=args.seed, num_hashes=args.num_hashes)
        raise ValueError(f"Unknown hasher {name}")

    hasher_names = ["Bloom", "MinHash", "WeightedMinHash", "SparseSRP", "DenseSRP"]
    selected_names = [h.strip() for h in args.variants.split(",")]
    if "all" in [h.lower() for h in selected_names]:
        selected_names = hasher_names

    variants_to_run = {}
    for name in selected_names:
        if name not in hasher_names:
            print(f"Warning: Skipping unknown variant '{name}'")
            continue
            
        if args.index_type in ["saffron", "both"]:
            algo = globals()[f"MLGTSaffron{name}"]
            variants_to_run[f"{name}(Saffron)"] = (get_hasher(name, D), algo)
        if args.index_type in ["global", "both"]:
            algo = globals()[f"MLGTGlobal{name}"]
            variants_to_run[f"{name}(Global)"] = (get_hasher(name, D), algo)

    if not variants_to_run:
        print(f"No valid variants selected.")
        return {}

    results = {}

    for name, (hasher_gen, algo_class) in variants_to_run.items():
        print(f"\n--- Testing {name} ---")
        print(f"Configuration: nh={args.num_hashes}, bits={args.bits}, threshold={args.threshold}, target_sparsity={args.target_sparsity}, normalize={args.normalize}")
        
        hasher = hasher_gen()
        
        start_build = time.time()
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
            "Variant": name,
            "Build_Time_s": build_time,
            "Avg_Latency_ms": avg_latency,
            "Avg_Recall": avg_recall
        }

    return results

if __name__ == "__main__":
    parser = ArgumentParser(description="Sophisticated Benchmark for MLGT Sparse variants.")
    parser.add_argument("--dataset", "-d", type=str, default="sparse1M", choices=DATASETS, help="Dataset for testing (default: sparse1M)")
    parser.add_argument("--num-features", "-n", type=int, default=-1, help="N, the dataset size (-1 for full)")
    parser.add_argument("--num-queries", "-q", type=int, default=100, help="Number of queries to benchmark (default: 100)")
    parser.add_argument("--num-neighbors", "-k", type=int, default=10, help="K for Recall@K (default: 10)")
    
    parser.add_argument("--num-hashes", "-H", type=int, default=50, help="Number of independent hashes (default: 50)")
    parser.add_argument("--bits", "-b", type=int, default=20, help="Bits per hash (or range power for MinHash) (default: 20)")
    parser.add_argument("--threshold", "-t", type=int, default=20, help="Match threshold for inverted index (default: 20)")
    
    parser.add_argument("--target-sparsity", "-s", type=int, default=100, help="Target sparsity (num_neighbors) for recovery (default: 100)")
    parser.add_argument("--normalize", "-l", action="store_true", help="Enable L2 normalization of vectors")
    
    parser.add_argument("--variants", "-V", type=str, default="all", help="Comma-separated hashers to test (Bloom,MinHash,WeightedMinHash,SparseSRP,DenseSRP) or 'all'")
    parser.add_argument("--index-type", "-I", type=str, choices=["saffron", "global", "both"], default="both", help="Which index strategy to use (default: both)")
    parser.add_argument("--output-csv", "-o", type=str, default="", help="Path to save results as CSV (optional)")
    
    parser.add_argument("--seed", "-S", type=int, default=42, help="Random seed for hashers (default: 42)")
    parser.add_argument("--debug", "-v", type=int, default=0, help="Debug verbosity level (default: 0)")
    
    args = parser.parse_args()

    print(f"=== MLGT Sparse Benchmark Tool ===")
    print(f"Dataset: {args.dataset}")
    
    try:
        X, Q = get_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    print(f"Dataset X: {X.shape[0]} rows, {X.shape[1]} cols, {X.nnz} non-zeros")
    print(f"Queries Q: {Q.shape[0]} rows, {Q.shape[1]} cols, {Q.nnz} non-zeros")

    results = run_benchmarks(X, Q, args)

    if results:
        print("\n" + "="*75)
        n_display = args.num_features if args.num_features > 0 else X.shape[0]
        header = f"FINAL PERFORMANCE SUMMARY (Recall@{args.num_neighbors}, N={n_display})"
        print(f"{header:^75}")
        print("="*75)
        print(f"{'Variant':<25} | {'Build (s)':<10} | {'Latency (ms)':<15} | {'Recall':<10}")
        print("-" * 75)
        for name, res in results.items():
            print(f"{name:<25} | {res['Build_Time_s']:<10.2f} | {res['Avg_Latency_ms']:<15.2f} | {res['Avg_Recall']:<10.4f}")
        print("="*75)

        if args.output_csv:
            fieldnames = ["Variant", "Build_Time_s", "Avg_Latency_ms", "Avg_Recall"]
            file_exists = os.path.isfile(args.output_csv)
            try:
                with open(args.output_csv, mode='a' if file_exists else 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    for row in results.values():
                        writer.writerow(row)
                print(f"Results successfully saved to {args.output_csv}")
            except Exception as e:
                print(f"Failed to write CSV: {e}")

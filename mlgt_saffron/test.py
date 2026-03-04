#!/usr/bin/env python3
import math, cmath, random, statistics
import os, sys, gc, time

import numpy as np
from numpy import array, ndarray, random as npr, linalg

from tqdm import tqdm
from typing import List, Tuple, Dict, Set, Literal, Optional, Callable, Iterable, Union, Any
from argparse import ArgumentParser

CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)
DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]

from mlgt_saffron import SaffronIndex, MLGTSaffron, BloomGroupTestingSaffron

def test_saffron(
        dataset: ndarray,
        query_set: ndarray,
        num_neighbors: int,
        algo_name: str = "mlgt",
        num_hashes: int = 50,
        hash_bits: int = 20,
        threshold: int = 20,
        verbose: int = 0,
) -> Tuple[float, float, float, float, float]:
    """
    Test the Saffron implementation on a given dataset and query set.

    Parameters:
    - dataset: ndarray of shape (num_items, num_features)
    - query_set: ndarray of shape (num_queries, num_features)
    - num_neighbors: int, number of nearest neighbors to retrieve
    - algo_name: str, "mlgt" or "bloom"
    - num_hashes: int, number of compound hash functions
    - hash_bits: int, number of bits per compound hash
    - threshold: int, number of matches required

    Returns:
    - Indexing time (for the dataset)
    - Average search time per query
    - Average naive search time per query
    - Average precision per query
    - Average recall per query
    """
    idx_start = time.time()
    if algo_name == "mlgt":
        saffron_index = MLGTSaffron(
            dataset, num_neighbors, 
            num_hashes=num_hashes, hash_bits=hash_bits, threshold=threshold,
            debug=verbose
        ) # type: ignore
    elif algo_name == "bloom":
        saffron_index = BloomGroupTestingSaffron(
            dataset, num_neighbors,
            num_hashes=num_hashes, hash_bits=hash_bits, threshold=threshold,
            debug=verbose
        ) # type: ignore
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    idx_time: float = time.time() - idx_start
    total_saffron_time: float = 0.0
    total_naive_time: float = 0.0
    total_precision: float = 0.0
    total_recall: float = 0.0
    num_queries: int = query_set.shape[0]
    
    for qidx in tqdm(range(num_queries), desc="Testing queries"):
        query: ndarray = query_set[qidx]
        
        # Saffron search
        start_time: float = time.time()
        retrieved_indices: List[int] = saffron_index.search(query) # type: ignore
        saffron_time: float = time.time() - start_time
        
        # Naive search
        start_time = time.time()
        distances: ndarray = dataset @ query
        true_indices: ndarray = np.argsort(distances)[-num_neighbors:]
        naive_time: float = time.time() - start_time
        
        # Compute precision and recall
        retrieved_set: Set[int] = set(retrieved_indices)
        true_set: Set[int] = set(true_indices)
        
        true_positives: int = len(retrieved_set.intersection(true_set))
        precision: float = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall: float = true_positives / len(true_set) if true_set else 0.0

        if (verbose > 0):
            print(f"Query {qidx + 1}/{num_queries}:")
            print(f"  Saffron retrieved indices: {retrieved_indices}")
            print(f"    Dot products: {distances[retrieved_indices].tolist()}")
            print(f"  True nearest indices: {true_indices.tolist()}")
            print(f"    Dot products: {distances[true_indices].tolist()}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"  Saffron time: {saffron_time:.6f} s, Naive time: {naive_time:.6f} s")
        
        # Aggregate results
        total_saffron_time += saffron_time
        total_naive_time += naive_time
        total_precision += precision
        total_recall += recall
    
    avg_saffron_time: float = total_saffron_time / num_queries
    avg_naive_time: float = total_naive_time / num_queries
    avg_precision: float = total_precision / num_queries
    avg_recall: float = total_recall / num_queries
    
    return idx_time, avg_saffron_time, avg_naive_time, avg_precision, avg_recall


if __name__ == "__main__":
    parser = ArgumentParser("Test Saffron implementation")
    parser.add_argument(
        "--data-path",
        "-p",
        type=str, 
        default="../data/",
        help="Path to the datasets' directory [default: ../data/]"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        nargs="+",
        default=["all"],
        help="The dataset name(s) [imagenet(default), imdb_wiki, insta_1m, mirflickr, all]"
    )
    parser.add_argument(
        "--num-features",
        "-n",
        type=int,
        default=-1,
        help="Number of dataset features to use (default: all)"
    )
    parser.add_argument(
        "--num-queries",
        "-q",
        type=int, 
        default=1000, 
        help="Number of queries to test (default: all)"
    )
    parser.add_argument(
        "--num-neighbors",
        "--k-val",
        "-k",
        type=int, 
        default=10, 
        help="Number of nearest neighbors to retrieve (default: 10)"
    )
    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        nargs="+",
        choices=["mlgt", "bloom"],
        default=["mlgt"],
        help="The algorithm(s) to test [mlgt(default), bloom]"
    )
    parser.add_argument(
        "--num-hashes",
        type=int,
        nargs="+",
        default=[100],
        help="Number of hashes for Bloom Filter (default: 100)"
    )
    parser.add_argument(
        "--hash-bits",
        type=int,
        nargs="+",
        default=[16],
        help="Bits per hash for Bloom Filter (default: 16)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        nargs="+",
        default=[10],
        help="Match threshold for Bloom Filter (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=0,
        help="Verbosity level (default: 0)"
    )
    args = parser.parse_args()

    # Determine which datasets to run
    datasets_to_run = args.dataset
    if "all" in datasets_to_run:
        datasets_to_run = DATASETS

    # Nested loops for parameters
    for dname in datasets_to_run:
        # Load data for this dataset
        data_path: str = os.path.join(args.data_path, dname)
        try:
            full_dataset: ndarray = np.load(os.path.join(data_path, "X.npy"))
            full_query_set: ndarray = np.load(os.path.join(data_path, "Q.npy"))
        except FileNotFoundError:
            print(f"\n[Error] Dataset files not found in {data_path}. Skipping.")
            continue

        if args.num_features > 0:
            full_dataset = full_dataset[:args.num_features, :]
        if args.num_queries > 0:
            full_query_set = full_query_set[:args.num_queries, :]

        for algo in args.algo:
            for nh, hb, th in zip(args.num_hashes, args.hash_bits, args.threshold):
                print(f"\n>>> Running: Dataset={dname}, Algo={algo}, k={args.num_neighbors}, hashes={nh}, bits={hb}, threshold={th}")
                
                idx_time, avg_saffron_time, avg_naive_time, avg_precision, avg_recall = test_saffron(
                    full_dataset, 
                    full_query_set, 
                    args.num_neighbors,
                    algo,
                    nh,
                    hb,
                    th,
                    args.verbose
                )

                print(f"--- Results for {algo.upper()} on {dname.upper()} ---")
                print(f"Indexing Time: {idx_time:.6f} seconds")
                print(f"Avg Saffron Search: {avg_saffron_time:.6f} seconds")
                print(f"Avg Naive Search:   {avg_naive_time:.6f} seconds")
                print(f"Avg Precision:      {avg_precision:.4f}")
                print(f"Avg Recall:         {avg_recall:.4f}")


#!/usr/bin/env python3
import os
import sys
import time

from argparse import ArgumentParser, Namespace
from typing import Literal, List, Tuple, Dict, Set, Iterable, Callable, Union, Optional, Any
from dataclasses import dataclass, field

import math
import cmath
import random

import numpy as np
from numpy import array, ndarray, linalg, random as npr, matrix, tensordot

import numba
from tqdm import tqdm, trange

# NOTE: You need to run `make` in this directory to build the mlgt_saffron module
from mlgt_saffron import MLGT

CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
REPO_DIR: str = os.path.abspath(os.path.join(CUR_DIR, ".."))
DATA_DIR: str = os.path.join(REPO_DIR, "data")

@dataclass
class Metrics:
    full_precision: float = 0.0
    full_recall: float = 0.0
    full_time: float = 0.0
    true_time: float = 0.0
    
    def avg_over(self, num_queries: int) -> None:
        self.full_precision /= num_queries
        self.full_recall /= num_queries
        self.full_time /= num_queries
        self.true_time /= num_queries

def test_dataset(
        dataset_name: str,
        num_data: int = -1,
        num_queries: int = 100,
        num_tables: int = 100,
        num_bits: int = 10,
        num_neighbors: int = 100,
        match_threshold: int = 1,
        num_pools: int = 5000,
        pools_per_point: int = 10,
        debug: int = 0
) -> Metrics:
    dataset_path: str = os.path.join(DATA_DIR, dataset_name)
    try:
        X: ndarray = np.load(os.path.join(dataset_path, "X.npy")).astype(np.float32)
        Q: ndarray = np.load(os.path.join(dataset_path, "Q.npy")).astype(np.float32)
    except FileNotFoundError:
        print(f"Dataset {dataset_name} not found in {dataset_path}")
        return Metrics()

    if num_data > 0:
        X = X[:num_data]
        
    num_features = X.shape[0]
    dimension = X.shape[1]
    
    # Calculate points_per_pool
    # tot_points * pools_per_point = num_pools * points_per_pool
    # points_per_pool = ceil(...)
    points_per_pool = math.ceil((num_features * pools_per_point) / num_pools)
    
    print(f"Configuration:")
    print(f"  Num Features: {num_features}")
    print(f"  Dimension: {dimension}")
    print(f"  Num Pools: {num_pools}")
    print(f"  Pools per Point: {pools_per_point}")
    print(f"  Points per Pool: {points_per_pool}")
    print(f"  Num Tables: {num_tables}")
    print(f"  Num Bits: {num_bits}")
    print(f"  Match Threshold: {match_threshold}")
    
    mlgt = MLGT(
        num_tables=num_tables,
        num_bits=num_bits,
        dimension=dimension,
        num_pools=num_pools,
        pools_per_point=pools_per_point,
        points_per_pool=points_per_pool,
        match_threshold=match_threshold,
        num_neighbors=num_neighbors,
        features=X, # type: ignore
        debug=debug
    )

    metrics = Metrics()
    
    # Normalize X for ground truth comparison (MLGT does it internally)
    X_norm = X.copy()
    norms = np.linalg.norm(X_norm, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    X_norm = X_norm / norms

    # Normalize Q
    Q_norm = Q.copy()
    q_norms = np.linalg.norm(Q_norm, axis=1, keepdims=True)
    q_norms[q_norms < 1e-6] = 1.0
    Q_norm = Q_norm / q_norms

    for qidx in tqdm(range(min(num_queries, Q.shape[0])), desc=f"Testing {dataset_name}"):
        query: ndarray = Q[qidx]
        query_norm: ndarray = Q_norm[qidx]
        
        start_time: float = time.time()
        neighbors: List[int] = mlgt.query(query) # type: ignore
        end_time: float = time.time()
        metrics.full_time += (end_time - start_time)

        start_time = time.time()
        # Ground truth
        matches: ndarray = X_norm @ query_norm
        true_neighbors: List[int] = np.argsort(matches)[::-1][:num_neighbors].tolist()
        end_time = time.time()
        metrics.true_time += (end_time - start_time)
        
        # Calculate precision/recall
        true_set = set(true_neighbors)
        retrieved_set = set(neighbors)
        
        if len(retrieved_set) > 0:
            intersect = len(true_set.intersection(retrieved_set))
            metrics.full_precision += intersect / len(retrieved_set)
            metrics.full_recall += intersect / len(true_set)

    metrics.avg_over(min(num_queries, Q.shape[0]))
    return metrics

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sift1M", help="Dataset name")
    parser.add_argument("--num_data", type=int, default=-1, help="Number of data points to use")
    parser.add_argument("--num_queries", type=int, default=10, help="Number of queries")
    parser.add_argument("--num_tables", type=int, default=500, help="Number of tables")
    parser.add_argument("--num_bits", type=int, default=10, help="Number of bits per hash")
    parser.add_argument("--num_neighbors", type=int, default=100, help="Number of neighbors")
    parser.add_argument("--match_threshold", type=int, default=50, help="Min table matches")
    parser.add_argument("--num_pools", type=int, default=5000, help="Number of pools")
    parser.add_argument("--pools_per_point", type=int, default=10, help="Number of pools per point")
    parser.add_argument("--debug", type=int, default=0, help="Debug level")
    
    args = parser.parse_args()
    
    metrics = test_dataset(
        dataset_name=args.dataset,
        num_data=args.num_data,
        num_queries=args.num_queries,
        num_tables=args.num_tables,
        num_bits=args.num_bits,
        num_neighbors=args.num_neighbors,
        match_threshold=args.match_threshold,
        num_pools=args.num_pools,
        pools_per_point=args.pools_per_point,
        debug=args.debug
    )
    
    print("-" * 40)
    print(f"Results for {args.dataset}:")
    print(f"Precision: {metrics.full_precision:.4f}")
    print(f"Recall:    {metrics.full_recall:.4f}")
    print(f"Query Time: {metrics.full_time:.6f} s")
    print(f"True Time:  {metrics.true_time:.6f} s")
    print(f"Speedup:    {metrics.true_time / metrics.full_time if metrics.full_time > 0 else 0:.2f}x")
    print("-" * 40)

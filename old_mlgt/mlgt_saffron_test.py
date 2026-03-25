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
from mlgt_saffron import SimHash, InvertedIndex, Saffron, MLGTSaffronIndex, MLGTsaffron

CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
REPO_DIR: str = os.path.abspath(os.path.join(CUR_DIR, ".."))
DATA_DIR: str = os.path.join(REPO_DIR, "data")

@dataclass
class Metrics:
    full_precision: float = 0.0
    full_recall: float = 0.0
    full_time: float = 0.0
    true_time: float = 0.0
    mlgt_hash_precision: float = 0.0
    mlgt_hash_recall: float = 0.0
    hash_time: float = 0.0
    hash_true_precision: float = 0.0
    hash_true_recall: float = 0.0

    def avg_over(self, num_queries: int) -> None:
        self.full_precision /= num_queries
        self.full_recall /= num_queries
        self.full_time /= num_queries
        self.true_time /= num_queries
        self.mlgt_hash_precision /= num_queries
        self.mlgt_hash_recall /= num_queries
        self.hash_time /= num_queries
        self.hash_true_precision /= num_queries
        self.hash_true_recall /= num_queries


def test_dataset(
        dataset_name: str,
        num_data: int = -1,
        num_queries: int = 100,
        num_tables: int = 500,
        num_bits: int = 10,
        num_neighbors: int = 100,
        threshold: int = 50,
        debug: int = 0
) -> Metrics:
    dataset_path: str = os.path.join(DATA_DIR, dataset_name)
    X: ndarray = np.load(os.path.join(dataset_path, "X.npy")).astype(np.float32)
    Q: ndarray = np.load(os.path.join(dataset_path, "Q.npy")).astype(np.float32)
    if num_data < 0:
        num_data = X.shape[0]
    else:
        X = X[:num_data]

    mlgt = MLGTsaffron(
        num_tables=num_tables,
        num_bits=num_bits,
        dimension=X.shape[1],
        num_neighbors=num_neighbors,
        threshold=threshold,
        features=X, # type: ignore
        debug=debug
    )

    metrics = Metrics()

    for qidx in tqdm(range(num_queries), desc=f"Testing {dataset_name}"):
        query: ndarray = Q[qidx]
        
        start_time: float = time.time()
        neighbors: List[int] = mlgt.query(query, k=num_neighbors) # type: ignore
        end_time: float = time.time()
        metrics.full_time += (end_time - start_time)

        start_time = time.time()
        hash_neighbors: List[int] = mlgt.get_top_k_hash_neighbors(query, k=num_neighbors) # type: ignore
        end_time = time.time()
        metrics.hash_time += (end_time - start_time)

        start_time = time.time()
        matches: ndarray = X @ query
        assert matches.shape == (X.shape[0],)
        true_neighbors: List[int] = np.argsort(matches)[::-1][:num_neighbors].tolist()
        end_time = time.time()
        metrics.true_time += (end_time - start_time)

        # Full MLGT SAFFRON precision/recall
        true_set: Set[int] = set(true_neighbors)
        neighbor_set: Set[int] = set(neighbors)
        intersection: Set[int] = true_set.intersection(neighbor_set)
        precision: float = len(intersection) / len(neighbor_set) if len(neighbor_set) > 0 else 0.0
        recall: float = len(intersection) / len(true_set) if len(true_set) > 0 else 0.0
        metrics.full_precision += precision
        metrics.full_recall += recall

        # MLGT Hash-only precision/recall
        hash_neighbor_set: Set[int] = set(hash_neighbors)
        hash_intersection: Set[int] = neighbor_set.intersection(hash_neighbor_set)
        hash_precision: float = len(hash_intersection) / len(hash_neighbor_set) if len(hash_neighbor_set) > 0 else 0.0
        hash_recall: float = len(hash_intersection) / len(neighbor_set) if len(neighbor_set) > 0 else 0.0
        metrics.mlgt_hash_precision += hash_precision
        metrics.mlgt_hash_recall += hash_recall

        # Hash-True precision/recall
        true_intersection: Set[int] = true_set.intersection(hash_neighbor_set)
        true_precision: float = len(true_intersection) / len(hash_neighbor_set) if len(hash_neighbor_set) > 0 else 0.0
        true_recall: float = len(true_intersection) / len(true_set) if len(true_set) > 0 else 0.0
        metrics.hash_true_precision += true_precision
        metrics.hash_true_recall += true_recall

    metrics.avg_over(num_queries)
    return metrics


if __name__ == "__main__":
    parser = ArgumentParser(description="Test MLGT SAFFRON on datasets")
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["imagenet", "imdb_wiki", "insta_1m", "mirflickr"], 
        required=True, 
        help="Name of the dataset to test (required; choices: imagenet, imdb_wiki, insta_1m, mirflickr)"
    )
    parser.add_argument(
        "--num_data",
        type=int,
        default=-1,
        help="Number of data points to use (all by default)"
    )
    parser.add_argument(
        "--num_queries", 
        type=int, 
        default=100, 
        help="Number of queries to test (100 by default)"
    )
    parser.add_argument(
        "--num_tables",
        type=int,
        default=500,
        help="Number of tables to use (500 by default)"
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=10,
        help="Number of bits per table (10 by default)"
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=100,
        help="Number of neighbors to consider (100 by default)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=50,
        help="Threshold for SAFFRON decoding (50 by default)"
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=0,
        help="Debug level: 0 (disabled) to 3 (verbose) (0 by default)"
    )
    args: Namespace = parser.parse_args()
    metrics: Metrics = test_dataset(
        dataset_name=args.dataset,
        num_data=args.num_data,
        num_queries=args.num_queries,
        num_tables=args.num_tables,
        num_bits=args.num_bits,
        num_neighbors=args.num_neighbors,
        threshold=args.threshold,
        debug=args.debug
    )
    print(f"Results for dataset: {args.dataset}")
    print(f"Full MLGT SAFFRON - Precision: {metrics.full_precision:.4f}, Recall: {metrics.full_recall:.4f}, Avg Time: {metrics.full_time:.4f} sec")
    print(f"MLGT vs Hash - Precision: {metrics.mlgt_hash_precision:.4f}, Recall: {metrics.mlgt_hash_recall:.4f}, Avg Time: {metrics.hash_time:.4f} sec")
    print(f"Hash vs True - Precision: {metrics.hash_true_precision:.4f}, Recall: {metrics.hash_true_recall:.4f}, Avg Time: {metrics.true_time:.4f} sec")
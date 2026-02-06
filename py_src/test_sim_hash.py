#!/usr/bin/env python3

import os
import sys
import gc
from argparse import ArgumentParser, Namespace

import math
import cmath
import random
import time

from tqdm import tqdm, trange
from typing import List, Tuple, Dict, Set, Union, Optional, Any, Iterable, Callable, NamedTuple
from dataclasses import dataclass, field

import numpy as np
from numpy import array, ndarray, linalg, matrix, strings, testing
import numba

# Add the parent directory to sys.path to allow imports from py_src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from py_src.sim_hash import SimHash, test_simhash

DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]


if __name__ == "__main__":
    parser = ArgumentParser(description="Test SimHash recall on datasets.")
    parser.add_argument("--num_bits", type=int, default=10, help="Number of bits per hash (default: 10)")
    # threshold arg removed
    parser.add_argument("--num_hashes", type=int, default=500, help="Number of hash functions/tables (default: 500)")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of queries to test (default: 100)")
    parser.add_argument("--k", type=int, default=100, help="Number of neighbors (K) (default: 100)")
    parser.add_argument("--dataset", type=str, default="all", choices=DATASETS + ["all"], help="Dataset to use (default: all)")
    
    args: Namespace = parser.parse_args()
    
    datasets_to_process = DATASETS if args.dataset == "all" else [args.dataset]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    for dataset_name in datasets_to_process:
        print(f"\n{'='*40}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*40}")
        
        dataset_path = os.path.join(data_dir, dataset_name)
        X_path = os.path.join(dataset_path, "X.npy")
        Q_path = os.path.join(dataset_path, "Q.npy")
        
        # Load dataset directly
        try:
            dataset = np.load(X_path)
            print(f"  Loaded Dataset shape: {dataset.shape}")
        except Exception as e:
            print(f"Failed to load dataset {X_path}: {e}")
            continue

        # Load queries (limit to requested number)
        try:
            queries_full = np.load(Q_path)
            print(f"  Loaded Queries shape: {queries_full.shape}")
            queries = queries_full[:args.num_queries]
            if queries.shape[0] < args.num_queries:
                print(f"  Warning: only {queries.shape[0]} queries available (requested {args.num_queries})")
        except Exception as e:
            print(f"Failed to load queries {Q_path}: {e}")
            continue
            
        D = dataset.shape[1]
        if queries.shape[1] != D:
            print(f"Dimension mismatch: Dataset {D}, Queries {queries.shape[1]}")
            continue
        
        print(f"Initializing SimHash (Tables={args.num_hashes}, Bits={args.num_bits}, Dim={D})...")
        simhash = SimHash(num_hashes=args.num_hashes, num_bits=args.num_bits, dimension=D)
        
        print("Running similarity search test...")
        start_time = time.time()
        mean_recall = test_simhash(dataset, queries, simhash, num_neighbors=args.k)
        end_time = time.time()
        
        print(f"Mean Recall for {dataset_name}: {mean_recall:.4f}")
        print(f"Time taken: {end_time - start_time:.2f}s")
        
        del dataset
        del queries
        gc.collect()

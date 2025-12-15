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


class SimHash:
    """
    SimHash: Locality-Sensitive Hashing using random projection hyperplanes.
    
    Uses signed random projections to create binary hash codes for vectors.
    That is, for each table and a hash bit, a random vector 
    (Gaussian distributed, of size `dimension`) is used to project the input vector. 
    The sign of the projection determines the bit value (0 or 1).
    
    Methods:
        - `__init__`: Initialize with hash parameters and generate random planes
        - `hash`: Compute hash value for a vector using a specific hash table
        - `__call__`: Convenience method to hash vectors (single or all tables)
        - `compare_hashes`: Compare two hash signatures and return matches based on Hamming distance
    """
    def __init__(
            self,
            num_hashes: int,
            num_bits: int,
            threshold: int,
            dimension: int
    ) -> None:
        """
        Initialize the SimHash object.

        Args:
            num_hashes (int): Number of hash tables (or hash functions).
            num_bits (int): Number of bits per hash signature.
            threshold (int): Hamming distance threshold for matching.
            dimension (int): Dimensionality of the input vectors.
        """
        self.num_hashes = num_hashes
        self.num_bits = num_bits
        self.threshold = threshold
        self.dimension = dimension

        self.hash_planes = np.random.randn(self.num_hashes, self.num_bits, self.dimension).astype(np.float64)
        self.bit_vals: ndarray = np.array([1 << i for i in range(self.num_bits)], dtype=np.int32)
    
    def hash(
            self,
            vector: ndarray,
            hash_indices: Union[int, List[int], ndarray] = []
    ) -> ndarray:
        """
        Compute hash values for the input vector(s).

        Args:
            vector (ndarray): Input vector (1D) or dataset (ND; last dim = dim for the vectors).
            hash_indices (Optional[Union[int, List[int], ndarray]]): Indices of hash tables to use. 
                                                                     If None, uses all tables.

        Returns:
            ndarray: Boolean array representing the hash signatures.
        """
        required_hash_planes: ndarray = self.hash_planes[hash_indices] if hash_indices != [] else self.hash_planes
        # vector: (N..., D)
        # planes: (T, B, D)
        # result: (N..., T, B)
        dot_products: ndarray = np.tensordot(vector, required_hash_planes, axes=([vector.ndim - 1], [2]))
        return (dot_products > 0).astype(np.bool)
    
    def __call__(self, vector: ndarray, hash_indices: Union[int, List[int], ndarray] = []) -> ndarray:
        """
        Convenience method to hash vectors using all hash tables.

        Args:
            vector (ndarray): Input vector (1D) or dataset (ND; last dim = dim for the vectors).
            hash_indices (Union[int, List[int], ndarray]): Indices of hash tables to use. 
                                                           If None, uses all tables.

        Returns:
            ndarray: Boolean array representing the hash signatures for all tables.
        """
        return self.hash(vector, hash_indices=hash_indices)

    def compare_hashes(
            self,
            hash1: ndarray,
            hash2: ndarray,
    ) -> ndarray:
        """
        Compare two sets of hash signatures and determine matches based on Hamming distance.
        Efficiently computes all-pairs comparison between hash1 and hash2.
        
        Args:
            hash1 (ndarray): Dataset hashes. Shape (N..., T, B).
            hash2 (ndarray): Query hashes. Shape (T, B).

        Returns:
            ndarray: Boolean array indicating matches. Shape (N..., T).
                Match in table T iff Hamming distance <= threshold.
        """
        # Validate shapes
        if hash2.shape != (self.num_hashes, self.num_bits):
            raise ValueError(f"Bad query hash dimentions: expected ({self.num_hashes}, {self.num_bits}), got {hash2.shape}")
        if hash1.shape[-2:] != (self.num_hashes, self.num_bits):
            raise ValueError(f"Bad dataset hash dimentions: expected ({', '.join(hash1.shape[:-2])}, {self.num_hashes}, {self.num_bits}), got {hash1.shape}")
        
        comparisons: ndarray = (hash1 != hash2).astype(np.int32)
        hamming_distances: ndarray = comparisons.sum(axis=-1)
        matches: ndarray = (hamming_distances <= self.threshold).astype(np.bool)
        return matches
    
    def hash_bits_to_value(
            self,
            hash_bits: ndarray
    ) -> ndarray:
        """
        Convert boolean hash bits to integer representation.

        Args:
            hash_bits (ndarray): Boolean array of shape (N..., T, B).

        Returns:
            ndarray: Integer representation of the hash bits.
        """
        return np.dot(hash_bits.astype(np.int32), self.bit_vals)
    
    def hash_value_to_bits(
            self,
            hash_values: int | ndarray
    ) -> ndarray:
        """
        Convert integer hash value to boolean bits.

        Args:
            hash_value (int): Integer representation of the hash.

        Returns:
            ndarray: Boolean array of shape (T, B) representing the hash bits.
        """
        if isinstance(hash_values, int) or hash_values.shape == ():
            hash_bits: ndarray = (self.bit_vals & int(hash_values) > 0).astype(np.bool) 
            assert hash_bits.shape == (self.num_bits,), \
                f"Expected hash bits shape ({self.num_bits}), got {hash_bits.shape}"
            return hash_bits
        
        hash_vals_shape: Tuple[int, ...] = hash_values.shape
        hash_values = hash_values.reshape(hash_vals_shape + (1,))
        hash_bits: ndarray = ((self.bit_vals & hash_values) > 0).astype(np.bool)
        assert hash_bits.shape == hash_vals_shape + (self.num_bits,), \
            f"Expected hash bits shape ({', '.join(map(str, hash_vals_shape))}, {self.num_bits}), got {hash_bits.shape}"
        return hash_bits



def test_simhash(
        dataset: ndarray,
        query_set: ndarray,
        hash_function: SimHash,
        num_neighbors: int
) -> float:
    """
    Measure the mean recall of SimHash-based similarity search.

    Args:
        dataset (ndarray): The dataset of vectors to search in. Shape (N, D).
        query_set (ndarray): The set of query vectors. Shape (Q, D).
        hash_function (SimHash): The initialized SimHash object.
        num_neighbors (int): The number of nearest neighbors to retrieve (k).

    Returns:
        float: The mean recall over all queries.
    """
    # Precompute dataset hashes
    # dataset_hashes: (N, T, B)
    dataset_hashes = hash_function.hash(dataset)
    
    recalls = []
    
    # Normalize dataset for ground truth calculation (cosine similarity)
    # Avoid division by zero
    dataset_norms = np.linalg.norm(dataset, axis=1, keepdims=True)
    dataset_normalized = dataset / (dataset_norms + 1e-10)

    for i in tqdm(range(query_set.shape[0]), desc="Running similarity search test for queries..."):
        query = query_set[i]
        # query_hash: (T, B)
        query_hash = hash_function.hash(query)
        
        # Ground truth: Cosine similarity
        query_norm = np.linalg.norm(query)
        query_normalized = query / (query_norm + 1e-10)
        
        # Cosine similarity: (N,)
        similarities = np.dot(dataset_normalized, query_normalized)
        
        # Get top k indices by dot cosine similarities
        true_neighbors = np.argsort(similarities)[-num_neighbors:]
        
        # SimHash candidates
        matches = hash_function.compare_hashes(dataset_hashes, query_hash) # (N, T)
        
        # Count top k indices by hash matches
        num_matches = matches.sum(axis=1)  # (N,)
        candidate_indices = np.argsort(num_matches)[-num_neighbors:]

        # Compute recall
        true_set = set(true_neighbors)
        candidate_set = set(candidate_indices)
        intersection_set = true_set.intersection(candidate_set)
        recall = len(intersection_set) / num_neighbors
        recalls.append(recall)
        gc.collect()
        
    return float(np.mean(recalls))


if __name__ == "__main__":
    # Provide a small dataset-based test runner similar to the old `test_sim_hash.py`.
    DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]

    parser = ArgumentParser(description="Test SimHash recall on datasets.")
    parser.add_argument("--num_bits", type=int, default=10, help="Number of bits per hash (default: 10)")
    parser.add_argument("--threshold", type=int, default=0, help="Hamming distance threshold (default: 0)")
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

        print(f"Initializing SimHash (Tables={args.num_hashes}, Bits={args.num_bits}, Threshold={args.threshold}, Dim={D})...")
        simhash = SimHash(num_hashes=args.num_hashes, num_bits=args.num_bits, threshold=args.threshold, dimension=D)

        print("Running similarity search test...")
        start_time = time.time()
        mean_recall = test_simhash(dataset, queries, simhash, num_neighbors=args.k)
        end_time = time.time()

        print(f"Mean Recall for {dataset_name}: {mean_recall:.4f}")
        print(f"Time taken: {end_time - start_time:.2f}s")

        del dataset
        del queries
        gc.collect()
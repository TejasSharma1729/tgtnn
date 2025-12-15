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
from typing import List, Tuple, Dict, Set, Union, Optional, Literal, Any, Iterable, Callable, NamedTuple
from dataclasses import dataclass, field

import numpy as np
from numpy import array, ndarray, linalg, matrix, strings, testing
import numba
from numba.typed import List as NumbaList, Dict as NumbaDict
from numba import types
import scipy.sparse

from sim_hash import SimHash, test_simhash

# Type aliases matching the reference
Array0D = ndarray
Array1D = ndarray
Array2D = ndarray
Array3D = ndarray

# Define Numba types globally to avoid inference errors inside njit
nb_int64 = types.int64
nb_int32 = types.int32
nb_list_int64 = types.ListType(types.int64)
nb_list_int32 = types.ListType(types.int32)

# Define types for CSR-like inverted index
nb_int32_arr = types.int32[:]
# (hashes, offsets, indices)
nb_table_struct = types.Tuple((nb_int32_arr, nb_int32_arr, nb_int32_arr))
nb_pool_struct = types.ListType(nb_table_struct)
nb_index_struct = types.ListType(nb_pool_struct)

CURRENT_PATH: str = os.path.dirname(os.path.abspath(__file__))
REPO_PATH: str = os.path.dirname(CURRENT_PATH)
DATASET_DIR: str = os.path.join(REPO_PATH, "data")


@numba.njit(parallel=True)
def create_random_pools(
        num_features: int,
        num_pools: int, 
        points_per_pool: int,
        pools_per_point: int
) -> Tuple[Array2D, Array2D]:
    """
    Create random pool assignment where each point appears in exactly pools_per_point pools.
    Uses a greedy randomized approach to minimize pool overlaps.
    """
    # Validate parameters
    if points_per_pool * num_pools < pools_per_point * num_features:
        # Numba doesn't support f-strings in raise
        raise ValueError("Inconsistent pooling parameters: Capacity < Demand")
    
    pools = np.full((num_pools, points_per_pool), -1, dtype=np.int64)
    point_pool_count = np.zeros(num_features, dtype=np.int64)
    pooling_matrix = np.zeros((num_pools, num_features), dtype=np.bool_)
    
    # Create assignments array directly
    total_assignments = num_features * pools_per_point
    point_assignments = np.empty(total_assignments, dtype=np.int64)
    
    curr_idx = 0
    for point_idx in range(num_features):
        for _ in range(pools_per_point):
            point_assignments[curr_idx] = point_idx
            curr_idx += 1
    
    print("Creating random pool assignment...")
    np.random.shuffle(point_assignments)
    pool_fill_count = np.zeros(num_pools, dtype=np.int64)

    for idx in range(len(point_assignments)):
        point_idx = point_assignments[idx]
        # Try random pools first
        selected_pool = -1
        for _ in range(100):
            p = np.random.randint(0, num_pools)
            if pool_fill_count[p] < points_per_pool and not pooling_matrix[p, point_idx]:
                selected_pool = p
                break
        if selected_pool == -1:
            # Fallback: linear search
            candidates = np.where((pool_fill_count < points_per_pool) & (~pooling_matrix[:, point_idx]))[0]
            if len(candidates) == 0:
                raise ValueError("Cannot assign point to pool (capacity exceeded)")
            selected_pool = candidates[np.random.randint(0, len(candidates))]

        pools[selected_pool, pool_fill_count[selected_pool]] = point_idx
        pooling_matrix[selected_pool, point_idx] = True
        pool_fill_count[selected_pool] += 1
        point_pool_count[point_idx] += 1
        if idx % 100000 == 0:
            print("Assigned", idx, "points")
    return pools, pooling_matrix


@numba.njit
def build_inverted_index(
    hash_features: Array2D,
    pools: Array2D
) -> NumbaList:
    """
    Build inverted index for hash features within each pool.
    Returns a NumbaList of NumbaLists of Tuples (CSR-like structure).
    Outer: pools, Middle: tables, Inner: (hashes, offsets, indices).
    """
    num_pools, points_per_pool = pools.shape
    num_tables = hash_features.shape[1]
    
    # Outer list: List of pools
    inverted_hash_tables = NumbaList()
    
    print("Building per-pool inverted index (CSR optimized)...")
    for pool_idx in range(num_pools):
        if pool_idx % 500 == 0:
            print("Built", pool_idx, "pool indices")
        # Inner list: List of tables for this pool
        pool_inverted_tables = NumbaList()
        pool_points = pools[pool_idx]
        
        for table_idx in range(num_tables):
            # Collect valid points
            valid_indices = []
            for p in pool_points:
                if p != -1:
                    valid_indices.append(p)
            
            if len(valid_indices) > 0:
                hashes = np.empty(len(valid_indices), dtype=np.int32)
                points = np.empty(len(valid_indices), dtype=np.int32)
                
                for i, p in enumerate(valid_indices):
                    hashes[i] = hash_features[p, table_idx]
                    points[i] = p
                
                # Sort by hash
                sort_idx = np.argsort(hashes)
                sorted_hashes = hashes[sort_idx]
                sorted_points = points[sort_idx]
                
                # Manual unique and offsets (np.unique return_index not supported in Numba)
                num_unique = 1
                for i in range(1, len(sorted_hashes)):
                    if sorted_hashes[i] != sorted_hashes[i-1]:
                        num_unique += 1
                
                hashes_arr = np.empty(num_unique, dtype=np.int32)
                offsets_arr = np.empty(num_unique + 1, dtype=np.int32)
                indices_arr = sorted_points
                
                curr_h = sorted_hashes[0]
                hashes_arr[0] = curr_h
                offsets_arr[0] = 0
                
                u_idx = 1
                for i in range(1, len(sorted_hashes)):
                    h = sorted_hashes[i]
                    if h != curr_h:
                        offsets_arr[u_idx] = i
                        hashes_arr[u_idx] = h
                        curr_h = h
                        u_idx += 1
                offsets_arr[u_idx] = len(sorted_hashes)
                
                # Append tuple
                pool_inverted_tables.append((hashes_arr, offsets_arr, indices_arr))
            else:
                # Empty table
                empty_arr = np.empty(0, dtype=np.int32)
                pool_inverted_tables.append((empty_arr, np.array([0], dtype=np.int32), empty_arr))
            
        inverted_hash_tables.append(pool_inverted_tables)
        
    return inverted_hash_tables


@numba.njit
def popcount(n: int) -> int:
    """
    Count the number of set bits in an integer.
    """
    c = 0
    while n > 0:
        c += 1
        n &= n - 1
    return c


@numba.njit
def does_pool_match_query_hash(
    query_hashes: Array1D,
    inverted_index: NumbaList,
    hash_threshold: int,
    match_threshold: int
) -> bool:
    """
    Check if a pool matches the query hash using the per-pool inverted index (CSR).
    """
    # Use a dictionary for sparse counting since point indices are global and large
    # Key: point_idx, Value: count
    match_counts = NumbaDict.empty(nb_int32, nb_int32)
    
    num_tables = len(query_hashes)
    
    for table_idx in range(num_tables):
        query_hash = query_hashes[table_idx]
        # Unpack tuple
        hashes, offsets, indices = inverted_index[table_idx]
        
        if len(hashes) == 0:
            continue
            
        if hash_threshold == 0:
            # Binary search
            idx = np.searchsorted(hashes, query_hash)
            if idx < len(hashes) and hashes[idx] == query_hash:
                start = offsets[idx]
                end = offsets[idx+1]
                for i in range(start, end):
                    p = indices[i]
                    c = match_counts.get(p, nb_int32(0)) + nb_int32(1)
                    match_counts[p] = c
                    if c >= match_threshold:
                        return True
        else:
            # Iterate all hashes
            for i in range(len(hashes)):
                h_val = hashes[i]
                # Hamming distance
                xor_val = h_val ^ query_hash
                dist = 0
                while xor_val > 0:
                    dist += 1
                    xor_val &= xor_val - 1
                
                if dist <= hash_threshold:
                    start = offsets[i]
                    end = offsets[i+1]
                    for k in range(start, end):
                        p = indices[k]
                        c = match_counts.get(p, nb_int32(0)) + nb_int32(1)
                        match_counts[p] = c
                        if c >= match_threshold:
                            return True
                    
    return False


@numba.njit(parallel=True)
def find_matching_points(
    hash_features: Array2D,
    query_hashes: Array1D,
    hash_threshold: int,
    match_threshold: int
) -> List[int]:
    """
    Find all points that match the query hash criteria.
    """
    num_points, num_tables = hash_features.shape
    matching_indices = []
    
    for i in range(num_points):
        matches = 0
        for t in range(num_tables):
            h_feat = hash_features[i, t]
            h_query = query_hashes[t]
            
            # Compute Hamming distance
            xor_val = h_feat ^ h_query
            dist = 0
            while xor_val > 0:
                dist += 1
                xor_val &= xor_val - 1
            
            if dist <= hash_threshold:
                matches += 1
        
        if matches >= match_threshold:
            matching_indices.append(i)
            
    matching_indices.sort()
    return matching_indices


@numba.njit
def my_matching_algo(
    pooling_matrix: ndarray,
    positive_pools: ndarray
) -> ndarray:
    """
    My implementation of matching algorithm, to solve for sparse vector x given y = Ax, 
        where y is the positive_pools and A is the pooling_matrix.
    
    :param pooling_matrix: The pooling matrix A
    :type pooling_matrix: ndarray
    :param positive_pools: The output vector y
    :type positive_pools: ndarray
    :return: The recovered sparse vector x
    :rtype: ndarray
    """
    num_pools, num_features = pooling_matrix.shape
    assert positive_pools.shape == (num_pools,), "positive_pools shape mismatch"

    possible = np.ones(num_features, dtype=np.int8)
    candidates = np.zeros(num_features, dtype=np.int8)
    for pool_idx in range(num_pools):
        if positive_pools[pool_idx] == 0:
            for j in range(num_features):
                if pooling_matrix[pool_idx, j]:
                    possible[j] = 0

    while True:
        newly_identified = np.zeros(num_features, dtype=np.int8)
        for pool_idx in range(num_pools):
            if positive_pools[pool_idx] == 0:
                continue
            pool_points = np.zeros(num_features, dtype=np.int8)
            for j in range(num_features):
                if pooling_matrix[pool_idx, j] and possible[j]:
                    pool_points[j] = 1
            if np.sum(pool_points) == 1:
                for j in range(num_features):
                    if pool_points[j]:
                        newly_identified[j] = 1
        if np.sum(newly_identified) == 0:
            break
        for j in range(num_features):
            if newly_identified[j]:
                candidates[j] = 1
                possible[j] = 0
        # Identify all pools that contain any newly identified points
        identified_pools = np.zeros(num_pools, dtype=np.int8)
        for pool_idx in range(num_pools):
            for j in range(num_features):
                if pooling_matrix[pool_idx, j] and newly_identified[j]:
                    identified_pools[pool_idx] = 1
                    break
        for pool_idx in range(num_pools):
            if identified_pools[pool_idx]:
                positive_pools[pool_idx] = 0
    return candidates


@numba.njit(parallel=True)
def OMP(
    A: ndarray,
    y: ndarray,
    s: int = 100,
) -> ndarray:
    """
    Solves the orthogonal matching pursuit (OMP) problem for sparse x, y = Ax.

    Args:
    - A: The pooling matrix (m x n)
    - y: The output vector (the pool tests, m)
    - s: Maximum sparsity of x
    """
    m, n = A.shape
    y = y.astype(np.float64)
    x = np.zeros(n, dtype=np.float64)
    residual = y.copy()
    support = np.zeros(s, dtype=np.int64)
    support_size = 0
    
    for iteration in range(s):
        if np.linalg.norm(residual) < 1e-10:
            break
        correlations = A.T @ residual
        best_idx = np.argmax(np.abs(correlations))
        
        # Check if already in support
        already_in = False
        for i in range(support_size):
            if support[i] == best_idx:
                already_in = True
                break
        if already_in:
            break
            
        support[support_size] = best_idx
        support_size += 1
        
        # Solve least squares: (A_s^T A_s) x_s = A_s^T y using normal equations
        A_s = A[:, support[:support_size]]
        ATA = A_s.T @ A_s
        ATy = A_s.T @ y
        x_s = np.linalg.solve(ATA, ATy)
        
        # Update residual
        residual = y - A_s @ x_s
    
    # Build sparse solution
    for i in range(support_size):
        A_s = A[:, support[:support_size]]
        ATA = A_s.T @ A_s
        ATy = A_s.T @ y
        x_s = np.linalg.solve(ATA, ATy)
        for j in range(support_size):
            x[support[j]] = x_s[j]
        break
    
    return x


class MLGT:
    """
    Multi-Level Group Testing (MLGT): Efficient approximate nearest neighbor search
    combining group testing theory with locality-sensitive hashing.
    
    Adapted to use the imported SimHash class.
    """
    def __init__(
            self,
            num_tables: int = 500,
            hash_bits: int = 10,
            input_dim: int = 1000,
            hash_threshold: int = 2,
            match_threshold: int = 20,
            num_pools: int = 5000,
            pools_per_point: int = 3,
            points_per_pool: int = 600,
            features: Optional[Array2D] = None,
            features_path: Optional[str] = None,
            use_srp: bool = False,
            threshold: Optional[int] = None,
    ) -> None:
        """
        Initialize the MLGT instance with parameters for the SimHash index.

        Args:
            num_tables (int): Number of hash tables.
            hash_bits (int): Number of bits per hash.
            input_dim (int): Dimension of input vectors.
            hash_threshold (int): Hamming distance threshold for hash matching.
            match_threshold (int): Minimum number of table matches for a point to be considered a candidate.
            num_pools (int): Number of pools for group testing.
            pools_per_point (int): Number of pools each point is assigned to.
            points_per_pool (int): Number of points per pool.
            features (Optional[Array2D]): Feature matrix (N, D).
            features_path (Optional[str]): Path to feature matrix file.
            use_srp (bool): Whether to use Sparse Random Projections (not implemented).
            threshold (Optional[int]): Alias for hash_threshold.
        """
        # Validate parameters
        self.num_tables = num_tables
        self.hash_bits = hash_bits
        self.input_dim = input_dim
        # support alias `threshold` for backwards compatibility
        self.hash_threshold = threshold if threshold is not None else hash_threshold
        self.match_threshold = match_threshold
        self.num_pools = num_pools
        self.pools_per_point = pools_per_point
        self.points_per_pool = points_per_pool
        self.use_srp = use_srp
        
        # Pooling capacity check deferred to build_index (dataset-dependent)

        # Load features if provided
        self.features: Optional[Array2D] = features
        self.num_features: int = 0
        if self.features is not None:
            assert self.features.ndim == 2, "Features must be a 2D array."
            assert self.features.shape[1] == input_dim, f"Features must have dimension {input_dim}, got {self.features.shape[1]}."
            self.num_features = self.features.shape[0]
        elif features_path is not None:
            if not os.path.exists(features_path):
                raise ValueError(f"Features path {features_path} does not exist.")
            self.features = np.load(features_path)
            assert self.features.ndim == 2, "Features must be a 2D array."
            assert self.features.shape[1] == input_dim, f"Features must have dimension {input_dim}, got {self.features.shape[1]}."
            self.num_features = self.features.shape[0]
        else:
            raise ValueError("Atleast one of `features` or `features_path` must be provided.")
        
        # Initialize other attributes
        self.hash_function: Optional[SimHash] = None
        self.hash_features: Optional[Array2D] = None
        self.pools: Optional[Array2D] = None
        self.pooling_matrix: Optional[Array2D] = None
        self.inverted_hash_tables: NumbaList = NumbaList()

        # Initialize the SimHash index
        self.hash_function = SimHash(
            num_hashes=num_tables, 
            num_bits=hash_bits, 
            threshold=self.hash_threshold, 
            dimension=input_dim
        )
        
        if self.hash_threshold >= self.hash_bits:
            print(f"WARNING: hash_threshold ({self.hash_threshold}) >= hash_bits ({self.hash_bits}). "
                  "This will cause all points to match, leading to slow queries and poor recall.")


    
    def build_index(self) -> None:
        """
        Build the index for the dataset using the SimHash class.
        """
        # Ensure capacity is sufficient
        required_capacity = self.pools_per_point * self.num_features
        current_capacity = self.num_pools * self.points_per_pool
        if current_capacity < required_capacity:
            new_ppp = math.ceil(required_capacity / self.num_pools) + 10 # Add buffer
            print(f"Warning: Adjusting points_per_pool from {self.points_per_pool} to {new_ppp} to satisfy demand.")
            self.points_per_pool = int(new_ppp)

        # Use pure Python version with tqdm (Numba removed)
        self.pools, self.pooling_matrix = create_random_pools(
            num_features=self.num_features,
            num_pools=self.num_pools,
            points_per_pool=self.points_per_pool,
            pools_per_point=self.pools_per_point
        )
        print(f"DEBUG: Pools shape: {self.pools.shape}, Pooling matrix shape: {self.pooling_matrix.shape}")
        
        # Optimization: Use int32 for hash features to save memory (sufficient for hash values)
        self.hash_features: Array2D = self.hash_function.hash_bits_to_value(self.hash_function(self.features)).astype(np.int32)
        print(f"DEBUG: Hash features shape: {self.hash_features.shape}")
        gc.collect()

        print(f"DEBUG: Computing inverted hash tables...")
        self.inverted_hash_tables = build_inverted_index(
            hash_features=self.hash_features,
            pools=self.pools
        )

        # Integrity checks / assertions: ensure pooling matrix dimensions and counts are reasonable
        assert self.pooling_matrix.shape == (self.num_pools, self.num_features), (
            f"pooling_matrix shape mismatch: expected ({self.num_pools}, {self.num_features}), got {self.pooling_matrix.shape}"
        )
        # Each point should appear in approximately `pools_per_point` pools (allow off-by-one due to ceil)
        per_point_counts = np.sum(self.pooling_matrix, axis=0)
        if self.pools_per_point is not None:
            min_expected = max(0, self.pools_per_point - 1)
            max_expected = self.pools_per_point + 1
            assert np.all((per_point_counts >= min_expected) & (per_point_counts <= max_expected)), (
                f"Per-point pool counts outside expected range [{min_expected},{max_expected}] (example counts: {per_point_counts[:10]})"
            )
        # Each pool should contain approximately `points_per_pool` points
        per_pool_counts = np.sum(self.pooling_matrix, axis=1)
        expected_avg = (self.num_features * self.pools_per_point) / self.num_pools
        # Allow some variance, especially if points_per_pool was padded
        min_pool = int(expected_avg * 0.8)
        max_pool = self.points_per_pool
        assert np.all((per_pool_counts >= min_pool) & (per_pool_counts <= max_pool)), (
            f"Per-pool point counts outside expected range [{min_pool},{max_pool}] (example counts: {per_pool_counts[:10]})"
        )
    
    def _solve_scipy(self, positive_pools: ndarray, method: str = "lsmr") -> ndarray:
        """
        Solve the group testing problem using SciPy's sparse solvers (LSMR or LSQR).
        Returns indices of candidates (threshold > 0.5).
        """
        from scipy.sparse import csr_matrix
        
        A_sparse = csr_matrix(self.pooling_matrix, dtype=float)
        y = positive_pools.astype(float)
        
        if method == "lsmr":
            from scipy.sparse.linalg import lsmr
            sol = lsmr(A_sparse, y)
            x = sol[0]
        elif method == "lsqr":
            from scipy.sparse.linalg import lsqr
            sol = lsqr(A_sparse, y)
            x = sol[0]
        else:
            raise ValueError(f"Unknown method: {method}")
            
        x = np.nan_to_num(x)
        return np.where(x > 0.5)[0]
    
    def _solve_omp(self, positive_pools: ndarray, sparsity: int = 500) -> ndarray:
        """
        Solve the group testing problem using Orthogonal Matching Pursuit (OMP).
        Returns indices of candidates (threshold > 0.5).
        """
        A = self.pooling_matrix.astype(float)
        y = positive_pools.astype(float)
        
        x = OMP(A, y, s=sparsity)
        x = np.nan_to_num(x)
        return np.where(x > 0.5)[0]

    def query(
            self, 
            query_vector: Array1D,
            algorithm: Literal["my_algo", "saffron", "grotesque", "lsmr", "lsqr", "omp"] = "my_algo",
            top_k: int = 100
    ) -> List[int]:
        """
        Query the MLGT index with a given query vector.
        Returns top_k candidates ranked by hash match scores.
        """
        positive_pools: ndarray = np.zeros((self.num_pools,), dtype=bool)
        query_hash = self.hash_function(query_vector)
        query_hash_values = self.hash_function.hash_bits_to_value(query_hash).astype(np.int32)

        for pool_idx in tqdm(range(self.num_pools), desc="Querying pools...", leave=False):
            positive_pools[pool_idx] = does_pool_match_query_hash(
                query_hashes=query_hash_values,
                inverted_index=self.inverted_hash_tables[pool_idx],
                hash_threshold=self.hash_threshold,
                match_threshold=self.match_threshold
            )
        
        candidates: List[int] = []
        if algorithm == "my_algo":
            candidates_mask = my_matching_algo(
                pooling_matrix=self.pooling_matrix,
                positive_pools=positive_pools
            )
            candidates = np.where(candidates_mask)[0]
        elif algorithm == "omp":
            candidates = self._solve_omp(positive_pools, sparsity=500)
        elif algorithm in ["lsmr", "lsqr"]:
            candidates = self._solve_scipy(positive_pools, method=algorithm)
        elif algorithm == "saffron":
            raise NotImplementedError("Saffron algorithm not implemented in this version.")
        elif algorithm == "grotesque":
            raise NotImplementedError("GROTESQUE algorithm not implemented in this version.")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Rank by hash match scores and return top K
        if len(candidates) > 0:
            candidate_matches: ndarray = (self.hash_features[candidates] == query_hash_values)
            candidate_scores: ndarray = candidate_matches.astype(np.int32).sum(axis=1)
            sorted_idx = np.argsort(-candidate_scores)[:top_k]
            topk_candidates = [candidates[idx] for idx in sorted_idx]
        else:
            # Fallback: return top K by hash scores
            hash_match_scores: ndarray = (self.hash_features == query_hash_values).astype(np.int32).sum(axis=1)
            topk_candidates = np.argsort(-hash_match_scores)[:top_k].tolist()

        return topk_candidates
    
    def __call__(self, query_vector: Array1D) -> List[int]:
        return self.query(query_vector)

    def get_hash_neighbors(self, query_vector: Array1D, num_neighbors: int = 100) -> List[int]:
        """
        Get all neighbors that satisfy the hash matching criteria (threshold).
        """
        query_hash: ndarray = self.hash_function(query_vector)
        query_hash_values: ndarray = self.hash_function.hash_bits_to_value(query_hash)
        hash_matches: ndarray = np.sum(self.hash_features == query_hash_values, axis=1)
        return np.argsort(-hash_matches)[:num_neighbors].tolist()

    def precision_and_recall(
        self, 
        query_vector: Array1D, 
        mlgt_neighbors: List[int], 
        brute_force_neighbors: List[int], 
        num_neighbors: int = 100
    ) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score.
        """
        s_mlgt = set(mlgt_neighbors)
        s_true = set(brute_force_neighbors)
        
        intersection = len(s_mlgt.intersection(s_true))
        
        precision = intersection / len(s_mlgt) if len(s_mlgt) > 0 else 1.0
        recall = intersection / len(s_true) if len(s_true) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def brute_force_search(self, query_vector: Array1D, num_neighbors: int = 100) -> List[int]:
        """
        Perform brute-force search to find the top-k nearest neighbors.
        """
        similarities = self.features @ query_vector / (linalg.norm(self.features, axis=1) * linalg.norm(query_vector) + 1e-10)
        topk_indices = np.argsort(-similarities)[:num_neighbors]
        return topk_indices.tolist()
    
    def get_metrics(
            self, 
            query_vector: Array1D, 
            algorithm: str = "my_algo"
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """
        Get precision & recall (final), precicion & recall (search vs hash-matches), precicion & recall (hash-matches vs true) and times
        """
        query_start = time.time()
        saffron_indices = self.query(query_vector, algorithm=algorithm)
        query_end = time.time()

        hash_start = time.time()
        # Get candidates that satisfy BOTH thresholds (hash_threshold and match_threshold)
        # This is the "Ground Truth" for the Group Testing problem
        hash_threshold_neighbors = self.get_hash_neighbors(query_vector, num_neighbors=100)
        hash_end = time.time()

        true_start = time.time()
        true_topk = self.brute_force_search(query_vector, num_neighbors=100)
        true_end = time.time()

        # Metrics
        set_saff = set(saffron_indices)
        set_true = set(true_topk)
        set_hash_thresh = set(hash_threshold_neighbors)

        # 1. Precision: How many of Saffron's output are in True Top 100?
        final_precision = len(set_saff.intersection(set_true)) / max(len(set_saff), 1)
        
        # 2. Recall: How many of True Top 100 did Saffron find?
        final_recall = len(set_saff.intersection(set_true)) / 100.0
        
        # 3. Recall (Saffron vs Hash Thresholds): Did Saffron find the points that satisfy the thresholds?
        # This measures the quality of the Group Testing recovery (Saffron)
        denom_saff_hash = len(set_hash_thresh)
        recall_search_vs_hash = len(set_saff.intersection(set_hash_thresh)) / denom_saff_hash if denom_saff_hash > 0 else 1.0
        precision_search_vs_hash = len(set_saff.intersection(set_hash_thresh)) / max(len(set_saff), 1)

        # 4. Recall (Hash Thresholds vs True): Do the points satisfying thresholds contain the True Top 100?
        # This measures the quality of the LSH parameters (tables, bits, thresholds)
        recall_hash_vs_true = len(set_hash_thresh.intersection(set_true)) / 100.0
        precision_hash_vs_true = len(set_hash_thresh.intersection(set_true)) / max(len(set_hash_thresh), 1)

        # 5. F1 for Saffron vs True
        f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0.0

        # --- Debug prints per query ---
        print("-- Hash vs True (LSH quality) --"
              f" Precision: {precision_hash_vs_true:.4f},"
              f" Recall: {recall_hash_vs_true:.4f},"
              f" Sizes: |hash|={len(set_hash_thresh)}, |true|={len(set_true)}, |∩|={len(set_hash_thresh.intersection(set_true))}")

        print("-- MLGT vs Hash (Recovery quality) --"
              f" Precision: {precision_search_vs_hash:.4f},"
              f" Recall: {recall_search_vs_hash:.4f},"
              f" Sizes: |mlgt|={len(set_saff)}, |hash|={len(set_hash_thresh)}, |∩|={len(set_saff.intersection(set_hash_thresh))}")

        print("-- MLGT vs True (End-to-end) --"
              f" Precision: {final_precision:.4f},"
              f" Recall: {final_recall:.4f},"
              f" F1: {f1:.4f},"
              f" Sizes: |mlgt|={len(set_saff)}, |true|={len(set_true)}, |∩|={len(set_saff.intersection(set_true))}")

        return (
            final_precision,
            final_recall, 
            recall_search_vs_hash, 
            precision_search_vs_hash, 
            recall_hash_vs_true, 
            precision_hash_vs_true, 
            query_end - query_start, 
            hash_end - hash_start, 
            true_end - true_start
        )


def dataset_runner(
        args: Namespace,
        dataset: str = "imagenet",
) -> None:
    dataset_path: str = os.path.join(DATASET_DIR, dataset)
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist. Please download the dataset first.")
    print(f"Running MLGT on dataset: {dataset} with parameters: {args}")
    mlgt = MLGT(
        num_tables=args.num_tables,
        hash_bits=args.hash_bits,
        input_dim=args.input_dim,
        hash_threshold=args.hash_threshold,
        match_threshold=args.match_threshold,
        num_pools=args.num_pools,
        pools_per_point=args.pools_per_point,
        features_path=os.path.join(dataset_path, "X.npy")
    )
    mlgt.build_index()
    print(f"Index built for dataset {dataset} with {mlgt.num_features} features.")

    query_set: Array2D = np.load(os.path.join(dataset_path, "Q.npy"))
    precisions = []
    recalls = []
    f1s = []
    recalls_saff_hash = []
    precs_saff_hash = []
    recalls_hash_true = []
    precs_hash_true = []
    times = []
    for qidx in tqdm(range(min(args.num_queries, query_set.shape[0])), desc="Testing on queries..."):
        query_vector: Array1D = query_set[qidx]
        precision, recall, recall_saff_hash, prec_saff_hash, recall_hash_true, prec_hash_true, t_saff, t_hash, t_true = mlgt.get_metrics(query_vector, algorithm=args.algorithm)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        recalls_saff_hash.append(recall_saff_hash)
        precs_saff_hash.append(prec_saff_hash)
        recalls_hash_true.append(recall_hash_true)
        precs_hash_true.append(prec_hash_true)
        times.append(t_saff)
        print(
            f"Query {qidx}: {args.algorithm} time {t_saff:.4f}s | "
            f"Precision: {precision:.4f}, Recall({args.algorithm} vs true): {recall:.4f}, "
            f"Recall(hash vs true): {recall_hash_true:.4f}, Recall({args.algorithm} vs hash): {recall_saff_hash:.4f}"
        )
    if len(precisions) > 0:
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_f1 = sum(f1s) / len(f1s)
        avg_recall_saff_hash = sum(recalls_saff_hash) / len(recalls_saff_hash)
        avg_prec_saff_hash = sum(precs_saff_hash) / len(precs_saff_hash)
        avg_recall_hash_true = sum(recalls_hash_true) / len(recalls_hash_true)
        avg_prec_hash_true = sum(precs_hash_true) / len(precs_hash_true)
        avg_time = sum(times) / len(times)
        print("\n==== Aggregate Statistics ====")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Average Recall(MLGT vs hash): {avg_recall_saff_hash:.4f}")
        print(f"Average Precision(MLGT vs hash): {avg_prec_saff_hash:.4f}")
        print(f"Average Recall(hash vs true): {avg_recall_hash_true:.4f}")
        print(f"Average Precision(hash vs true): {avg_prec_hash_true:.4f}")
        print(f"Average Query Time: {avg_time:.4f}s")

if __name__ == "__main__":
    # Example usage
    parser = ArgumentParser(description="MLGT Example")
    parser.add_argument("--num_tables", type=int, default=500, help="Number of hash tables")
    parser.add_argument("--hash_bits", type=int, default=10, help="Number of bits per hash")
    parser.add_argument("--input_dim", type=int, default=1000, help="Input vector dimension")
    parser.add_argument("--hash_threshold", type=int, default=0, help="Hamming distance threshold")
    parser.add_argument("--match_threshold", type=int, default=50, help="Minimum number of table matches")
    parser.add_argument("--num_pools", type=int, default=5000, help="Number of pools")
    parser.add_argument("--pools_per_point", type=int, default=10, help="Number of pools per point")
    parser.add_argument("--num_queries", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--algorithm", type=str, choices=["my_algo", "saffron", "grotesque", "lsmr", "lsqr"], default="my_algo", help="Recovery algorithm to use")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "imdb_wiki", "insta_1m", "mirflickr", "all"], default="all", help="Dataset to use")
    args = parser.parse_args()

    if args.dataset in ["imagenet", "all"]:
        dataset_runner(args, dataset="imagenet")
    if args.dataset in ["imdb_wiki", "all"]:
        dataset_runner(args, dataset="imdb_wiki")
    if args.dataset in ["insta_1m", "all"]:
        dataset_runner(args, dataset="insta_1m")
    if args.dataset in ["mirflickr", "all"]:
        dataset_runner(args, dataset="mirflickr")
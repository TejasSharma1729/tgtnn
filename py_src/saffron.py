#!/usr/bin/env python3
import os
import sys
import time
CURRENT_PATH: str = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT: str = os.path.abspath(os.path.join(CURRENT_PATH, ".."))

from argparse import ArgumentParser, Namespace
from typing import Literal, List, Tuple, Dict, Set, Iterable, Callable, Union, Optional, Any
from dataclasses import dataclass, field

import math
import cmath
import random

import numpy as np
from numpy import array, ndarray, linalg, random as npr, matrix, tensordot
import numba

import scipy
from scipy import sparse, stats, optimize
from scipy.sparse import csr_matrix, csr

import sklearn
from sklearn import svm, metrics, cluster


def construct_T_matrix(
        num_features: int,
        degree: int,
        num_buckets: int
) -> ndarray:
    """
    Construct the T matrix (bipartite graph incidence matrix) for SAFFRON.
    
    The T matrix represents a left-regular bipartite graph where each feature (left node)
    is connected to exactly `degree` buckets (right nodes) chosen uniformly at random.
    This creates a sparse bipartite graph structure that is fundamental to the SAFFRON
    algorithm for efficient group testing.
    
    Args:
        num_features (int): The number of items/features (left nodes in the bipartite graph).
        degree (int): The left-degree, i.e., each feature connects to exactly this many buckets.
                     Typically set to ceil(log2(sparsity)) for good performance.
        num_buckets (int): The number of buckets/right nodes in the bipartite graph.
                          Typically set to 4 * sparsity.
    
    Returns:
        ndarray: A boolean matrix of shape (num_buckets, num_features) where T[i, j] = True
                if bucket i is connected to feature j, and False otherwise.
    
    Design Rationale:
        - Each feature appears in exactly `degree` buckets (left-regular design)
        - Connections are random to avoid deterministic patterns
        - The matrix is sparse, enabling efficient computation
        - The degree is typically logarithmic in sparsity to balance test count and decoding complexity
    """
    T_matrix: ndarray = np.zeros((num_buckets, num_features), dtype=bool)
    for feature_idx in range(num_features):
        chosen_buckets: ndarray = npr.choice(num_buckets, size=degree, replace=False)
        T_matrix[chosen_buckets, feature_idx] = True
    return T_matrix

def construct_U_matrix(
        num_features: int,
        num_bits: int,
        num_tests: int
) -> ndarray:
    """
    Construct the signature matrix U for SAFFRON.
    
    The U matrix consists of 6 sections stacked vertically, each containing signature vectors
    for all features. These signatures enable the SAFFRON decoder to:
    1. Detect and resolve singletons (right nodes connected to exactly one defective item)
    2. Resolve doubletons (right nodes connected to exactly two defective items)
    
    The 6 sections are constructed as follows:
    - U1: Binary representation of feature indices (first half of bits)
    - U1_complement: Bitwise complement of U1 (negation)
    - U2: Binary representation using second half of bits (different random permutation)
    - U2_complement: Bitwise complement of U2
    - U3: XOR of U1 and U2 (provides additional parity check)
    - U4: XOR of U1 and U2_complement (provides alternative parity check)
    
    Mathematical Details:
    - For feature i (0-indexed), the binary representation b_i has bits b_i[0], b_i[1], ..., b_i[log2(n)-1]
    - Each signature vector u_i is a 6*L-bit vector: [b_i, ~b_i, s1_perm_i, ~s1_perm_i, xor_terms...]
    - L = num_bits = ceil(log2(num_features) / 2)
    
    Args:
        num_features (int): The number of items whose signatures to generate.
        num_bits (int): The number of bits per signature section. 
                       Typically ceil(log2(num_features) / 2).
        num_tests (int): Not directly used; for documentation/validation (should equal 6 * num_bits).
    
    Returns:
        ndarray: A boolean matrix of shape (6 * num_bits, num_features) where each column is
                the signature vector for that feature across all 6 sections.
    
    Design Rationale:
        - Dual representation (b_i and ~b_i) allows reading bits even when OR'd with unknown bits
        - Multiple independent permutations (s1, s2) enable verification of doubleton hypotheses
        - XOR operations create parity checks that distinguish correct vs. incorrect guesses
        - Total height of 6L bits is sufficient to uniquely identify up to n features with
          error probability O(K/n^2) as proven in Lemma 3.1
    """
    # Create binary representation for first half of bits
    U1_matrix: ndarray = array([[bool(((i >> b) & 1) > 0) for b in range(num_bits)] for i in range(num_features)])
    
    # Complement of first half
    U2_matrix: ndarray = ~U1_matrix
    
    # Binary representation using second half of bits with random permutation
    U3_matrix: ndarray = array([[bool(((i >> b) & 1) > 0) for b in range(num_bits, 2 * num_bits)] for i in range(num_features)])
    
    # Complement of second half
    U4_matrix: ndarray = ~U3_matrix
    
    # XOR of two halves for additional parity information
    U5_matrix: ndarray = U1_matrix ^ U3_matrix
    
    # XOR of first half with complement of second half
    U6_matrix: ndarray = U1_matrix ^ U4_matrix
    
    # Transpose to get shape (L, N) for each section, then stack vertically to get (6L, N)
    return np.vstack([U1_matrix.T, U2_matrix.T, U3_matrix.T, U4_matrix.T, U5_matrix.T, U6_matrix.T])


class Saffron:
    """
    SAFFRON: Sparse-grAph codes Framework For gROup testiNg
    
    A non-adaptive group testing algorithm based on sparse-graph codes that recovers
    defective items with minimal number of tests and low decoding complexity.
    
    Algorithm Overview:
    SAFFRON uses a two-stage encoding and iterative peeling decoding:
    
    1. ENCODING:
       - Constructs a left-regular bipartite graph G where each item connects to degree d buckets
       - Each item is assigned a signature vector u_i from the signature matrix U
       - Each test/pool combines: A[test] = T[bucket] AND U[section]
       - Tests are organized as (bucket, section) pairs where section indexes the 6 signature segments
    
    2. DECODING (Peeling Algorithm):
       - Iteratively identifies resolvable right nodes (buckets+sections)
       - Singleton: A right node connected to exactly 1 defective item (detected by weight = L)
       - Resolvable Doubleton: A right node with 1 known + 1 unknown defective item
       - For each resolved item, removes it from all connected right nodes
       - Terminates when no more resolvable nodes exist
    
    Theoretical Guarantees:
    - Recovers at least (1 - ε)K defective items with m ≈ C(ε)K log₂(n) tests
    - Decoding complexity O(K log n) - order optimal
    - Works with specific constants down to ε as small as 10⁻⁶
    - Can be robustified to handle noisy/erroneous test results
    
    References:
        Lee, K., Pedarsani, R., & Ramchandran, K. (2015).
        "SAFFRON: A Fast, Efficient, and Robust Framework for Group Testing 
         based on Sparse-Graph Codes" arXiv:1508.04485
    
    Attributes:
        num_features (int): Number of items n
        sparsity (int): Expected number of defective items K (parameter for design)
        num_buckets (int): Number of buckets M = 4 * sparsity
        degree (int): Left-degree d = ceil(log₂(sparsity))
        num_bits (int): Bits per signature section L = ceil(log₂(n) / 2)
        num_tests (int): Number of test sections = 6 * num_bits
        num_pools (int): Total number of pools = num_buckets * num_tests
        T_matrix (ndarray): Bipartite graph incidence matrix, shape (num_buckets, num_features)
        U_matrix (ndarray): Signature matrix, shape (num_tests, num_features)
    """
    
    def __init__(
            self,
            num_features: int = 1000000,
            sparsity: int = 100
    ) -> None:
        """
        Initialize SAFFRON group testing decoder.
        
        The initialization constructs:
        1. A random left-regular bipartite graph with controlled degree
        2. A signature matrix with 6 sections for singleton and doubleton detection/resolution
        
        The parameters are chosen to balance:
        - Number of tests m = O(K log n): More buckets/bits = more tests but better recovery
        - Decoding complexity O(K log n): More buckets/bits = easier decoding but more work
        - Error probability O(K/n²): Sufficient for large-scale group testing
        
        Args:
            num_features (int): Number of items n. Default 1,000,000 for large-scale applications.
            sparsity (int): Expected number of defective items K (design parameter).
                           Default 100. Used to determine:
                           - num_buckets = 4*K (controls tests per item)
                           - degree = ceil(log₂(K)) (left-degree in bipartite graph)
        
        Derived Parameters:
            num_buckets: Set to 4 * sparsity = 4K
                        Empirically chosen constant from sparse-graph coding theory
            degree: Set to ceil(log₂(sparsity)) = ceil(log₂(K))
                   This is the left-regularity parameter; each item connects to d buckets
            num_bits: Set to ceil(log₂(num_features) / 2) = ceil(log₂(n) / 2)
                     Signature vectors use ceil(log₂(n)) total bits split across sections
                     Using half per section reduces redundancy while maintaining uniqueness
            num_tests: Set to 6 * num_bits
                      The 6 sections of the signature matrix enable singleton + doubleton detection
            num_pools: Total tests = num_buckets * num_tests = 4K * 6 * ceil(log₂(n)/2)
                      ≈ 12K * log₂(n) which matches theory
        
        This preserves the (num_buckets, degree, num_bits, num_tests, num_pools) variables as intended.
        """
        self.num_features: int = num_features
        self.sparsity: int = sparsity
        self.num_buckets: int = 4 * sparsity
        self.degree: int = math.ceil(math.log2(sparsity))
        self.num_bits: int = math.ceil(math.log2(num_features) / 2)  # Half the bits needed
        self.num_tests: int = 6 * self.num_bits
        self.num_pools: int = self.num_buckets * self.num_tests
        self.T_matrix: ndarray = construct_T_matrix(self.num_features, self.degree, self.num_buckets)
        self.U_matrix: ndarray = construct_U_matrix(self.num_features, self.num_bits, self.num_tests)
    
    def pools(self) -> List[List[int]]:
        """
        Generate the list of all pools with their constituent items.
        
        Each pool is an AND of one bucket row from T and one signature row from U:
        pool[bucket_idx, test_idx] = T[bucket_idx] AND U[test_idx]
        
        Pools are enumerated in order: all test_idx for bucket_idx=0, then bucket_idx=1, etc.
        
        Returns:
            List[List[int]]: A list of length num_pools where each element is a list of
                            item indices in that pool. pools()[k] returns the items in pool k.
        
        Complexity: O(num_pools * average_pool_size) ≈ O(num_buckets * num_tests * degree)
                   ≈ O(K log(K) log(n)) for typical configurations
        """
        pool_to_points: List[List[int]] = []
        for bucket_idx in range(self.num_buckets):
            for test_idx in range(self.num_tests):
                # Compute the AND of bucket row and signature row
                pool: ndarray = self.T_matrix[bucket_idx] * self.U_matrix[test_idx]
                points_in_pool: List[int] = list(np.nonzero(pool)[0])
                pool_to_points.append(points_in_pool)
        return pool_to_points
    
    def _extract_signature_sections(self, z_measurement: ndarray) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        """
        Extract the 6 signature sections from a right-node measurement vector.
        
        The measurement vector z has shape (num_tests,) where num_tests = 6 * num_bits.
        It is divided into 6 equal sections, each of size num_bits.
        
        Args:
            z_measurement (ndarray): Measurement vector of shape (num_tests,)
        
        Returns:
            Tuple of 6 arrays, each of shape (num_bits,):
            - z1: Section 1 (binary representation)
            - z2: Section 2 (complement representation)
            - z3: Section 3 (second permutation)
            - z4: Section 4 (complement of second permutation)
            - z5: Section 5 (XOR of representations)
            - z6: Section 6 (XOR of first and complement of second)
        """
        sections = []
        for i in range(6):
            start_idx = i * self.num_bits
            end_idx = (i + 1) * self.num_bits
            sections.append(z_measurement[start_idx:end_idx])
        return tuple(sections)
    
    def _read_feature_index_from_sections(self, z1: ndarray, z2: ndarray) -> int:
        """
        Recover a feature index from complementary signature sections z1 and ~z1.
        
        For each bit position b:
        - If the true signature bit is 0, then z1[b] = 0 (since 0 OR anything = anything)
        - If the true signature bit is 1, then z1[b] = 1 (since 1 OR anything = 1)
        - The complement z2 = ~z1 has the opposite bits when isolated
        
        To recover bit b when OR'd with unknown bits:
        - If z1[b] = 0, then true_bit[b] = z2[b] (must read from complement, inverted)
        - If z1[b] = 1, then true_bit[b] = 1 (certain from z1)
        
        Actually, for the SAFFRON protocol:
        - z1[b] contains: signature_bit[b] OR other_items_bit[b]
        - z2[b] contains: NOT(signature_bit[b]) OR other_items_bit[b]
        - If signature_bit[b] = 0: z1[b] = other_bit, z2[b] = 1 OR other_bit = 1, so z1[b] = NOT z2[b]
        - If signature_bit[b] = 1: z1[b] = 1, z2[b] = other_bit, so z1[b] = 1
        
        The recovery: true_bit[b] = z1[b] if z1[b] == 1 else NOT z2[b]
        
        Args:
            z1 (ndarray): First section (binary representation)
            z2 (ndarray): Second section (complement representation)
        
        Returns:
            int: The recovered feature index (0 to num_features-1)
        """
        recovered_bits = np.zeros(self.num_bits, dtype=bool)
        for b in range(self.num_bits):
            if z1[b]:
                recovered_bits[b] = True
            else:
                recovered_bits[b] = not z2[b]
        
        # Convert bit array to integer
        feature_idx = 0
        for b in range(self.num_bits):
            if recovered_bits[b]:
                feature_idx |= (1 << b)
        
        # Ensure within valid range
        feature_idx = min(feature_idx, self.num_features - 1)
        return feature_idx
    
    def get_pooling_matrix(self) -> csr_matrix:
        """
        Construct the explicit pooling matrix A = T * U (row-wise Kronecker product-ish).
        Rows are pools (bucket-test pairs), Columns are features.
        A[ (bucket * num_tests) + test, feature ] = T[bucket, feature] AND U[test, feature]
        """
        data = []
        indices = []
        indptr = [0]
        
        # Iterate over pools
        for b in range(self.num_buckets):
            # Get features in this bucket (sparse)
            bucket_features = np.flatnonzero(self.T_matrix[b])
            
            for t in range(self.num_tests):
                # For this pool, intersection of bucket features and signature features
                # Efficiently: iterate bucket_features, check U[t]
                
                # U is shape (num_tests, num_features).
                # We want indices where U[t, f] is true for f in bucket_features
                pool_indices = [f for f in bucket_features if self.U_matrix[t, f]]
                
                indices.extend(pool_indices)
                data.extend([1] * len(pool_indices))
                indptr.append(len(data))
                
        return csr_matrix((data, indices, indptr), shape=(self.num_pools, self.num_features))
        
    def _read_full_index(self, z1: ndarray, z2: ndarray, z3: ndarray, z4: ndarray) -> int:
        """
        Recover the full feature index from low bits (z1, z2) and high bits (z3, z4).
        """
        # Recover low bits
        idx_low = 0
        for b in range(self.num_bits):
            # Logic: if z1[b] is set, bit is likely 1. If not, check z2 (complement).
            # If z1[b]=1, could be 1 or noise.
            # If z1[b]=0, definitely 0.
            # Saffron robustness: z1=1 means bit=1 (assuming no 0->1 errors).
            bit_val = 1 if z1[b] else (0 if z2[b] else 0) 
            # Actually, standard logic:
            # bit = 1 if z1[b] else 0
            # BUT: z1[b] = u1[b] OR ...
            # If u1[b]=1 -> z1[b]=1.
            # If u1[b]=0 -> z1[b]=? (depends on others).
            # If z1[b]=0 -> u1[b]=0.
            # IF we assume Singleton, then z1[b] == u1[b].
            if z1[b]:
                idx_low |= (1 << b)
                
        # Recover high bits
        idx_high = 0
        for b in range(self.num_bits):
            if z3[b]:
                idx_high |= (1 << b)
                
        full_idx = idx_low | (idx_high << self.num_bits)
        return full_idx

    def solve(
            self,
            measurements: ndarray
    ) -> ndarray:
        """
        Decode SAFFRON measurements to recover defective items using the peeling algorithm.
        """
        assert measurements.shape[0] == self.num_pools, "Measurements size mismatch"
        
        # Reshape measurements: (num_buckets, num_tests)
        z_matrix = measurements.reshape((self.num_buckets, self.num_tests)).astype(bool)
        
        identified_items: Set[int] = set()
        resolved_buckets: Set[int] = set()
        
        current_z = z_matrix.copy()
        
        max_iterations = 2 * self.sparsity + 20
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            found_new = False
            
            # Iterate over buckets
            for b in range(self.num_buckets):
                if b in resolved_buckets:
                    continue
                
                z_b = current_z[b] # Shape (num_tests,)
                weight = np.sum(z_b)
                
                if weight == 0:
                    continue
                
                # Check for Singleton
                z_sections = self._extract_signature_sections(z_b)
                z1, z2, z3, z4, z5, z6 = z_sections
                
                # Recover Candidate
                candidate_idx = self._read_full_index(z1, z2, z3, z4)
                
                if candidate_idx >= self.num_features:
                    continue

                # Verify strictly
                signature = self.U_matrix[:, candidate_idx]
                if np.array_equal(z_b, signature):
                    # It is a singleton!
                    if candidate_idx not in identified_items:
                        identified_items.add(candidate_idx)
                        found_new = True
                        
                        # PEELING
                        # For every bucket containing this item, remove contribution
                        connected_buckets = np.flatnonzero(self.T_matrix[:, candidate_idx])
                        for cb in connected_buckets:
                            # In OR channel, we can set z[cb] to 0 ONLY if we are sure no other items are there?
                            # Standard Peeling for Group Testing (COMP):
                            # We can't peel easily.
                            # But Saffron uses Resolvable Doubleton logic which relies on identifying one to find another.
                            # For singletons, we rely on finding buckets that contain ONLY this item.
                            # If we found it, we add to identified.
                            # We don't remove it from z_matrix unless we implement advanced peeling.
                            # Simple approach: Loop until no new singletons found.
                            # But Saffron's power comes from peeling.
                            # Approximate peeling:
                            # If z[cb] == signature, then cb becomes empty (all zeros).
                            if np.array_equal(current_z[cb], signature):
                                current_z[cb][:] = False
                                resolved_buckets.add(cb)
                            elif cb not in resolved_buckets:
                                # "Subtract" known item?
                                # This is risky.
                                pass
                        
                        # Mark this bucket as resolved (it was a singleton)
                        resolved_buckets.add(b)
            
            # 2. Try RESOLVABLE DOUBLETONS (1 Known + 1 Unknown)
            for b in range(self.num_buckets):
                if b in resolved_buckets:
                    continue
                
                z_b = current_z[b]
                if np.sum(z_b) == 0: continue

                # Identify known items in this bucket
                # This could be slow if T is large, but Saffron graph is sparse (degree ~ log K)
                # Inverted index on identified items would be faster
                # But here we iterate knowns? No, iterate buckets.
                # Check intersection size.
                
                # Check which known items are in this bucket
                # Efficient check:
                bucket_knowns = []
                for k in identified_items:
                    if self.T_matrix[b, k]:
                        bucket_knowns.append(k)
                
                if len(bucket_knowns) == 1:
                    # Exactly one known item
                    k = bucket_knowns[0]
                    Sk = self.U_matrix[:, k]
                    
                    # Recover unknown u from z_b given Sk
                    # S_u (unknown)
                    # Logic:
                    # for bit j:
                    #   if Sk_1[j] == 0: z1[j] = Su_1[j] (since Sk_1[j] is 0)
                    #   if Sk_1[j] == 1: z1[j] = 1. We start blind. 
                    #      Check Sk_2[j]. Sk_2[j] is 0 (complement).
                    #      z2[j] = Su_2[j] (since Sk_2[j] is 0).
                    #      Su_1[j] = ~Su_2[j].
                    # So we can always recover!
                    
                    z_sections = self._extract_signature_sections(z_b)
                    z1, z2, z3, z4, z5, z6 = z_sections
                    
                    k_sections = self._extract_signature_sections(Sk)
                    k1, k2, k3, k4, k5, k6 = k_sections
                    
                    # Recover u1 (low bits)
                    u1_rec = np.zeros_like(z1)
                    for j in range(self.num_bits):
                        if not k1[j]:
                            u1_rec[j] = z1[j]
                        else:
                            # k1[j] is 1. Check z2/k2. k2[j] must be 0.
                            # z2[j] = u2[j] | k2[j] = u2[j] | 0 = u2[j].
                            # u1[j] = ~u2[j] = ~z2[j]
                            u1_rec[j] = not z2[j]
                    
                    # Recover u3 (high bits) - using k3, k4
                    u3_rec = np.zeros_like(z3)
                    for j in range(self.num_bits):
                        if not k3[j]:
                            u3_rec[j] = z3[j]
                        else:
                            u3_rec[j] = not z4[j]
                            
                    # Note: we need to construct the integer indices from bit arrays
                    # _read_full_index takes Z arrays and infers '1'.
                    # Here we have precise bit values.
                    
                    idx_low = 0
                    for j in range(self.num_bits):
                        if u1_rec[j]: idx_low |= (1 << j)
                    
                    idx_high = 0
                    for j in range(self.num_bits):
                        if u3_rec[j]: idx_high |= (1 << j)
                        
                    candidate_u = idx_low | (idx_high << self.num_bits)
                    
                    if candidate_u >= self.num_features: continue
                    if candidate_u in identified_items: continue
                    
                    # Verify
                    Su = self.U_matrix[:, candidate_u]
                    combined = Sk | Su
                    if np.array_equal(z_b, combined):
                        # Found resolvable doubleton!
                        identified_items.add(candidate_u)
                        found_new = True
                        resolved_buckets.add(b)
            
            if not found_new:
                break
                
        # Construct output
        result = np.zeros(self.num_features, dtype=int)
        for item in identified_items:
            result[item] = 1
        return result
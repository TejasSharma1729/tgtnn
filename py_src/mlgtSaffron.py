
import os
import sys
import time
import numpy as np
from numpy import ndarray, array, linalg
from typing import List, Optional, Literal, Tuple
from scipy.sparse import csr_matrix
import gc
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

# Add python source directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mlgt import MLGT, identify_positive_pools, build_global_inverted_index
from saffron import Saffron
import numba
from numba.typed import List as NbList, Dict as NbDict

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(CURR_DIR)
DATASET_DIR = os.path.join(REPO_DIR, "data")


@numba.njit
def identify_candidate_pools(
        query_hashes: ndarray,
        T_global_index: NbList,
        T_pools: ndarray,
        U_global_index: NbList,
        U_pools: ndarray,
        num_features: int,
        match_threshold: int,
        feature_to_buckets: ndarray,
        feature_to_tests: ndarray,
        num_buckets: int,
        num_tests: int
) -> ndarray:
    """
    Identifies positive pools by checking intersection of T-Index and U-Index members.
    """
    num_tables = len(query_hashes)
    num_pools = num_buckets * num_tests
    
    # Store table matches for points. 
    # Logic: Only increment if point matches in BOTH T and U for the same table.
    point_matches = np.zeros(num_features, dtype=np.int16)
    
    # Scratchpad to track if a point was seen in T for the current table
    # Initialized to -1. Values will be current 'table_idx'.
    last_seen_T = np.full(num_features, -1, dtype=np.int16)
    
    for table_idx in range(num_tables):
        q_hash = query_hashes[table_idx]
        
        # Only proceed if hash exists in both indices
        if (q_hash in T_global_index[table_idx]) and (q_hash in U_global_index[table_idx]):
            
            # 1. Mark points matching in T-index
            t_bucket = T_global_index[table_idx][q_hash]
            t_p_ids = t_bucket[0] # Pool indices in T_pools
            t_l_ids = t_bucket[1] # Local indices in T_pools
            
            for i in range(len(t_p_ids)):
                p = T_pools[t_p_ids[i], t_l_ids[i]]
                if p != -1:
                    last_seen_T[p] = table_idx
            
            # 2. Check points matching in U-index
            u_bucket = U_global_index[table_idx][q_hash]
            u_p_ids = u_bucket[0]
            u_l_ids = u_bucket[1]
            
            for i in range(len(u_p_ids)):
                p = U_pools[u_p_ids[i], u_l_ids[i]]
                if p != -1:
                    # Logic: Intersection(T, U) for this table
                    if last_seen_T[p] == table_idx:
                        point_matches[p] += 1
                        
    # Map high-scoring points to Positive Pools
    positive_pools = np.zeros(num_pools, dtype=np.int8)
    
    for p in range(num_features):
        if point_matches[p] >= match_threshold:
            # Mark all pools this candidate belongs to
            # Iterate buckets
            for i in range(feature_to_buckets.shape[1]):
                b = feature_to_buckets[p, i]
                if b == -1: continue
                # Iterate tests
                for j in range(feature_to_tests.shape[1]):
                    t = feature_to_tests[p, j]
                    if t == -1: continue
                    
                    positive_pools[b * num_tests + t] = 1
                    
    return positive_pools




class MLGTSaffron(MLGT):
    """
    MLGT implementation using SAFFRON for group testing recovery.
    Overrides the pooling matrix construction and recovery algorithm.
    """
    
    def __init__(
            self,
            # MLGT params
            num_tables: int = 500,
            hash_bits: int = 10,
            input_dim: int = 1000,
            match_threshold: int = 20,
            # Saffron params
            sparsity: int = 100,
            # Common
            features: Optional[ndarray] = None,
            **kwargs
    ):
        """
        Initialize MLGT-Saffron.
        
        Args:
            sparsity (int): Expected number of candidates (K). 
                           Controls Saffron structure.
        """
        # Initialize base MLGT
        # We pass dummy values for pools/points_per_pool as they will be determined by Saffron
        super().__init__(
            num_tables=num_tables,
            hash_bits=hash_bits,
            input_dim=input_dim,
            match_threshold=match_threshold,
            num_pools=0, # Will be set by Saffron
            pools_per_point=0,
            points_per_pool=0,
            features=features,
            **kwargs
        )
        
        self.sparsity = sparsity
        self.saffron_decoder: Optional[Saffron] = None
        
    def build_index(self) -> None:
        """
        Build separate indices for T (Buckets) and U (Tests).
        Satisfies the '2 sub-indices' structure requirement.
        """
        assert self.features is not None, "Features must be provided"
        self.num_features = self.features.shape[0]
        
        print(f"Building Saffron structures for K={self.sparsity}, N={self.num_features}...")
        self.saffron_decoder = Saffron(
            num_features=self.num_features,
            sparsity=self.sparsity
        )
        
        # 1. Build T Pools (Buckets) and Feature->Bucket Map
        print("Constructing T-Index (Buckets)...")
        # Ensure T is CSR for row access
        T_dense = self.saffron_decoder.T_matrix
        T_csr = csr_matrix(T_dense)
        
        # Feature -> Buckets map (CSC column access is efficient for this)
        T_csc = T_csr.tocsc()
        self.feature_to_buckets = np.full((self.num_features, self.saffron_decoder.degree), -1, dtype=np.int32)
        
        # Fill feature_to_buckets
        # We can iterate cols of T (features)
        for f in range(self.num_features):
            buckets = T_csc.indices[T_csc.indptr[f]:T_csc.indptr[f+1]]
            count = min(len(buckets), self.saffron_decoder.degree)
            self.feature_to_buckets[f, :count] = buckets[:count]

        # Build T pools for Inverted Index
        max_bucket_size = np.max(np.diff(T_csr.indptr))
        self.T_pools = np.full((self.saffron_decoder.num_buckets, max_bucket_size), -1, dtype=np.int64)
        
        for b in range(self.saffron_decoder.num_buckets):
            pts = T_csr.indices[T_csr.indptr[b]:T_csr.indptr[b+1]]
            self.T_pools[b, :len(pts)] = pts
            
        # 2. Build U Pools (Tests) and Feature->Test Map
        print("Constructing U-Index (Tests)...")
        U_dense = self.saffron_decoder.U_matrix
        U_csr = csr_matrix(U_dense)
        U_csc = U_csr.tocsc()
        
        # Feature -> Tests map
        max_tests_per_pts = np.max(np.diff(U_csc.indptr))
        self.feature_to_tests = np.full((self.num_features, max_tests_per_pts), -1, dtype=np.int32)
        # Also store counts for intersection sum
        self.feature_to_tests_count = np.empty(self.num_features, dtype=np.int32)
        
        for f in range(self.num_features):
            tests = U_csc.indices[U_csc.indptr[f]:U_csc.indptr[f+1]]
            length = len(tests)
            self.feature_to_tests[f, :length] = tests
            self.feature_to_tests_count[f] = length
            
        # Build U pools for Inverted Index
        # Rows of U have approx N/2 items. This is dense but fits in memory (60 * 500k ints = 120MB)
        max_test_size = np.max(np.diff(U_csr.indptr))
        self.U_pools = np.full((self.saffron_decoder.num_tests, max_test_size), -1, dtype=np.int64)
        for t in range(self.saffron_decoder.num_tests):
            pts = U_csr.indices[U_csr.indptr[t]:U_csr.indptr[t+1]]
            self.U_pools[t, :len(pts)] = pts

        # Compute hashes
        print("Computing hashes...")
        self.hash_features = self.hash_function.hash_bits_to_value(
            self.hash_function(self.features)
        ).astype(np.int32)
        gc.collect()
        
        # Build 2 Global Indices
        print("Building T Inverted Index...")
        self.T_index = build_global_inverted_index(
            hash_features=self.hash_features,
            pools=self.T_pools
        )
        print("Building U Inverted Index...")
        self.U_index = build_global_inverted_index(
            hash_features=self.hash_features,
            pools=self.U_pools
        )
        print("Indices built (Dual Index Structure).")


    def query(
            self, 
            query_vector: ndarray,
            algorithm: str = "saffron",
            top_k: int = 100
    ) -> List[int]:
        """
        Query using Dual-Index Saffron recovery.
        """
        assert self.saffron_decoder is not None, "Index not built"
        
        query_hash = self.hash_function(query_vector)
        query_hash_values = self.hash_function.hash_bits_to_value(query_hash).astype(np.int32)
        
        t0 = time.time()
        
        # 1. Identify Positive Pools (Saffron Measurements)
        # Uses explicit intersection of T and U indices as requested - Single Function
        saffron_measurements = identify_candidate_pools(
            query_hashes=query_hash_values,
            T_global_index=self.T_index,
            T_pools=self.T_pools,
            U_global_index=self.U_index,
            U_pools=self.U_pools,
            num_features=self.num_features,
            match_threshold=self.match_threshold,
            feature_to_buckets=self.feature_to_buckets,
            feature_to_tests=self.feature_to_tests,
            num_buckets=self.saffron_decoder.num_buckets,
            num_tests=self.saffron_decoder.num_tests
        )
        
        t1 = time.time()
        
        # 2. Recover using Saffron
        # Helper returns binary vector
        candidates_mask = self.saffron_decoder.solve(
            measurements=saffron_measurements.astype(int)
        )
        
        t2 = time.time()
        
        # 3. Retrieve candidates from mask
        final_candidates = np.flatnonzero(candidates_mask)
        
        # 4. Rank
        if len(final_candidates) > 0:
            candidate_matches = (self.hash_features[final_candidates] == query_hash_values)
            candidate_scores = candidate_matches.astype(np.int32).sum(axis=1)
            sorted_idx = np.argsort(-candidate_scores)[:top_k]
            topk_candidates = [final_candidates[idx] for idx in sorted_idx]
        else:
             topk_candidates = []
             
        self.last_pool_time = t1 - t0 
        self.last_solve_time = t2 - t1
        
        return topk_candidates

def dataset_runner_saffron(
        args: Namespace,
        dataset: str = "imagenet",
) -> None:
    dataset_path: str = os.path.join(DATASET_DIR, dataset)
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path {dataset_path} does not exist. Please download the dataset first.")
    
    print(f"Running MLGT-Saffron on dataset: {dataset} with parameters: {args}")
    
    # Saffron specific: sparsity
    # If not provided, default to 100 or something reasonable?
    # args.sparsity comes from parser
    
    mlgt = MLGTSaffron(
        num_tables=args.num_tables,
        hash_bits=args.hash_bits,
        input_dim=args.input_dim,
        match_threshold=args.match_threshold,
        sparsity=args.sparsity,
        features_path=os.path.join(dataset_path, "X.npy")
    )
    
    mlgt.build_index()
    print(f"Index built for dataset {dataset} with {mlgt.num_features} features.")

    query_set_path = os.path.join(dataset_path, "Q.npy")
    if not os.path.exists(query_set_path):
         print(f"Query file {query_set_path} not found. Generating random queries.")
         # Fallback for testing if no queries
         np.random.seed(42)
         query_set = np.random.randn(args.num_queries, args.input_dim).astype(np.float32)
    else:
         query_set = np.load(query_set_path)

    precisions = []
    recalls = []
    f1s = []
    times = []
    pool_times = []
    solve_times = []
    
    # Run queries
    num_queries = min(args.num_queries, query_set.shape[0])
    for qidx in tqdm(range(num_queries), desc="Testing on queries..."):
        query_vector = query_set[qidx]
        
        # We use 'algorithm' argument purely for logging, but MLGTSaffron.query defaults to saffron logic
        # But wait, MLGTSaffron.query takes 'algorithm' arg.
        # And MLGT.get_metrics calls self.query(..., algorithm=algorithm).
        # So we should call mlgt.get_metrics(...)
        
        # MLGTSaffron uses MLGT.get_metrics which calls self.query.
        # We need to ensure MLGTSaffron.query handles the algorithm string or ignores it.
        # My implementation of MLGTSaffron.query accepts algorithm="saffron".
        
        metrics = mlgt.get_metrics(query_vector, algorithm="saffron")
        
        # Unpack metrics (same structure as MLGT)
        (final_precision, final_recall, 
         recall_saff_hash, prec_saff_hash, 
         recall_hash_true, prec_hash_true, 
         t_total, t_hash, t_true, t_pool, t_solve) = metrics
         
        f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0.0
        
        precisions.append(final_precision)
        recalls.append(final_recall)
        f1s.append(f1)
        times.append(t_total)
        pool_times.append(t_pool)
        solve_times.append(t_solve)
        
        print(
            f"Query {qidx}: Saffron time {t_total:.4f}s (Pool: {t_pool:.4f}s, Solve: {t_solve:.4f}s) | "
            f"Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1: {f1:.4f}"
        )

    if len(precisions) > 0:
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        avg_f1 = sum(f1s) / len(f1s)
        avg_time = sum(times) / len(times)
        avg_pool = sum(pool_times) / len(pool_times)
        avg_solve = sum(solve_times) / len(solve_times)
        
        print("\n==== Aggregate Statistics (MLGT-Saffron) ====")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Average Query Time: {avg_time:.4f}s")
        print(f"  - Avg Pool Time:  {avg_pool:.4f}s")
        print(f"  - Avg Solve Time: {avg_solve:.4f}s")

if __name__ == "__main__":
    parser = ArgumentParser(description="MLGT-Saffron Runner")
    parser.add_argument("--num_tables", type=int, default=500, help="Number of hash tables")
    parser.add_argument("--hash_bits", type=int, default=10, help="Number of bits per hash")
    parser.add_argument("--input_dim", type=int, default=1000, help="Input vector dimension")
    parser.add_argument("--match_threshold", type=int, default=20, help="Minimum number of table matches")
    # Saffron specific
    parser.add_argument("--sparsity", type=int, default=100, help="Expected sparsity (K) for Saffron")
    
    parser.add_argument("--num_queries", type=int, default=10, help="Number of queries to run")
    parser.add_argument("--dataset", type=str, choices=["imagenet", "imdb_wiki", "insta_1m", "mirflickr", "all"], default="imagenet", help="Dataset to use")
    
    args = parser.parse_args()

    if args.dataset == "all":
        for ds in ["imagenet", "imdb_wiki", "insta_1m"]:
            try:
                dataset_runner_saffron(args, dataset=ds)
            except Exception as e:
                print(f"Skipping {ds}: {e}")
    else:
        dataset_runner_saffron(args, dataset=args.dataset)

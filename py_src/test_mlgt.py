
import sys
import os
import time
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from py_src.mlgt import MLGT

def test_mlgt_synthetic():
    print("Testing MLGT with synthetic data...")
    dim = 64
    N = 2000
    Q = 5
    
    # Generate random data
    np.random.seed(42)
    # Normalize features for cosine similarity context
    dataset = np.random.randn(N, dim).astype(np.float32)
    dataset /= np.linalg.norm(dataset, axis=1, keepdims=True)
    
    queries = np.random.randn(Q, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Initialize MLGT
    # Using parameters suitable for small test but exercising the global index
    mlgt = MLGT(
        num_tables=50,
        hash_bits=8,
        input_dim=dim,
        match_threshold=5,
        num_pools=100,
        pools_per_point=3,
        points_per_pool=100,
        features=dataset
    )
    
    print("Building index...")
    t0 = time.time()
    mlgt.build_index()
    t1 = time.time()
    print(f"Index built in {t1 - t0:.4f}s")
    
    print("Querying...")
    for i in range(Q):
        query = queries[i]
        t_start = time.time()
        # Ensure we test the default algorithm
        candidates = mlgt.query(query, top_k=10)
        t_end = time.time()
        print(f"Query {i}: Found {len(candidates)} candidates in {t_end - t_start:.4f}s")
        
    print("MLGT synthetic test passed.")

if __name__ == "__main__":
    test_mlgt_synthetic()

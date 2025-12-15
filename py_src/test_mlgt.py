
import sys
import os
import numpy as np
from argparse import ArgumentParser

# Add the parent directory to sys.path to allow imports from py_src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from py_src.sim_hash import SimHash
from py_src.mlgt import MLGT

def test_mlgt_synthetic():
    print("Testing MLGT with synthetic data...")
    dim = 16
    N = 1000
    Q = 10
    K = 5
    
    # Generate random data
    np.random.seed(42)
    dataset = np.random.randn(N, dim).astype(np.float32)
    queries = np.random.randn(Q, dim).astype(np.float32)
    
    # Initialize SimHash
    # Use small parameters for testing
    simhash = SimHash(num_hashes=10, num_bits=8, threshold=2, dimension=dim)
    
    # Initialize MLGT
    mlgt = MLGT(simhash, max_candidates=100, expansion_radius=1, use_cosine_refine=True)
    
    print("Fitting MLGT...")
    mlgt.fit(dataset)
    
    print("Querying MLGT...")
    for i in range(Q):
        query = queries[i]
        results = mlgt.query(query, k=K)
        print(f"Query {i}: Found {len(results)} results. Top score: {results[0][1] if results else 'N/A'}")
        
        # Basic sanity check: scores should be descending
        scores = [r[1] for r in results]
        if not all(scores[j] >= scores[j+1] for j in range(len(scores)-1)):
             print(f"  WARNING: Scores not descending for query {i}: {scores}")

    print("MLGT synthetic test passed (basic execution).")

if __name__ == "__main__":
    test_mlgt_synthetic()

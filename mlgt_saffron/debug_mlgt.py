#!/usr/bin/env python3
import os, sys, time
import numpy as np
from tqdm import tqdm
from mlgt_saffron import MLGTSaffron, SaffronIndex

def test_config(dataset, query_set, k, num_hashes, hash_bits, threshold, sparsity_mult=1.0):
    sparsity = int(k * sparsity_mult)
    print(f"\nTesting: hashes={num_hashes}, bits={hash_bits}, threshold={threshold}, sparsity_growth={sparsity_mult}x")
    try:
        start = time.time()
        idx = MLGTSaffron(dataset, sparsity, num_hashes, hash_bits, threshold, debug=0)
        idx_time = time.time() - start
        
        total_time = 0
        total_recall = 0
        num_queries = min(20, len(query_set))
        
        for i in range(num_queries):
            query = query_set[i]
            
            # True neighbors
            distances = dataset @ query
            true_indices = set(np.argsort(distances)[-k:])
            
            start_search = time.time()
            retrieved = set(idx.search(query))
            total_time += time.time() - start_search
            
            recall = len(retrieved.intersection(true_indices)) / k
            total_recall += recall
            
        avg_time = total_time / num_queries
        avg_recall = total_recall / num_queries
        
        print(f"Index Time: {idx_time:.2f}s, Avg Search Time: {avg_time:.4f}s, Avg Recall: {avg_recall:.4f}")
        return avg_time, avg_recall
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    dataset_name = "imagenet" 
    data_path = f"../data/{dataset_name}/"
    dataset = np.load(os.path.join(data_path, "X.npy"))
    query_set = np.load(os.path.join(data_path, "Q.npy"))
    k = 10
    
    # Normalizing dataset for dot product (cosine similarity)
    norms = np.linalg.norm(dataset, axis=1, keepdims=True)
    dataset = dataset / (norms + 1e-9)
    q_norms = np.linalg.norm(query_set, axis=1, keepdims=True)
    query_set = query_set / (q_norms + 1e-9)

    configs = [
        (50, 20, 20, 10.0), # Current best
        (32, 20, 13, 15.0), # Proposed: Less hashing, more pools
    ]
    
    for nh, hb, th, mult in configs:
        test_config(dataset, query_set, k, nh, hb, th, mult)

import numpy as np
import sys, os
from typing import List, Set
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

from mlgt_saffron import SaffronIndex

# Load small subset
data_path = "../data/imagenet"
dataset = np.load(os.path.join(data_path, "X.npy"))[:1000, :].astype(np.float32)
query_set = np.load(os.path.join(data_path, "Q.npy"))[:10, :].astype(np.float32)

print(f"Dataset shape: {dataset.shape}, Query shape: {query_set.shape}")
print(f"Dataset dtype: {dataset.dtype}, Query dtype: {query_set.dtype}")

# Create index with debug=1
saffron_index = SaffronIndex(dataset, num_neighbors=10, num_hash_bits=18, debug=1)

# Test first query
query = query_set[0]
print(f"\nQuerying with shape {query.shape}")

# Saffron search
retrieved_indices = saffron_index.search(query)
print(f"Saffron retrieved {len(retrieved_indices)} indices: {retrieved_indices[:20]}")

# Ground truth using inner product
scores = dataset @ query
true_indices = np.argsort(scores)[-10:][::-1]  # Top 10 in descending order
print(f"Ground truth top 10 (inner product): {true_indices}")

# Ground truth using L2 distance
distances = np.linalg.norm(dataset - query, axis=1)
true_indices_l2 = np.argsort(distances)[:10]
print(f"Ground truth top 10 (L2 distance): {true_indices_l2}")

# Check overlap
retrieved_set = set(retrieved_indices)
true_set_ip = set(np.argsort(scores)[-10:])
true_set_l2 = set(np.argsort(distances)[:10])

print(f"\nSaffron results: {len(retrieved_set)} items")
print(f"Overlap with IP ground truth: {len(retrieved_set & true_set_ip)}")
print(f"Overlap with L2 ground truth: {len(retrieved_set & true_set_l2)}")

# Inspect dataset stats
print(f"\nDataset statistics:")
print(f"Min: {dataset.min():.6f}, Max: {dataset.max():.6f}")
print(f"Mean: {dataset.mean():.6f}, Std: {dataset.std():.6f}")
print(f"Sparsity: {(dataset == 0).mean() * 100:.2f}%")

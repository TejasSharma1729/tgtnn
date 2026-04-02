import numpy as np
from scipy.sparse import csr_matrix
import time
import os
from tqdm import tqdm
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any


CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(CUR_DIR, "..", "data")
DATASETS: List[str] = ["sparse1M", "sparseFull", "movielens", "kddb", "avazu"]
HASHER_TYPES = ["MinHasher", "WeightedMinHasher", "BloomHashFunction", "SparseSRPHasher", "DenseSRPHasher"]
MLGT_SAFFRON_TYPES = ["MLGTSaffron" + suffix for suffix in ["Bloom", "MinHash", "WeightedMinHash", "SparseSRP", "DenseSRP"]]

def read_binary_csr(file_path: str) -> csr_matrix:
    """
    Reads a CSR matrix from the custom binary format used in this project.
    Confirmed format:
    - num_rows (uint64, 8 bytes)
    - num_cols (uint64, 8 bytes)
    - num_nonzero (uint64, 8 bytes)
    - indptr ((num_rows + 1) * uint64, 8 bytes each)
    - indices (num_nonzero * uint32, 4 bytes each)
    - values (num_nonzero * float32, 4 bytes each)
    """
    with open(file_path, 'rb') as f:
        num_vectors = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        dim = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        num_nonzero = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        
        indptr = np.frombuffer(f.read(8 * (num_vectors + 1)), dtype=np.uint64)
        indices = np.frombuffer(f.read(4 * num_nonzero), dtype=np.uint32)
        data = np.frombuffer(f.read(4 * num_nonzero), dtype=np.float32)
        
    return csr_matrix((data, indices, indptr), shape=(num_vectors, dim))


def get_dataset(dataset: str) -> Tuple[csr_matrix, csr_matrix]:
    """Loads X and Q for a given dataset name from the data directory."""
    path = os.path.join(DATA_DIR, dataset)
    x_path = os.path.join(path, "X.csr")
    q_path = os.path.join(path, "Q.csr")
    
    if not os.path.exists(x_path):
        # Try lowercase or other variations if needed, but here we assume exact match
        raise FileNotFoundError(f"Dataset X not found at {x_path}")
    if not os.path.exists(q_path):
        raise FileNotFoundError(f"Dataset Q not found at {q_path}")
        
    return read_binary_csr(x_path), read_binary_csr(q_path)


def calculate_recall(found_indices, ground_truth_indices, k):
    """
    Calculates recall@K.
    Since the SAFFRON index returns a set of candidates (up to target_sparsity),
    we check how many of the true top-K ground truth items are present in the 
    returned candidate set.
    """
    if not found_indices or len(ground_truth_indices) == 0:
        return 0.0
    
    found_set = set(found_indices)
    gt_set = set(ground_truth_indices[:k])
    intersection = found_set.intersection(gt_set)
    return len(intersection) / k
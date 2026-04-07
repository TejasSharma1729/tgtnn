#!/usr/bin/env python3
"""
Sanity-check for MinHasher: compares top-K recall between exact dot-product
ranking and hash-space dot-product ranking.
"""
import mlgt_sparse
from mlgt_sparse import Hasher, MLGTSaffron
HASHER_TYPES = ["MinHasher", "WeightedMinHasher", "BloomHashFunction", "SparseSRPHasher", "DenseSRPHasher"]
from dataset_utils import read_binary_csr, DATASETS

import numpy as np
from tqdm import tqdm
from numpy import array, ndarray
import os
from argparse import ArgumentParser


CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(CUR_DIR, "..", "data")


def hash_matrix(hasher: Hasher, M) -> np.ndarray:
    """Hash all rows of a CSR matrix, returning an (N, num_hashes) float array."""
    H = np.zeros((M.shape[0], hasher.num_hashes))
    for i in tqdm(range(M.shape[0]), desc="Hashing"):
        H[i] = array(hasher(M[i].data, M[i].indices, M[i].nnz))
    return H


def get_num_matches(X_hashes: ndarray, Q_hashes: ndarray, chunk: int = 64) -> ndarray:
    """Count per-hash-function collisions between every query and every dataset row.

    Returns an (nQ, nX) int array where entry [q, x] is the number of hash
    functions on which query q and item x produced the same value.

    Processes queries in chunks of `chunk` to bound peak memory to
    chunk * nX * H * 1 byte.
    """
    assert X_hashes.shape[1] == Q_hashes.shape[1]
    nQ, nX = Q_hashes.shape[0], X_hashes.shape[0]
    matches = np.empty((nQ, nX), dtype=np.int32)
    for i in tqdm(range(0, nQ, chunk), desc="Counting matches"):
        q_block = Q_hashes[i:i + chunk]           # (c, H)
        matches[i:i + chunk] = (
            q_block[:, None, :] == X_hashes[None, :, :]
        ).sum(axis=2)                              # (c, nX)
    return matches


def recall_at_k(exact_scores, hash_scores, k: int) -> float:
    """Average Recall@K between exact and hash-space top-K rankings."""
    score = 0.0
    for i in tqdm(range(exact_scores.shape[0]), "Recall"):
        top_k_exact = set(np.argsort(exact_scores[i])[-k:][::-1])
        top_k_hash  = set(np.argsort(hash_scores[i])[-k:][::-1])
        score += len(top_k_exact & top_k_hash) / k
    return score / exact_scores.shape[0]


def run(args):
    data_path = os.path.join(DATA_DIR, args.dataset)
    X = read_binary_csr(os.path.join(data_path, "X.csr"))
    Q = read_binary_csr(os.path.join(data_path, "Q.csr"))
    assert isinstance(X.shape, tuple) and len(X.shape) == 2
    assert isinstance(Q.shape, tuple) and len(Q.shape) == 2

    D = X.shape[1]
    t = args.hasher_type
    if t in ("MinHasher", "WeightedMinHasher"):
        hasher: Hasher = getattr(mlgt_sparse, t)(
            num_hashes=args.num_hashes,
            hashes_per_table=args.hashes_per_table,
            hash_range_pow=args.hash_range_pow,
            seed=args.seed,
        )
    elif t == "SparseSRPHasher":
        hasher = mlgt_sparse.SparseSRPHasher(
            num_hashes=args.num_hashes,
            num_bits=args.num_bits,
            seed=args.seed,
        )
    elif t == "DenseSRPHasher":
        hasher = mlgt_sparse.DenseSRPHasher(
            num_hashes=args.num_hashes,
            num_bits=args.num_bits,
            dimension=D,
            seed=args.seed,
        )
    elif t == "BloomHashFunction":
        hasher = mlgt_sparse.BloomHashFunction(
            num_hashes=args.num_hashes,
            num_bits=args.num_bits,
            dimension=D,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unknown hasher type: {t}")

    print("Hashing the dataset...")
    X_hashes = hash_matrix(hasher, X)
    print("Hashing the queries...")
    Q_hashes = hash_matrix(hasher, Q)

    print("Calculating true matches...")
    exact_scores: ndarray = (Q @ X.T).toarray()
    print("Calculating hash matches...")
    hash_scores: ndarray = get_num_matches(X_hashes, Q_hashes)

    avg_recall = recall_at_k(exact_scores, hash_scores, args.k)
    print(f"The average recall@{args.k} is {avg_recall:0.6f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Hasher recall@K sanity check.")
    parser.add_argument("--dataset", "-d", type=str, default="sparse1M",
                        choices=DATASETS, help="Dataset to use (default: sparse1M)")
    parser.add_argument("--hasher-type", "-t", type=str, choices=HASHER_TYPES,
                        default="MinHasher", help="Hasher to use (default: MinHasher)")
    parser.add_argument("--num-hashes", "-H", type=int, default=100,
                        help="Number of hash tables (default: 100)")
    parser.add_argument("--hashes-per-table", "-p", type=int, default=1,
                        help="Min-hashes combined per table, MinHash/WeightedMinHash only (default: 1)")
    parser.add_argument("--hash-range-pow", "-b", type=int, default=20,
                        help="Output range = 2^b, MinHash/WeightedMinHash only (default: 20)")
    parser.add_argument("--num-bits", "-B", type=int, default=16,
                        help="SRP projection bits per hash, SRP/Bloom only (default: 16)")
    parser.add_argument("--seed", "-S", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("-k", type=int, default=10,
                        help="K for Recall@K (default: 10)")

    run(parser.parse_args())

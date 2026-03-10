#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import List, Optional
import sys, os, time, gc
import argparse
import csv

import math, cmath, random, statistics
from tqdm import tqdm

import numpy as np
from numpy import array, ndarray, linalg, random as npr, matrix
from matplotlib import pyplot as plt

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)
REPO_DIR = os.path.abspath(os.path.join(CUR_DIR, '..'))
DATA_DIR = os.path.join(REPO_DIR, 'data')

DATASETS: List[str] = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]
ALGOS: List[str] = ["single", "single_threaded", "single_parallel", "double", "double_threaded"]

import gtnn
from gtnn import KNNSIndexDataset, read_matrix, save_matrix

# Plot time vs N and time vs K for all datasets, 4 algoithms in a figure, 4 dataset-figures in 1 plot
N_vals = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]
N_times = {dset: {algo: [] for algo in ALGOS} for dset in DATASETS}

# The second plot: similar
K_vals = [1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]
K_times = {dset: {algo: [] for algo in ALGOS} for dset in DATASETS}

Nq = 128

for dset in DATASETS:
    print(f"--- Processing dataset: {dset} ---")
    X = np.load(os.path.join(DATA_DIR, dset, "X.npy"))
    Q = np.load(os.path.join(DATA_DIR, dset, "Q.npy"))
    Qidxes = npr.choice(Q.shape[0], Nq, replace=False)
    Q = Q[Qidxes]
    for N in N_vals:
        print(f"  N={N}")
        knn_index = gtnn.KNNSIndexDataset(X[:N], k_val=10)
        for algo in ALGOS:
            print(f"    Algorithm: {algo}")
            t0 = time.time()
            if algo == "single":
                for q in Q:
                    knn_index.search(q, use_threading=False)
            elif algo == "single_threaded":
                for q in Q:
                    knn_index.search(q, use_threading=True)
            elif algo == "single_parallel":
                knn_index.search_batch_binary(Q, use_threading=True)
            elif algo == "double":
                knn_index.search_multiple(Q, use_threading=False)
            elif algo == "double_threaded":
                knn_index.search_multiple(Q, use_threading=True)
            N_times[dset][algo].append((time.time() - t0) * 1000 / Nq) # Convert to milliseconds

    for K in K_vals:
        print(f"  K={K}")
        for algo in ALGOS:
            print(f"    Algorithm: {algo}")
            knn_index = gtnn.KNNSIndexDataset(X, k_val=K)
            t0 = time.time()
            if algo == "single":
                for q in Q:
                    knn_index.search(q, use_threading=False)
            elif algo == "single_threaded":
                for q in Q:
                    knn_index.search(q, use_threading=True)
            elif algo == "single_parallel":
                knn_index.search_batch_binary(Q, use_threading=True)
            elif algo == "double":
                knn_index.search_multiple(Q, use_threading=False)
            elif algo == "double_threaded":
                knn_index.search_multiple(Q, use_threading=True)
            K_times[dset][algo].append((time.time() - t0) * 1000 / Nq) # Convert to milliseconds
    
    print()

# Plot to CSV file
CSV_FILE = os.path.join(CUR_DIR, "ablation_results.csv")
with open(CSV_FILE, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Dataset", "Algorithm", "N/K", "Value", "Time (ms)"])
    for dset in DATASETS:
        for algo in ALGOS:
            for N, time_ms in zip(N_vals, N_times[dset][algo]):
                writer.writerow([dset, algo, f"N={N}", "", time_ms])
            for K, time_ms in zip(K_vals, K_times[dset][algo]):
                writer.writerow([dset, algo, f"K={K}", "", time_ms])

# Run the plotting script
print("--- Results saved to ablation_results.csv ---")
print("--- Generating plots using plot_ablation.py ---")
import subprocess
subprocess.run(["python3", os.path.join(CUR_DIR, "plot_ablation.py")], check=True)
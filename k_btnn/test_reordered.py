#!/usr/bin/env python3
import gtnn
import numpy as np
import time
import os
import gc
import csv

def test_on_data(data, queries, k, name="Dataset"):
    print(f"\n{'='*20} Testing on {name} {'='*20}")
    print(f"Data shape: {data.shape}, Queries shape: {queries.shape}, k: {k}")

    # Normalize for cosine similarity
    print("Normalizing data...")
    data = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-9)
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9)

    results_list = []

    print("\nBuilding indices...")
    t0 = time.time()
    idx_std = gtnn.KNNSIndexDataset(data, k)
    std_build = time.time() - t0

    t0 = time.time()
    idx_reordered = gtnn.KNNReorderedIndexDataset(data, k)
    reordered_build = time.time() - t0

    print(f"Standard Build: {std_build:.4f}s")
    print(f"Reordered Build: {reordered_build:.4f}s")

    # Define the 5 modes clearly:
    # 1. Single Search (Serial): search(q, False) in a loop
    # 2. Single Search (Threaded): search(q, True) in a loop (Parallel across trees)
    # 3. Batch Binary (Threaded): search_batch_binary(qs, True) (Parallel across queries)
    # 4. Double GT (Serial): search_multiple(qs, False)
    # 5. Double GT (Threaded): search_multiple(qs, True)
    
    modes = [
        ("Single Search (Serial)", lambda idx, q: [idx.search(qv, False) for qv in q]),
        ("Single Search (Threaded)", lambda idx, q: [idx.search(qv, True) for qv in q]),
        ("Batch Binary (Threaded)", lambda idx, q: idx.search_batch_binary(q, True)),
        ("Double GT (Serial)", lambda idx, q: idx.search_multiple(q, False)),
        ("Double GT (Threaded)", lambda idx, q: idx.search_multiple(q, True)),
    ]

    for mode_name, search_func in modes:
        print(f"\nMode: {mode_name}")
        
        # Test Standard
        t0 = time.time()
        res_std_raw = search_func(idx_std, queries)
        std_time = time.time() - t0
        
        # Test Reordered
        t0 = time.time()
        res_reordered_raw = search_func(idx_reordered, queries)
        reordered_time = time.time() - t0

        # Extract results and dot product counts
        if mode_name.startswith("Single Search"):
            res_std = [r[0] for r in res_std_raw]
            dots_std = sum([r[1] for r in res_std_raw])
            res_reordered = [r[0] for r in res_reordered_raw]
            dots_reordered = sum([r[1] for r in res_reordered_raw])
        else:
            res_std, dots_std = res_std_raw
            res_reordered, dots_reordered = res_reordered_raw

        # Verify results match
        match_count = 0
        for i in range(len(queries)):
            s_set = set(res_std[i])
            r_set = set(res_reordered[i])
            if s_set == r_set:
                match_count += 1
            else:
                # Check dot products for ties
                d_std = sorted([np.dot(queries[i], data[j]) for j in res_std[i]], reverse=True)
                d_reordered = sorted([np.dot(queries[i], data[j]) for j in res_reordered[i]], reverse=True)
                if len(d_std) == len(d_reordered) and np.allclose(d_std, d_reordered):
                    match_count += 1
        
        match_rate = match_count / len(queries)
        print(f"  Standard  - Time: {std_time:.4f}s, Dots: {dots_std}")
        print(f"  Reordered - Time: {reordered_time:.4f}s, Dots: {dots_reordered}")
        print(f"  Match Rate: {match_rate:.4%}")

        results_list.append({
            "Dataset": name,
            "Mode": mode_name,
            "NumPoints": data.shape[0],
            "NumQueries": queries.shape[0],
            "StdBuildTime": std_build,
            "ReorderedBuildTime": reordered_build,
            "StdSearchTime": std_time,
            "ReorderedSearchTime": reordered_time,
            "StdDots": dots_std,
            "ReorderedDots": dots_reordered,
            "MatchRate": match_rate
        })
    
    del idx_std
    del idx_reordered
    gc.collect()
    return results_list

def main():
    print("Benchmark Configuration:")
    print("1. Single Search (Serial): search(q, False) in a loop")
    print("2. Single Search (Threaded): search(q, True) in a loop (Parallel across trees)")
    print("3. Batch Binary (Threaded): search_batch_binary(qs, True) (Parallel across queries)")
    print("4. Double GT (Serial): search_multiple(qs, False)")
    print("5. Double GT (Threaded): search_multiple(qs, True)\n")

    all_results = []
    datasets = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(CUR_DIR, '..', 'data')
    
    found_any = False
    for ds in datasets:
        ds_path = os.path.join(DATA_DIR, ds)
        if os.path.exists(ds_path):
            try:
                print(f"Loading dataset: {ds}")
                X = np.load(os.path.join(ds_path, "X.npy"))
                Q = np.load(os.path.join(ds_path, "Q.npy"))
                
                # Use 1000 queries or max available
                num_queries = min(1000, Q.shape[0])
                ds_results = test_on_data(X, Q[:num_queries], k=10, name=ds)
                all_results.extend(ds_results)
                
                del X
                del Q
                gc.collect()
                found_any = True
            except Exception as e:
                print(f"Error loading {ds}: {e}")

    if not found_any:
        print("No datasets found in ../data/. Running synthetic test.")
        N, D, Nq = 1000000, 1000, 1000
        data = np.random.rand(N, D).astype(np.float64)
        queries = np.random.rand(Nq, D).astype(np.float64)
        ds_results = test_on_data(data, queries, k=10, name="Synthetic")
        all_results.extend(ds_results)

    if all_results:
        csv_file = os.path.join(CUR_DIR, "reordered_benchmark_results.csv")
        print(f"\nSaving results to {csv_file}...")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    print("\nBenchmark Finished.")

if __name__ == "__main__":
    main()

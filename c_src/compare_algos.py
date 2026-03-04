import numpy as np
from numpy import array, ndarray, linalg, random as npr
import numba
import gtnn
import baselines_wrapper as baselines
import time
import argparse
import sys, os
import csv
import gc

DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]  # Expected datasets in data/

def generate_sparse_data(n, dim, density=0.1):
    data = np.random.rand(n, dim).astype(np.float64)
    mask = np.random.rand(n, dim) > density
    data[mask] = 0
    return data

@numba.njit
def compute_recall_scores(gt_scores: ndarray, res_scores: ndarray, tolerance: float = 1e-9) -> float:
    """
    Computes recall based on dot product values to handle ties in sorting.
    """
    recall_sum = 0
    num_queries = len(gt_scores)
    for i in range(num_queries):
        gts = np.sort(gt_scores[i])[::-1]
        res = np.sort(res_scores[i])[::-1]
        
        count = 0
        idx_g, idx_r = 0, 0
        while idx_g < len(gts) and idx_r < len(res):
            if np.abs(gts[idx_g] - res[idx_r]) < tolerance:
                count += 1
                idx_g += 1
                idx_r += 1
            elif gts[idx_g] > res[idx_r]:
                idx_g += 1
            else:
                idx_r += 1
        recall_sum += count / len(gts) if len(gts) > 0 else 1.0
    return recall_sum / num_queries if num_queries > 0 else 0.0

@numba.njit
def compute_precision_scores(gt_scores: ndarray, res_scores: ndarray, tolerance: float = 1e-9) -> float:
    """
    Computes precision based on dot product values to handle ties in sorting.
    """
    precision_sum = 0
    num_queries = len(gt_scores)
    for i in range(num_queries):
        gts = np.sort(gt_scores[i])[::-1]
        res = np.sort(res_scores[i])[::-1]
        
        count = 0
        idx_g, idx_r = 0, 0
        while idx_g < len(gts) and idx_r < len(res):
            if np.abs(gts[idx_g] - res[idx_r]) < tolerance:
                count += 1
                idx_g += 1
                idx_r += 1
            elif gts[idx_g] > res[idx_r]:
                idx_g += 1
            else:
                idx_r += 1
        precision_sum += count / len(res) if len(res) > 0 else 1.0
    return precision_sum / num_queries if num_queries > 0 else 0.0

def benchmark_algo(name, algo_class, data, queries, k, double_group=False, use_threading=True, **kwargs):
    print(f"\n--- Benchmarking {name} (Threading: {use_threading}) ---")
    t0 = time.time()
    try:
        if algo_class == gtnn.ThresholdIndexDataset:
            threshold = kwargs.pop("threshold", 0.5)
            idx = algo_class(data, threshold=threshold)
        else:
            idx = algo_class(data, k, **kwargs)
        build_time = time.time() - t0
        print(f"Build time: {build_time:.4f}s")
    except Exception as e:
        print(f"Build failed: {e}")
        return None, None, 0.0, 0.0

    t0 = time.time()
    try:
        is_our_algo = algo_class in [gtnn.KNNSIndexDataset, gtnn.ThresholdIndexDataset]
        if is_our_algo:
            if double_group:
                results, dots = idx.search_multiple(queries, use_threading=use_threading)
            else:
                results, dots = idx.search_batch_binary(queries, use_threading=use_threading)
        else:
            # For baselines, use original call without threading flag
            results = idx.search_multiple(queries)
            dots = 0 # Not tracked for baselines
            
        search_time = time.time() - t0
        res_indices = results[0] if isinstance(results, tuple) else results
        qps = len(queries) / search_time if search_time > 0 else 0
        print(f"Search time: {(search_time*1000):.5f}ms, QPS: {qps:.4f}, Dots: {dots}")
        return res_indices, idx, build_time, search_time
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 0.0, 0.0

def parse_list(s):
    if not s: return []
    if isinstance(s, list): return s
    return [int(x.strip()) for x in s.split(',')]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GTNN algorithms")
    parser.add_argument("--dataset", type=str, default="all", choices=DATASETS + ["all"], help="Name of dataset in data/ or 'all'.")
    parser.add_argument("--num_features", "-n", type=str, default="300000,1000,199000", help="Comma-separated points to add. Use -1 for all.")
    parser.add_argument("--num_queries", "-q", type=str, default="1000", help="Comma-separated queries per phase.")
    parser.add_argument("--dim", type=int, default=128, help="Dimension (only for synthetic)")
    parser.add_argument("--k", type=int, default=10, help="Top K")
    parser.add_argument("--double-group", action="store_true", help="Use double group testing for GTNN.")
    args = parser.parse_args()

    original_num_features = parse_list(args.num_features)
    if args.num_features == "-1": original_num_features = [-1]
    original_num_queries = parse_list(args.num_queries)
    
    algo_configs = [
        ("KNNS Binary (Threaded)", gtnn.KNNSIndexDataset, {"double_group": False, "use_threading": True}),
        ("KNNS Binary (Serial)", gtnn.KNNSIndexDataset, {"double_group": False, "use_threading": False}),
        ("KNNS Double (Threaded)", gtnn.KNNSIndexDataset, {"double_group": True, "use_threading": True}),
        ("KNNS Double (Serial)", gtnn.KNNSIndexDataset, {"double_group": True, "use_threading": False}),
        ("Linscan (Base)", baselines.LinscanWrapper, {}),
        ("Cufe (Base)", baselines.CufeWrapper, {}),
        ("SHNSW (Base)", baselines.SHNSWWrapper, {"ef_construction": 200}),
        ("Faiss HNSW", baselines.FaissHNSWWrapper, {"M": 32, "efConstruction": 128}),
        ("Faiss GT", baselines.FaissGTWrapper, {}),
        ("Scann", baselines.ScannWrapper, {}),
        ("Falconn", baselines.FalconnWrapper, {"num_tables": 50}),
        # ("NLE (Base)", baselines.NLEWrapper, {"t1": 10, "t2": 500}),
    ]

    results_records = []
    datasets_to_run = DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets_to_run:
        print(f"\n{'#'*40}\n### DATASET: {dataset}\n{'#'*40}")
        indexes = {}
        points_so_far = 0
        queries_so_far = 0
        num_features = list(original_num_features)
        num_queries = list(original_num_queries)

        CUR_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(CUR_DIR, '..', 'data')
        dataset_dir = os.path.join(DATA_DIR, dataset)
        
        try:
            full_data = np.load(os.path.join(dataset_dir, "X.npy"))
            full_queries = np.load(os.path.join(dataset_dir, "Q.npy"))
            if num_features[0] == -1: num_features = [full_data.shape[0]]
        except Exception as e:
            print(f"Load error for {dataset}: {e}. Skipping.")
            continue

        if len(num_queries) < len(num_features):
            num_queries.extend([num_queries[-1]] * (len(num_features) - len(num_queries)))
        num_queries = num_queries[:len(num_features)]

        for step_idx, (N, Nq) in enumerate(zip(num_features, num_queries)):
            print(f"\nPHASE {step_idx+1}: Add {N}, Query {Nq} (Dataset: {dataset})")
            if points_so_far == 0:
                current_data = full_data[:N]
                points_so_far = N
                action = "Build"
            else:
                new_data = full_data[points_so_far : points_so_far + N]
                points_so_far += N
                action = "Update"
            
            actual_nq = min(Nq, full_queries.shape[0] - queries_so_far)
            if actual_nq <= 0: break
            current_queries = full_queries[queries_so_far : queries_so_far + actual_nq]
            active_data = full_data[:points_so_far]
            
            print(f"Computing GT...")
            t0 = time.time()
            all_scores = np.dot(current_queries, active_data.T)
            gt_indices = []
            gt_scores_top_k = []
            for i in range(current_queries.shape[0]):
                top_k = np.argsort(all_scores[i])[-args.k:][::-1]
                gt_indices.append(top_k.tolist())
                gt_scores_top_k.append(all_scores[i][top_k])
            print(f"GT time: {time.time()-t0:.4f}s")

            for name, cls, kwargs in algo_configs:
                cfg_kwargs = kwargs.copy()
                dg = cfg_kwargs.pop("double_group", args.double_group)
                ut = cfg_kwargs.pop("use_threading", True)
                record = {
                    "Dataset": dataset, "Phase": step_idx + 1, "Algo": name,
                    "Points": points_so_far, "Queries": actual_nq, "Action": action,
                    "BuildTime": 0.0, "UpdateTime": 0.0, "SearchTime": 0.0, 
                    "Recall": 0.0, "Precision": 0.0, "QPS": 0.0,
                    "Threading": ut
                }

                if action == "Build":
                    res_indices, idx_obj, b_time, s_time = benchmark_algo(name, cls, current_data, current_queries, args.k, double_group=dg, use_threading=ut, **cfg_kwargs)
                    if idx_obj is not None and res_indices is not None:
                        indexes[name] = (idx_obj, dg, ut)
                        
                        # Calculate dot products for recall calculation
                        # active_data contains the dataset
                        res_scores = []
                        target_k = args.k
                        for i in range(len(current_queries)):
                            # Handle potential None or empty results
                            if res_indices[i] is None:
                                scores = np.array([])
                            else:
                                scores = np.dot(active_data[res_indices[i]], current_queries[i])
                            
                            # Pad with -inf if fewer results than k to ensure rectangular shape for Numba
                            if len(scores) < target_k:
                                padded = np.full(target_k, -np.inf, dtype=np.float32)
                                if len(scores) > 0:
                                    padded[:len(scores)] = scores
                                scores = padded
                            elif len(scores) > target_k:
                                scores = scores[:target_k]
                                
                            res_scores.append(scores)
                        
                        rec = compute_recall_scores(np.array(gt_scores_top_k), np.array(res_scores))
                        prec = compute_precision_scores(np.array(gt_scores_top_k), np.array(res_scores))
                        record.update({
                            "BuildTime": round(b_time, 4), 
                            "SearchTime": round(s_time * 1000, 5), 
                            "Recall": round(rec, 4), "Precision": round(prec, 4), 
                            "QPS": round(actual_nq/s_time, 4) if s_time>0 else 0
                        })
                else:
                    if name in indexes:
                        idx, dg, ut = indexes[name]
                        if hasattr(idx, 'streaming_update'):
                            t0 = time.time()
                            try:
                                idx.streaming_update(new_data)
                                record["UpdateTime"] = round(time.time() - t0, 4)
                                t0 = time.time()
                                is_our_algo = isinstance(idx, (gtnn.KNNSIndexDataset, gtnn.ThresholdIndexDataset))
                                if is_our_algo:
                                    if dg:
                                        results = idx.search_multiple(current_queries, use_threading=ut)
                                    else:
                                        results = idx.search_batch_binary(current_queries, use_threading=ut)
                                else:
                                    results = idx.search_multiple(current_queries)
                                
                                s_time = time.time() - t0
                                record["SearchTime"] = round(s_time * 1000, 5)
                                res_indices = results[0] if isinstance(results, tuple) else results
                                
                                # Calculate dot products for recall calculation in update case
                                res_scores = []
                                target_k = args.k
                                for i in range(len(current_queries)):
                                    # Handle potential None or empty results
                                    if res_indices[i] is None:
                                        scores = np.array([])
                                    else:
                                        scores = np.dot(active_data[res_indices[i]], current_queries[i])
                                    
                                    # Pad with -inf if fewer results than k
                                    if len(scores) < target_k:
                                        padded = np.full(target_k, -np.inf, dtype=np.float32)
                                        if len(scores) > 0:
                                            padded[:len(scores)] = scores
                                        scores = padded
                                    elif len(scores) > target_k:
                                        scores = scores[:target_k]
                                        
                                    res_scores.append(scores)
                                
                                record["Recall"] = round(compute_recall_scores(np.array(gt_scores_top_k), np.array(res_scores)), 4)
                                record["Precision"] = round(compute_precision_scores(np.array(gt_scores_top_k), np.array(res_scores)), 4)
                                record["QPS"] = round(actual_nq / s_time, 4) if s_time > 0 else 0
                                print(f"{name} Update: {record['UpdateTime']:.4f}s, QPS: {record['QPS']:.4f}, Recall: {record['Recall']:.4f}")
                            except Exception as e: 
                                print(f"{name} failed: {e}")
                                import traceback
                                traceback.print_exc()
                results_records.append(record)
            queries_so_far += actual_nq
            gc.collect()

    if results_records:
        csv_file = "benchmark_results.csv"
        print(f"\nSaving results to {csv_file}...")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_records[0].keys())
            writer.writeheader()
            writer.writerows(results_records)
    print("\nBenchmark Finished.")

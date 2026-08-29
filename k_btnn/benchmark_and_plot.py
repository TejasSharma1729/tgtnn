#!/usr/bin/env python3
"""
Unified Benchmarking + Plotting Tool

This script:
1. Runs comprehensive benchmarks on all algorithms (with optional streaming support)
2. Generates CSV results
3. Automatically creates comparison plots

Usage:
    python benchmark_and_plot.py --dataset imagenet --streaming
    python benchmark_and_plot.py --dataset all --no-streaming --plot
    python benchmark_and_plot.py --dataset imagenet --streaming --skip-plot
"""

import numpy as np
from numpy import array, ndarray, linalg, random as npr
import numba
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set thread count to 16 for all algorithms BEFORE importing other libraries
NUM_THREADS = 16
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(NUM_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(NUM_THREADS)

import gtnn
import baselines_wrapper as baselines
import time
import argparse
import csv
import gc
import psutil
import threading

CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]


class MemoryStats:
    """Container for memory statistics."""
    def __init__(self, start_mb, end_mb, peak_mb, min_mb, avg_mb, growth_mb, num_samples):
        self.start_mb = start_mb
        self.end_mb = end_mb
        self.peak_mb = peak_mb
        self.min_mb = min_mb
        self.avg_mb = avg_mb
        self.growth_mb = growth_mb
        self.num_samples = num_samples
    
    def __repr__(self):
        return (f"MemoryStats(start={self.start_mb:.2f}MB, end={self.end_mb:.2f}MB, "
                f"peak={self.peak_mb:.2f}MB, avg={self.avg_mb:.2f}MB, growth={self.growth_mb:.2f}MB)")
    
    def as_dict(self):
        """Return statistics as dictionary for CSV output."""
        return {
            "peak_mb": round(self.peak_mb, 2),
            "avg_mb": round(self.avg_mb, 2),
            "start_mb": round(self.start_mb, 2),
            "end_mb": round(self.end_mb, 2),
            "growth_mb": round(self.growth_mb, 2),
        }


class PeakMemoryMonitor:
    """
    Comprehensive memory monitor that tracks peak, average, start, end, and growth.
    Polls process RSS (all threads) at ~10ms intervals.
    """
    def __init__(self, interval=0.01):
        self._interval = interval
        self._stop = threading.Event()
        self._start_bytes = 0
        self._peak_bytes = 0
        self._min_bytes = float('inf')
        self._sum_bytes = 0
        self._samples = []
        self._thread = None
        self._count = 0

    def start(self):
        proc = psutil.Process(os.getpid())
        self._start_bytes = proc.memory_info().rss
        self._peak_bytes = self._start_bytes
        self._min_bytes = self._start_bytes
        self._samples = [self._start_bytes]
        self._stop.clear()
        
        def _run():
            p = psutil.Process(os.getpid())
            while not self._stop.wait(self._interval):
                try:
                    rss = p.memory_info().rss
                    self._samples.append(rss)
                    self._sum_bytes += rss
                    self._count += 1
                    if rss > self._peak_bytes:
                        self._peak_bytes = rss
                    if rss < self._min_bytes:
                        self._min_bytes = rss
                except psutil.Error:
                    pass
        
        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring and return comprehensive memory statistics."""
        self._stop.set()
        if self._thread:
            self._thread.join()
        
        # Get final memory
        try:
            end_bytes = psutil.Process(os.getpid()).memory_info().rss
        except psutil.Error:
            end_bytes = self._peak_bytes
        
        # Calculate statistics
        start_mb = self._start_bytes / (1024 * 1024)
        end_mb = end_bytes / (1024 * 1024)
        peak_mb = self._peak_bytes / (1024 * 1024)
        min_mb = self._min_bytes / (1024 * 1024)
        growth_mb = (self._peak_bytes - self._start_bytes) / (1024 * 1024)
        
        # Average (from samples collected during monitoring)
        if self._count > 0:
            avg_bytes = (self._sum_bytes + self._start_bytes + end_bytes) / (self._count + 2)
            avg_mb = avg_bytes / (1024 * 1024)
        else:
            avg_mb = peak_mb
        
        num_samples = len(self._samples) + self._count
        
        return MemoryStats(start_mb, end_mb, peak_mb, min_mb, avg_mb, growth_mb, num_samples)


def generate_sparse_data(n, dim, density=0.1):
    data = np.random.rand(n, dim).astype(np.float64)
    mask = np.random.rand(n, dim) > density
    data[mask] = 0
    return data


@numba.njit
def compute_recall_scores(gt_scores: ndarray, res_scores: ndarray, tolerance: float = 1e-9) -> float:
    """Computes recall based on dot product values to handle ties in sorting."""
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
    """Computes precision based on dot product values to handle ties in sorting."""
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
    """Run benchmark for a single algorithm."""
    monitor = PeakMemoryMonitor()
    monitor.start()
    t0 = time.time()
    query_args = {}
    if "budget" in kwargs:
        query_args["budget"] = kwargs.pop("budget")
    try:
        if algo_class == gtnn.ThresholdIndexDataset:
            threshold = kwargs.pop("threshold", 0.5)
            idx = algo_class(data, threshold=threshold)
        else:
            idx = algo_class(data, k, **kwargs)
        if query_args and hasattr(idx, "set_query_arguments"):
            idx.set_query_arguments(query_args)
        build_time = time.time() - t0
        print(f"Build time: {build_time:.4f}s")
    except Exception as e:
        print(f"Build failed: {e}")
        mem_stats = monitor.stop()
        gc.collect()
        return None, None, 0.0, 0.0, mem_stats

    t0 = time.time()
    try:
        is_our_algo = algo_class in [gtnn.KNNSIndexDataset, gtnn.KNNReorderedIndexDataset, gtnn.ThresholdIndexDataset]
        if is_our_algo:
            if double_group:
                results, dots = idx.search_multiple(queries, use_threading=use_threading)
            else:
                results, dots = idx.search_batch_binary(queries, use_threading=use_threading)
        else:
            results = idx.search_multiple(queries)
            dots = 0

        search_time = time.time() - t0
        res_indices = results[0] if isinstance(results, tuple) else results
        qps = len(queries) / search_time if search_time > 0 else 0
        mem_stats = monitor.stop()
        print(f"Search time: {(search_time*1000):.5f}ms, QPS: {qps:.4f}, Dots: {dots}")
        print(f"Memory: {mem_stats}")
        gc.collect()
        return res_indices, idx, build_time, search_time, mem_stats
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        mem_stats = monitor.stop()
        gc.collect()
        return None, None, 0.0, 0.0, mem_stats


def parse_list(s):
    """Parse comma-separated list of integers."""
    if not s: return []
    if isinstance(s, list): return s
    return [int(x.strip()) for x in s.split(',')]


def run_benchmarks(args):
    """Run comprehensive benchmarks."""
    original_num_features = parse_list(args.num_features)
    if args.num_features == "-1": original_num_features = [-1]
    original_num_queries = parse_list(args.num_queries)
    
    algo_configs = [
        ("KNNS Binary (Threaded)", gtnn.KNNSIndexDataset, {"double_group": False, "use_threading": True}),
        ("KNNS Binary (Serial)", gtnn.KNNSIndexDataset, {"double_group": False, "use_threading": False}),
        ("KNNS Double (Threaded)", gtnn.KNNSIndexDataset, {"double_group": True, "use_threading": True}),
        ("KNNS Double (Serial)", gtnn.KNNSIndexDataset, {"double_group": True, "use_threading": False}),
        ("KNNS Reordered Binary (Threaded)", gtnn.KNNReorderedIndexDataset, {"double_group": False, "use_threading": True}),
        ("KNNS Reordered Binary (Serial)", gtnn.KNNReorderedIndexDataset, {"double_group": False, "use_threading": False}),
        ("KNNS Reordered Double (Threaded)", gtnn.KNNReorderedIndexDataset, {"double_group": True, "use_threading": True}),
        ("KNNS Reordered Double (Serial)", gtnn.KNNReorderedIndexDataset, {"double_group": True, "use_threading": False}),
        ("Linscan (budget: 1)", baselines.LinscanWrapper, {"budget": 1}),
        ("Linscan (budget: 10)", baselines.LinscanWrapper, {"budget": 10}),
        ("Linscan (budget: 100)", baselines.LinscanWrapper, {"budget": 100}),
        ("Linscan (budget: 1000)", baselines.LinscanWrapper, {"budget": 1000}),
        ("Linscan (base)", baselines.LinscanWrapper, {}),
        ("Cufe (budget: 1)", baselines.CufeWrapper, {"budget": 1}),
        ("Cufe (budget: 10)", baselines.CufeWrapper, {"budget": 10}),
        ("Cufe (budget: 100)", baselines.CufeWrapper, {"budget": 100}),
        ("Cufe (budget: 1000)", baselines.CufeWrapper, {"budget": 1000}),
        ("Cufe (base)", baselines.CufeWrapper, {}),
        ("SHNSW (ef_search: 40)", baselines.SHNSWWrapper, {"M": 32, "ef_construction": 200, "ef_search": 40}),
        ("SHNSW (ef_search: 200)", baselines.SHNSWWrapper, {"M": 32, "ef_construction": 200, "ef_search": 200}),
        ("SHNSW (ef_search: 500)", baselines.SHNSWWrapper, {"M": 32, "ef_construction": 200, "ef_search": 500}),
        ("SHNSW (ef_search: 1000)", baselines.SHNSWWrapper, {"M": 32, "ef_construction": 200, "ef_search": 1000}),
        ("Faiss HNSW (efSearch: 64)", baselines.FaissHNSWWrapper, {"M": 32, "efConstruction": 128, "efSearch": 64}),
        ("Faiss HNSW (efSearch: 128)", baselines.FaissHNSWWrapper, {"M": 32, "efConstruction": 128, "efSearch": 128}),
        ("Faiss HNSW (efSearch: 256)", baselines.FaissHNSWWrapper, {"M": 32, "efConstruction": 128, "efSearch": 256}),
        ("Faiss GT", baselines.FaissGTWrapper, {}),
        ("Scann", baselines.ScannWrapper, {}),
        # ("Falconn (num_tables: 50)", baselines.FalconnWrapper, {"num_tables": 50}),
        # ("Falconn (num_tables: 100)", baselines.FalconnWrapper, {"num_tables": 100}),
    ]

    results_records = []
    datasets_to_run = DATASETS if args.dataset == "all" else [args.dataset]

    for dataset in datasets_to_run:
        print(f"\n{'#'*40}\n### DATASET: {dataset}\n{'#'*40}")
        ground_truths = []
        points_so_far = 0
        queries_so_far = 0
        num_features = list(original_num_features)
        num_queries = list(original_num_queries)

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

        # Determine phases based on streaming mode
        phases = [(num_features[0], num_queries[0], "Build")]
        if args.streaming and len(num_features) > 1:
            for i in range(1, len(num_features)):
                phases.append((num_features[i], num_queries[i], "Update"))

        for step_idx, (N, Nq, action_type) in enumerate(phases):
            phase_num = step_idx + 1
            print(f"\nPHASE {phase_num}: {action_type} with {N} points, {Nq} queries -- Ground Truth Computation")

            if action_type == "Build":
                points_so_far = N
            else:
                points_so_far += N

            actual_nq = min(Nq, full_queries.shape[0] - queries_so_far)
            if actual_nq <= 0: break
            current_queries = full_queries[queries_so_far : queries_so_far + actual_nq]
            active_data = full_data[:points_so_far]

            t0 = time.time()
            all_scores = np.dot(current_queries, active_data.T)
            gt_indices = []
            gt_scores_top_k = []
            for i in range(current_queries.shape[0]):
                top_k = np.argsort(all_scores[i])[-args.k:][::-1]
                gt_indices.append(top_k.tolist())
                gt_scores_top_k.append(all_scores[i][top_k])
            print(f"GT time: {time.time()-t0:.4f}s")

            ground_truths.append({"indices": gt_indices, "top_k_scores": gt_scores_top_k})
            del all_scores, gt_indices, gt_scores_top_k, phase_num
            gc.collect()

        for name, cls, kwargs in algo_configs:
            queries_so_far = 0
            points_so_far = 0
            search_index: dict[str, tuple] = { "name" : () }
            print(f"\n--- Benchmarking {name} (Threading: {kwargs.get('use_threading', True)}) ---")

            for step_idx, (N, Nq, action_type) in enumerate(phases):
                print(f"\nPHASE {step_idx + 1}: {action_type} with {N} points, {Nq} queries")
                if action_type == "Build":
                    current_data = full_data[:N]
                    points_so_far = N
                else:
                    new_data = full_data[points_so_far : points_so_far + N]
                    points_so_far += N
    
                actual_nq = min(Nq, full_queries.shape[0] - queries_so_far)
                assert actual_nq > 0, f"No queries left for phase {step_idx + 1}"
                current_queries = full_queries[queries_so_far : queries_so_far + actual_nq]
                active_data = full_data[:points_so_far]

                cfg_kwargs = kwargs.copy()
                dg = cfg_kwargs.pop("double_group", True)
                ut = cfg_kwargs.pop("use_threading", True)
                record = {
                    "Dataset": dataset, "Phase": step_idx + 1, "Algo": name,
                    "Points": points_so_far, "Queries": actual_nq, "Action": action_type,
                    "BuildTime": 0.0, "UpdateTime": 0.0, "SearchTime": 0.0, 
                    "Recall": 0.0, "Precision": 0.0, "QPS": 0.0,
                    "peak_mb": 0.0, "avg_mb": 0.0, "start_mb": 0.0, "end_mb": 0.0, "growth_mb": 0.0,
                    "Threading": ut
                }

                if action_type == "Build":
                    res_indices, idx_obj, b_time, s_time, peak_mem = benchmark_algo(
                        name, cls, current_data, current_queries, args.k, 
                        double_group=dg, use_threading=ut, **cfg_kwargs
                    )
                    if idx_obj is not None and res_indices is not None:
                        search_index[name] = (idx_obj, dg, ut)
                        
                        # Calculate recall/precision
                        res_scores = []
                        target_k = args.k
                        for i in range(len(current_queries)):
                            if res_indices[i] is None:
                                scores = np.array([])
                            else:
                                scores = np.dot(active_data[res_indices[i]], current_queries[i])
                            
                            if len(scores) < target_k:
                                padded = np.full(target_k, -np.inf, dtype=np.float64)
                                if len(scores) > 0:
                                    padded[:len(scores)] = scores
                                scores = padded
                            elif len(scores) > target_k:
                                scores = scores[:target_k]
                            res_scores.append(scores)
                        
                        rec = compute_recall_scores(np.array(ground_truths[step_idx]["top_k_scores"]), np.array(res_scores))
                        prec = compute_precision_scores(np.array(ground_truths[step_idx]["top_k_scores"]), np.array(res_scores))
                        mem_dict = peak_mem.as_dict() if peak_mem else {"peak_mb": 0.0, "avg_mb": 0.0, "start_mb": 0.0, "end_mb": 0.0, "growth_mb": 0.0}
                        record.update({
                            "BuildTime": round(b_time, 4), 
                            "SearchTime": round(s_time * 1000, 5), 
                            "Recall": round(rec, 4), "Precision": round(prec, 4), 
                            "QPS": round(actual_nq/s_time, 4) if s_time>0 else 0,
                            **mem_dict
                        })
                else:  # Update
                    if args.streaming:
                        idx, dg, ut = search_index[name]
                        if hasattr(idx, 'streaming_update'):
                            monitor = PeakMemoryMonitor()
                            monitor.start()
                            t0 = time.time()
                            try:
                                idx.streaming_update(new_data)
                                record["UpdateTime"] = round(time.time() - t0, 4)
                                t0 = time.time()
                                is_our_algo = isinstance(idx, (gtnn.KNNSIndexDataset, gtnn.KNNReorderedIndexDataset, gtnn.ThresholdIndexDataset))
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

                                res_scores = []
                                target_k = args.k
                                for i in range(len(current_queries)):
                                    if res_indices[i] is None:
                                        scores = np.array([])
                                    else:
                                        scores = np.dot(active_data[res_indices[i]], current_queries[i])

                                    if len(scores) < target_k:
                                        padded = np.full(target_k, -np.inf, dtype=np.float64)
                                        if len(scores) > 0:
                                            padded[:len(scores)] = scores
                                        scores = padded
                                    elif len(scores) > target_k:
                                        scores = scores[:target_k]

                                    res_scores.append(scores)

                                record["Recall"] = round(compute_recall_scores(np.array(ground_truths[step_idx]["top_k_scores"]), np.array(res_scores)), 4)
                                record["Precision"] = round(compute_precision_scores(np.array(ground_truths[step_idx]["top_k_scores"]), np.array(res_scores)), 4)
                                record["QPS"] = round(actual_nq / s_time, 4) if s_time > 0 else 0
                                print(f"{name} Update: {record['UpdateTime']:.4f}s, QPS: {record['QPS']:.4f}, Recall: {record['Recall']:.4f}")
                            except Exception as e:
                                print(f"{name} failed: {e}")
                                import traceback
                                traceback.print_exc()
                            finally:
                                mem_stats = monitor.stop()
                                mem_dict = mem_stats.as_dict() if mem_stats else {"peak_mb": 0.0, "avg_mb": 0.0, "start_mb": 0.0, "end_mb": 0.0, "growth_mb": 0.0}
                                gc.collect()
                                record.update(mem_dict)
                
                results_records.append(record)
                gc.collect()
            queries_so_far += actual_nq
            del search_index
            gc.collect()

    return results_records


def generate_plots(csv_path):
    """Generate comparison plots from CSV results."""
    print(f"\n{'#'*40}\n### GENERATING PLOTS\n{'#'*40}")
    
    try:
        df = pd.read_csv(csv_path)
        df = df[(df["Phase"] == 1) & (df["Action"] == "Build")]
        
        OUR_ALGOS = {
            "KNNS Binary (Threaded)": ("*",  200),
            "KNNS Binary (Serial)":   ("P",  120),
            "KNNS Double (Threaded)": ("D",  100),
            "KNNS Double (Serial)":   ("^",  100),
            "KNNS Reordered Binary (Threaded)": ("*", 200),
            "KNNS Reordered Binary (Serial)": ("P", 120),
            "KNNS Reordered Double (Threaded)": ("D", 100),
            "KNNS Reordered Double (Serial)": ("^", 100),
        }
        
        COMPETING_GROUPS = {
            "Linscan":   ["Linscan (budget: 1)", "Linscan (budget: 10)", "Linscan (budget: 100)",
                          "Linscan (budget: 1000)", "Linscan (base)"],
            "CuFe":      ["Cufe (budget: 1)", "Cufe (budget: 10)", "Cufe (budget: 100)",
                          "Cufe (budget: 1000)", "Cufe (base)"],
            "SHNSW":     ["SHNSW (ef_search: 40)", "SHNSW (ef_search: 200)",
                          "SHNSW (ef_search: 500)", "SHNSW (ef_search: 1000)"],
            "Faiss HNSW":["Faiss HNSW (efSearch: 64)", "Faiss HNSW (efSearch: 128)",
                          "Faiss HNSW (efSearch: 256)"],
            "Faiss GT":  ["Faiss GT"],
            "ScaNN":     ["Scann"],
            "Falconn":   ["Falconn (num_tables: 50)", "Falconn (num_tables: 100)"],
        }
        
        COMP_COLORS = {
            "Linscan":    "#e41a1c",
            "CuFe":       "#ff7f00",
            "SHNSW":      "#984ea3",
            "Faiss HNSW": "#4daf4a",
            "Faiss GT":   "#a65628",
            "ScaNN":      "#999999",
            "Falconn":    "#f781bf",
        }
        OUR_COLOR = "#1f78b4"
        
        DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]
        TITLES = {"imagenet": "ImageNet", "imdb_wiki": "IMDb-Wiki",
                  "insta_1m": "Instagram-1M", "mirflickr": "MIRFlickr"}
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for ax, dataset in zip(axes, DATASETS):
            sub = df[df["Dataset"] == dataset]
            
            # Plot competing algorithms
            for grp_name, algos in COMPETING_GROUPS.items():
                rows = sub[sub["Algo"].isin(algos)].sort_values("Recall")
                if rows.empty:
                    continue
                lw = 1.0 if len(rows) == 1 else 1.5
                mkr = "o"
                ax.plot(rows["Recall"], rows["QPS"],
                        marker=mkr, color=COMP_COLORS[grp_name], label=grp_name,
                        linewidth=lw, markersize=5, alpha=0.85)
            
            # Plot our algorithms (all at recall=1.0, stacked vertically)
            for algo_name, (mkr, sz) in OUR_ALGOS.items():
                row = sub[sub["Algo"] == algo_name]
                if row.empty:
                    continue
                ax.scatter(row["Recall"], row["QPS"],
                           color=OUR_COLOR, marker=mkr, s=sz, zorder=6,
                           edgecolors="black", linewidths=0.5,
                           label=algo_name)
            
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.1f}"))
            ax.set_xlim(-0.02, 1.08)
            ax.set_title(TITLES[dataset], fontsize=13, fontweight="bold")
            ax.set_xlabel("Recall@10", fontsize=11)
            ax.set_ylabel("QPS  (log scale)", fontsize=11)
            ax.grid(True, which="both", alpha=0.25, linestyle="--")
            ax.tick_params(labelsize=9)
        
        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        seen = set(labels)
        for ax in axes[1:]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    handles.append(h); labels.append(l); seen.add(l)
        
        fig.legend(handles, labels, loc="lower center", ncol=6,
                   bbox_to_anchor=(0.5, -0.03), fontsize=9,
                   frameon=True, framealpha=0.9)
        
        fig.suptitle("QPS vs Recall@10  —  300 K datapoints  (Phase 1 Build)",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout(rect=(0, 0.07, 1, 1))
        
        plot_path = csv_path.replace(".csv", ".png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {plot_path}")
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified Benchmark + Plot Tool: Run benchmarks and generate comparison plots")
    parser.add_argument("--dataset", type=str, default="all", choices=DATASETS + ["all"],
                        help="Dataset to benchmark")
    parser.add_argument("--num_features", "-n", type=str, default="300000,1000,199000",
                        help="Comma-separated feature counts per phase")
    parser.add_argument("--num_queries", "-q", type=str, default="1000",
                        help="Comma-separated query counts per phase")
    parser.add_argument("--dim", type=int, default=128,
                        help="Dimension for synthetic data")
    parser.add_argument("--k", type=int, default=10,
                        help="Top K neighbors")
    parser.add_argument("--double-group", action="store_true",
                        help="Use double group testing for GTNN")
    parser.add_argument("--streaming", action="store_true",
                        help="Enable streaming/update phases (default: build only)")
    parser.add_argument("--skip-plot", action="store_true",
                        help="Skip plot generation after benchmarks")
    parser.add_argument("--output-csv", type=str, default="benchmark_results.csv",
                        help="Output CSV filename")
    
    args = parser.parse_args()
    
    # Run benchmarks
    print("Starting benchmarks...")
    results = run_benchmarks(args)
    
    # Save CSV
    if results:
        csv_file = os.path.join(CUR_DIR, args.output_csv)
        print(f"\nSaving results to {csv_file}...")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved → {csv_file}")
        
        # Generate plots unless skipped
        if not args.skip_plot:
            generate_plots(csv_file)
    
    print("\n" + "="*40)
    print("Benchmark Complete!")
    print("="*40)

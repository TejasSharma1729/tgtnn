#!/usr/bin/env python3
"""
Ablation benchmark: recall vs. dot-product cost for varying L and Lq.

Plot 1 — L ablation:  fix Lq=5, sweep L ∈ {6,7,8,9,10,11,12}
Plot 2 — Lq ablation: fix L=10, sweep Lq ∈ {2,3,4,5,6,7,8}

Results are saved to ablation_L_results.csv.
"""
import sys, os, time, gc, csv

# Set thread count to 16 for all algorithms BEFORE importing other libraries
NUM_THREADS = 16
os.environ['OMP_NUM_THREADS'] = str(NUM_THREADS)
os.environ['OPENBLAS_NUM_THREADS'] = str(NUM_THREADS)
os.environ['MKL_NUM_THREADS'] = str(NUM_THREADS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(NUM_THREADS)
os.environ['NUMEXPR_NUM_THREADS'] = str(NUM_THREADS)

import numpy as np

CUR_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
DATA_DIR = os.path.join(REPO_DIR, "data")

sys.path.insert(0, CUR_DIR)
import gtnn_ablation as ga

# ── configuration ───────────────────────────────────────────────────────────

K = 10           # number of nearest neighbours to retrieve
NUM_QUERIES = 1000
NUM_RUNS = 1     # runs per algorithm config; results are averaged

L_ABLATION_LQ  = 5
L_ABLATION_Ls  = [6, 7, 8, 9, 10, 11, 12]

LQ_ABLATION_L  = 10
LQ_ABLATION_Lqs = [2, 3, 4, 5, 6, 7, 8]

DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]

# class name → (L, Lq) for reporting
L_ABLATION_CLASSES = [
    (L, L_ABLATION_LQ,
     f"KNNReordered_L{L}_Lq{L_ABLATION_LQ}")
    for L in L_ABLATION_Ls
]
LQ_ABLATION_CLASSES = [
    (LQ_ABLATION_L, Lq,
     f"KNNReordered_L{LQ_ABLATION_L}_Lq{Lq}")
    for Lq in LQ_ABLATION_Lqs
]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_dataset(name):
    path = os.path.join(DATA_DIR, name)
    X = np.load(os.path.join(path, "X.npy"))
    Q = np.load(os.path.join(path, "Q.npy"))
    return X, Q

def normalize(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / norms

def compute_ground_truth(data, queries, k):
    """Precompute the k-th best dot-product threshold for each query.

    Using index-set intersection is wrong when ties exist (np.argsort picks an
    arbitrary subset of tied indices).  IMDB-Wiki has many near-duplicate face
    embeddings where this is pervasive.  We store the kth dot-product value so
    recall can be checked without rerunning brute-force for every algorithm config.
    """
    thresholds = np.empty(len(queries))
    for i, q in enumerate(queries):
        dots = data @ q
        thresholds[i] = float(np.partition(dots, -k)[-k])
    return thresholds

def recall_from_results(data, queries, results, k, gt_thresholds):
    """Mean recall@k given pre-computed kth-dot thresholds."""
    recalls = []
    for i, q in enumerate(queries):
        found = results[i] if i < len(results) else []
        if found:
            # Only compute dot products for the k returned items.
            # This differs from the full-matrix path used in compute_ground_truth,
            # but is safe whenever items are strictly above the tie threshold (recall=1.0).
            item_scores = data[np.array(found, dtype=np.intp)] @ q
            match_count = int(np.sum(item_scores >= gt_thresholds[i] - 1e-9))
        else:
            match_count = 0
        recalls.append(min(1.0, match_count / k))
    return float(np.mean(recalls))

def run_config(data, queries, cls_name, k, gt_thresholds, single_query=False, num_runs=NUM_RUNS):
    """
    Build index for cls_name and run search num_runs times, returning averaged metrics.
    single_query=False → search_multiple  (K-dgtnn: double-grouping, queries reordered)
    single_query=True  → search_batch_binary (K-btnn*: per-query, data tree only)
    Returns (mean_recall, mean_dots_per_query, mean_build_time_s, mean_query_time_s).
    """
    cls = getattr(ga, cls_name)
    recalls, dpqs, build_times, query_times = [], [], [], []

    for _ in range(num_runs):
        t_build0 = time.perf_counter()
        idx = cls(data, k)
        build_times.append(time.perf_counter() - t_build0)

        t_query0 = time.perf_counter()
        if single_query:
            results, total_dots = idx.search_batch_binary(queries, use_threading=True)
        else:
            results, total_dots = idx.search_multiple(queries, use_threading=True)
        query_times.append(time.perf_counter() - t_query0)

        recalls.append(recall_from_results(data, queries, results, k, gt_thresholds))
        dpqs.append(total_dots / len(queries))
        del idx
        gc.collect()

    return (float(np.mean(recalls)), float(np.mean(dpqs)),
            float(np.mean(build_times)), float(np.mean(query_times)) / len(queries))

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    rows = []

    found_any = False
    for ds_name in DATASETS:
        ds_path = os.path.join(DATA_DIR, ds_name)
        if not os.path.isdir(ds_path):
            continue
        try:
            print(f"\n{'='*60}")
            print(f"Dataset: {ds_name}")
            X, Q = load_dataset(ds_name)
            X = normalize(X.astype(np.float64))
            Q = normalize(Q[:NUM_QUERIES].astype(np.float64))
            print(f"  N={X.shape[0]}, D={X.shape[1]}, Nq={Q.shape[0]}")
            found_any = True
        except Exception as e:
            print(f"  Skipping {ds_name}: {e}")
            continue

        # ── precompute ground-truth thresholds once per dataset ──
        print(f"\n  Computing ground-truth thresholds (cached for all configs)...")
        gt_thresholds = compute_ground_truth(X, Q, K)

        # ── L ablation (K-dgtnn) ──
        print(f"\n  [K-dgtnn / L ablation] Lq={L_ABLATION_LQ}  ({NUM_RUNS} runs each)")
        for L, Lq, cls_name in L_ABLATION_CLASSES:
            try:
                recall, dpq, build_t, mqt = run_config(X, Q, cls_name, K, gt_thresholds)
                print(f"    L={L:2d} Lq={Lq}: recall={recall:.4f}  dots/q={dpq:.1f}"
                      f"  build={build_t:.2f}s  mean_query={mqt*1e3:.2f}ms")
                rows.append(dict(algorithm="dgtnn", dataset=ds_name, ablation="L", L=L, Lq=Lq,
                                 recall=recall, dots_per_query=dpq,
                                 build_time_s=build_t, mean_query_time_ms=mqt*1e3))
                gc.collect()
            except Exception as e:
                print(f"    L={L} Lq={Lq}: ERROR {e}")

        # ── Lq ablation (K-dgtnn) ──
        print(f"\n  [K-dgtnn / Lq ablation] L={LQ_ABLATION_L}  ({NUM_RUNS} runs each)")
        for L, Lq, cls_name in LQ_ABLATION_CLASSES:
            try:
                recall, dpq, build_t, mqt = run_config(X, Q, cls_name, K, gt_thresholds)
                print(f"    L={L} Lq={Lq:2d}: recall={recall:.4f}  dots/q={dpq:.1f}"
                      f"  build={build_t:.2f}s  mean_query={mqt*1e3:.2f}ms")
                rows.append(dict(algorithm="dgtnn", dataset=ds_name, ablation="Lq", L=L, Lq=Lq,
                                 recall=recall, dots_per_query=dpq,
                                 build_time_s=build_t, mean_query_time_ms=mqt*1e3))
                gc.collect()
            except Exception as e:
                print(f"    L={L} Lq={Lq}: ERROR {e}")

        # ── L ablation (K-btnn*) ──
        print(f"\n  [K-btnn* / L ablation] Lq={L_ABLATION_LQ}  ({NUM_RUNS} runs each)")
        for L, Lq, cls_name in L_ABLATION_CLASSES:
            try:
                recall, dpq, build_t, mqt = run_config(X, Q, cls_name, K, gt_thresholds, single_query=True)
                print(f"    L={L:2d} Lq={Lq}: recall={recall:.4f}  dots/q={dpq:.1f}"
                      f"  build={build_t:.2f}s  mean_query={mqt*1e3:.2f}ms")
                rows.append(dict(algorithm="btnn", dataset=ds_name, ablation="L", L=L, Lq=Lq,
                                 recall=recall, dots_per_query=dpq,
                                 build_time_s=build_t, mean_query_time_ms=mqt*1e3))
                gc.collect()
            except Exception as e:
                print(f"    L={L} Lq={Lq}: ERROR {e}")

        del X, Q
        gc.collect()

    if not found_any:
        print("No real datasets found — running synthetic fallback (N=50000, D=128, Nq=1000).")
        N, D = 50_000, 128
        X = normalize(np.random.rand(N, D))
        Q = normalize(np.random.rand(NUM_QUERIES, D))
        gt_thresholds = compute_ground_truth(X, Q, K)
        # dgtnn: L and Lq ablation; btnn: L ablation only (no query tree → Lq irrelevant)
        sweep = [
            ("dgtnn", False, "L",  L_ABLATION_CLASSES),
            ("dgtnn", False, "Lq", LQ_ABLATION_CLASSES),
            ("btnn",  True,  "L",  L_ABLATION_CLASSES),
        ]
        for algo, single_query, ablation_tag, configs in sweep:
            for L, Lq, cls_name in configs:
                try:
                    recall, dpq, build_t, mqt = run_config(X, Q, cls_name, K, gt_thresholds, single_query=single_query)
                    print(f"  [{algo}/{ablation_tag}-abl] L={L:2d} Lq={Lq:2d}: recall={recall:.4f}"
                          f"  dots/q={dpq:.1f}  build={build_t:.2f}s  mean_query={mqt*1e3:.2f}ms")
                    rows.append(dict(algorithm=algo, dataset="synthetic", ablation=ablation_tag,
                                     L=L, Lq=Lq, recall=recall, dots_per_query=dpq,
                                     build_time_s=build_t, mean_query_time_ms=mqt*1e3))
                except Exception as e:
                    print(f"  [{algo}/{ablation_tag}-abl] L={L} Lq={Lq}: ERROR {e}")

    csv_path = os.path.join(CUR_DIR, "ablation_L_results.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to {csv_path}")

    print("\nRunning plot script...")
    import subprocess
    subprocess.run(["python3", os.path.join(CUR_DIR, "plot_ablation_L.py")], check=True)

if __name__ == "__main__":
    main()

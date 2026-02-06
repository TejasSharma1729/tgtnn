import sys
import time
import sparse_gtnn

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python benchmark_optimized.py <data_file> <query_file>")
        sys.exit(1)
    data_file = sys.argv[1]
    query_file = sys.argv[2]

    print("Loading data...")
    data, data_dim = sparse_gtnn.read_sparse_matrix(data_file)
    queries, query_dim = sparse_gtnn.read_sparse_matrix(query_file)
    print(f"Data: {len(data)} vectors, {data_dim} dimensions")
    print(f"Queries: {len(queries)} vectors, {query_dim} dimensions")

    print("\n=== Testing ThresholdLinscanOptimizedFast ===")
    linscan = sparse_gtnn.ThresholdLinscanOptimizedFast(0.8)
    start = time.time()
    linscan.fit(data)
    stop = time.time()
    fit_time = (stop - start) * 1000
    print(f"Fit time: {fit_time:.2f} ms")

    # Warmup
    linscan.search(queries[:min(10, len(queries))])

    # Benchmark
    start = time.time()
    results = linscan.search(queries)
    stop = time.time()
    search_time = (stop - start) * 1000
    avg_time_per_query = search_time / len(queries)
    qps = 1000.0 / avg_time_per_query
    print(f"Total search time: {search_time:.2f} ms")
    print(f"Average time per query: {avg_time_per_query:.2f} ms")
    print(f"QPS: {qps:.2f}")
    total_results = sum(len(r) for r in results)
    print(f"Total results found: {total_results}")

    print("\n=== Testing ThresholdBatchProcessor ===")
    start = time.time()
    batch_results = sparse_gtnn.ThresholdBatchProcessor.processBatch(queries, data, 0.8)
    stop = time.time()
    batch_time = (stop - start) * 1000
    batch_avg_time = batch_time / len(queries)
    batch_qps = 1000.0 / batch_avg_time
    print(f"Total batch time: {batch_time:.2f} ms")
    print(f"Average time per query: {batch_avg_time:.2f} ms")
    print(f"QPS: {batch_qps:.2f}")

    print("\n=== Testing Single-threaded ===")
    start = time.time()
    single_results = linscan.search(queries, 1)
    stop = time.time()
    single_time = (stop - start) * 1000
    single_avg_time = single_time / len(queries)
    single_qps = 1000.0 / single_avg_time
    print(f"Single-threaded time: {single_time:.2f} ms")
    print(f"Average time per query: {single_avg_time:.2f} ms")
    print(f"QPS: {single_qps:.2f}")

if __name__ == "__main__":
    main()

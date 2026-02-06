import sys
import sparse_gtnn

def run_naive_search(dataset_name, dataset_path, queries, threshold):
    print("\n" + "="*50)
    print(f"TESTING {dataset_name}")
    print("="*50)
    print(f"Loading dataset from: {dataset_path}")
    try:
        import time
        start_load = time.time()
        dataset, dataset_dim = sparse_gtnn.read_sparse_matrix(dataset_path)
        stop_load = time.time()
        load_duration = (stop_load - start_load) * 1000
        print(f"Dataset loaded: {len(dataset)} vectors, dimension: {dataset_dim}")
        print(f"Dataset load time: {load_duration:.2f} ms")
        if dataset_name == "1M Dataset" and len(dataset) != 1000000:
            print(f"Warning: Expected 1M vectors, got {len(dataset)}")
        if dataset_dim != 30109:
            print(f"Warning: Expected dimension 30109, got {dataset_dim}")
        print(f"\nStarting naive search...")
        print(f"Threshold: {threshold}")
        start_search = time.time()
        results, total_dot_products = sparse_gtnn.naive_search(dataset, queries, threshold)
        stop_search = time.time()
        total_search_time_ms = (stop_search - start_search) * 1000
        avg_search_time_per_query_ms = total_search_time_ms / len(queries)
        total_results = sum(len(r) for r in results)
        print(f"\n=== {dataset_name} NAIVE SEARCH RESULTS ===")
        print(f"Total search time: {total_search_time_ms:.2f} ms")
        print(f"Average time per query: {avg_search_time_per_query_ms:.2f} ms")
        print(f"Total dot products computed: {total_dot_products}")
        print(f"Expected dot products: {len(dataset) * len(queries)}")
        print(f"Total results found: {total_results}")
        print(f"Average results per query: {total_results / len(queries):.2f}")
        queries_per_second = len(queries) * 1000.0 / total_search_time_ms
        dot_products_per_second = total_dot_products * 1000.0 / total_search_time_ms
        print(f"Throughput: {queries_per_second:.2f} queries/second")
        print(f"Dot product rate: {dot_products_per_second / 1e6:.2f} M dot products/second")
    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

def main():
    dataset_1m_path = "/home/tejassharma/big-ann-benchmarks/data/sparse/base_1M.csr"
    dataset_full_path = "/home/tejassharma/big-ann-benchmarks/data/sparse/base_full.csr"
    query_path = "/home/tejassharma/big-ann-benchmarks/data/sparse/queries.dev.csr"
    if len(sys.argv) >= 2:
        dataset_1m_path = sys.argv[1]
    if len(sys.argv) >= 3:
        dataset_full_path = sys.argv[2]
    if len(sys.argv) >= 4:
        query_path = sys.argv[3]
    print(f"Loading queries from: {query_path}")
    try:
        import time
        start_load = time.time()
        queries, query_dim = sparse_gtnn.read_sparse_matrix(query_path)
        stop_load = time.time()
        load_duration = (stop_load - start_load) * 1000
        print(f"Queries loaded: {len(queries)} vectors, dimension: {query_dim}")
        print(f"Queries load time: {load_duration:.2f} ms")
        if len(queries) != 6980:
            print(f"Warning: Expected 6980 query vectors, got {len(queries)}")
        if query_dim != 30109:
            print(f"Warning: Expected query dimension 30109, got {query_dim}")
        threshold = 0.5
        run_naive_search("1M Dataset", dataset_1m_path, queries, threshold)
        run_naive_search("FULL Dataset", dataset_full_path, queries, threshold)
        print("\n" + "="*50)
        print("ALL TESTS COMPLETED")
        print("="*50)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

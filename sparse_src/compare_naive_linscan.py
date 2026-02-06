import sys
import sparse_gtnn

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python compare_naive_linscan.py <data_file> <query_file>")
        sys.exit(1)
    data_file = sys.argv[1]
    query_file = sys.argv[2]

    print(f"Loading data from {data_file}...")
    data, data_dim = sparse_gtnn.read_sparse_matrix(data_file)
    print(f"Loading queries from {query_file}...")
    queries, query_dim = sparse_gtnn.read_sparse_matrix(query_file)

    print(f"Using full dataset: {len(data)} vectors")
    if len(queries) > 100:
        queries = queries[:100]
        print("Limited queries to 100 vectors for testing")
    else:
        print(f"Using all {len(queries)} queries")

    sparse_gtnn.analyze_threshold_distribution(data, queries, 1000)

    print("\n=== Single Query Performance Test ===")
    single_test = sparse_gtnn.NaiveSearch(0.6)
    import time
    single_start = time.time()
    single_result = single_test.search([queries[0]], data)
    single_stop = time.time()
    single_duration = (single_stop - single_start) * 1000
    print(f"Single query time: {single_duration:.3f} ms")
    print(f"Single query results: {len(single_result[0])} matches")

    print("\n=== Testing with threshold 0.6 ===")
    sparse_gtnn.compare_approaches(data, queries, 0.6)

if __name__ == "__main__":
    main()

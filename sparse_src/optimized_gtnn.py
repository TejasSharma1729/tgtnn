import sys
import sparse_gtnn

def main():
    if len(sys.argv) < 3:
        print("Usage: python optimized_gtnn.py <base_file> <query_file>")
        sys.exit(1)
    base_file = sys.argv[1]
    query_file = sys.argv[2]

    gt = sparse_gtnn.GroupTestingNN(base_file, 10)
    query_set, _ = sparse_gtnn.read_sparse_matrix(query_file)
    mean_search_time = 0.0
    net_num_dots = 0
    net_search_time = 0.0
    net_naive_time = 0.0
    mean_recall = 0.0
    result_set, net_num_dots = gt.search(query_set)
    net_search_time = gt.last_search_time_ms
    print(f"Mean search time: {net_search_time / len(query_set)} ms")
    print(f"Net number of dot products: {net_num_dots}")
    for i, query in enumerate(query_set):
        time_and_recall = gt.verify_results(query, result_set[i])
        net_naive_time += time_and_recall[0]
        mean_recall += time_and_recall[1]
        print(f"Query {i}: naive time: {time_and_recall[0]} ms, recall: {time_and_recall[1]}")
    print(f"Mean search time: {net_search_time / len(query_set)} ms")
    print(f"Net number of dot products: {net_num_dots}")
    print(f"Mean naive time: {net_naive_time / len(query_set)} ms")
    print(f"Mean recall: {mean_recall / len(query_set)}")
    print(f"Naive: number of dot products: {gt.size() * len(query_set)}")

if __name__ == "__main__":
    main()

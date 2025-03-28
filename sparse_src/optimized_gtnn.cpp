#include "GTnn/sparse_optimized.hpp"

int main(int argc, char ** argv) {
    GTnn::GroupTestingNN<6> gt(argv[1], 0.8);
    auto [query_set, _] = GTnn::read_sparse_matrix(argv[2]);
    size_t net_num_dots;
    double mean_search_time = 0.0;
    double mean_naive_time = 0.0;
    double mean_precision = 0.0;
    double mean_recall = 0.0;
    for (size_t i = 0; i < query_set.size(); i++) {
        auto start = high_resolution_clock::now();
        auto [result, num_dots] = gt.search(query_set[i]);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        mean_search_time += duration.count() / 1.0e+3;
        net_num_dots += num_dots;
        array<double, 3> results = gt.verify_results(query_set[i], result);
        mean_naive_time += results[0];
        mean_precision += results[1];
        mean_recall += results[2];
    }
    cout << "Mean search time: " << mean_search_time / query_set.size() << " ms" << endl;
    cout << "Mean naive time: " << mean_naive_time / query_set.size() << " ms" << endl;
    cout << "Mean precision: " << mean_precision / query_set.size() << endl;
    cout << "Mean recall: " << mean_recall / query_set.size() << endl;
    cout << "Net number of dot products: " << net_num_dots << endl;
    cout << "Naive: number of dot products: " << gt.data_set.size() * query_set.size() << endl;
    return 0;
}
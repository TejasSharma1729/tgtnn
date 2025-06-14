#include "GTnn/knns_double.hpp"

// int main1(int argc, char ** argv) {
//     GTnn::GroupTestingNN gt(argv[1], 10);
//     auto [query_set, _] = GTnn::read_sparse_matrix(argv[2]);
//     double mean_search_time = 0.0;
//     size_t net_num_dots = 0;
//     double mean_naive_time = 0.0;
//     double mean_recall = 0.0;

//     for (size_t i = 0; i < query_set.rows(); i++) {
//         auto start = high_resolution_clock::now();
//         auto [result, num_dots] = gt.search(query_set[i]);
//         auto stop = high_resolution_clock::now();
//         double search_time = duration_cast<microseconds>(stop - start).count() / 1.0e+3;
//         mean_search_time += search_time;
//         net_num_dots += num_dots;
//         array<double, 2> time_and_recall = gt.verify_results(query_set[i], result);
//         mean_naive_time += time_and_recall[0];
//         mean_recall += time_and_recall[1];
//         cout << "Query " << i << ": time: " << search_time << "ms, naive time: "
//          << time_and_recall[0] << " ms, recall: " << time_and_recall[1] << endl;
//     }
    
//     cout << "Mean search time: " << mean_search_time / query_set.rows() << " ms" << endl;
//     cout << "Net number of dot products: " << net_num_dots << endl;
//     cout << "Mean naive time: " << mean_naive_time / query_set.rows() << " ms" << endl;
//     cout << "Mean recall: " << mean_recall / query_set.rows() << endl;
//     cout << "Naive: number of dot products: " << gt.rows() * query_set.rows() << endl;
//     return 0;
// }

int main2(int argc, char ** argv) {
    GTnn::GroupTestingNN gt(GTnn::path_append(argv[1], "X.txt"), 10);
    GTnn::matrix_t query_set;
    if (!GTnn::extract_matrix(GTnn::path_append(argv[1], "Q.txt"), query_set)) {
        cerr << "Error extracting query matrix." << endl;
        return 1;
    }
    double mean_search_time = 0.0;
    size_t net_num_dots = 0;
    double net_search_time = 0.0;
    double net_naive_time = 0.0;
    double mean_recall = 0.0;
    map<uint, vector<uint>> result_set;

    for (int i = 0; i < 10; i++) {
        auto start = high_resolution_clock::now();
        tie(result_set, net_num_dots) = gt.search(query_set);
        auto stop = high_resolution_clock::now();
        net_search_time += duration_cast<microseconds>(stop - start).count() / (1.0e+4);
    }
    
    cout << "Mean search time: " << net_search_time / query_set.rows() << " ms" << endl;
    cout << "Net number of dot products: " << net_num_dots << endl;

    // for (long i = 0; i < query_set.rows(); i++) {
    //     array<double, 2> time_and_recall = gt.verify_results(query_set.row(i), result_set[i]);
    //     net_naive_time += time_and_recall[0];
    //     mean_recall += time_and_recall[1];
    //     cout << "Query " << i << ": naive time: " << time_and_recall[0] << 
    //     " ms, recall: " << time_and_recall[1] << endl;
    // }
    
    // cout << "Mean search time: " << net_search_time / query_set.rows() << " ms" << endl;
    // cout << "Net number of dot products: " << net_num_dots << endl;
    // cout << "Mean naive time: " << net_naive_time / query_set.rows() << " ms" << endl;
    // cout << "Mean recall: " << mean_recall / query_set.rows() << endl;
    // cout << "Naive: number of dot products: " << gt.size() * query_set.rows() << endl;
    return 0;
}

int main(int argc, char ** argv) {
    return main2(argc, argv);
}

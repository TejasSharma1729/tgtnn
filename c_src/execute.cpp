#include "GTnn/dim_based.hpp"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <path> <dataset> [threshold]" << endl;
        return 1;
    }
    string path = argv[1];
    string dataset = argv[2];
    double threshold = (argc > 3) ? atof(argv[3]) : 0.8;

    GTnn::dim_based_gtnn_t gtnn(path, dataset, threshold);
    GTnn::matrix_t queries;
    GTnn::extract_matrix(GTnn::path_append(path, "Q.txt"), queries);

    for (size_t i = 0; i < queries.rows(); i++) {
        GTnn::vector_t query = queries.row(i);
        vector<uint> result;
        size_t num_dot_products;
        high_resolution_clock::time_point start = high_resolution_clock::now();
        auto result_and_num_dots = gtnn.search(query);
        high_resolution_clock::time_point end = high_resolution_clock::now();
        double elapsed_time = duration_cast<microseconds>(end - start).count() / 1.0e+3;
        result = result_and_num_dots.first;
        num_dot_products = result_and_num_dots.second;

        array<double, 3> verification_result = gtnn.exhaustive_search(query, result);
        cout << "Query " << i << ": " << endl;
        cout << "Result size: " << result.size() << endl;
        cout << "Number of dot products: " << num_dot_products << endl;
        cout << "Search time: " << elapsed_time << " ms" << endl;
        cout << "Exhaustive time: " << verification_result[0] << " ms" << endl;
        cout << "Precision: " << verification_result[1] << endl;
        cout << "Recall: " << verification_result[2] << endl;
        usleep(1000000); // Sleep for 1 second
    }

    return 0;
}
#include "GTnn/optimized_double.hpp"

int main(int argc, char ** argv) {
    GTnn::OptimizedDoubleNN gt(GTnn::path_append(argv[1], "X.txt"), 0.8);
    GTnn::matrix_t query_set;
    if (!GTnn::extract_matrix(GTnn::path_append(argv[1], "Q.txt"), query_set)) {
        cerr << "Error extracting query matrix." << endl;
        return 1;
    }

    vector<double> search_times;
    vector<size_t> dot_products;
    
    cout << "Running optimized_double batch search 10 times..." << endl;
    
    for (int run = 0; run < 10; run++) {
        auto start = high_resolution_clock::now();
        auto [result_set, num_dots] = gt.search(query_set);
        auto stop = high_resolution_clock::now();
        
        double total_search_time = duration_cast<microseconds>(stop - start).count() / 1.0e+3;
        double avg_search_time = total_search_time / query_set.rows();
        
        search_times.push_back(avg_search_time);
        dot_products.push_back(num_dots);
        
        cout << "Run " << (run + 1) << ": Avg search time per query: " << avg_search_time 
             << " ms, Total dot products: " << num_dots << endl;
    }
    
    // Calculate mean and standard deviation
    double mean_time = 0.0;
    for (double time : search_times) {
        mean_time += time;
    }
    mean_time /= search_times.size();
    
    double variance_time = 0.0;
    for (double time : search_times) {
        variance_time += (time - mean_time) * (time - mean_time);
    }
    variance_time /= search_times.size();
    double std_dev_time = sqrt(variance_time);
    
    double mean_dots = 0.0;
    for (size_t dots : dot_products) {
        mean_dots += dots;
    }
    mean_dots /= dot_products.size();
    
    double variance_dots = 0.0;
    for (size_t dots : dot_products) {
        variance_dots += (dots - mean_dots) * (dots - mean_dots);
    }
    variance_dots /= dot_products.size();
    double std_dev_dots = sqrt(variance_dots);
    
    cout << "\n=== OPTIMIZED DOUBLE RESULTS ===" << endl;
    cout << "Mean search time per query: " << mean_time << " ± " << std_dev_time << " ms" << endl;
    cout << "Mean dot products per run: " << mean_dots << " ± " << std_dev_dots << endl;
    cout << "Number of queries: " << query_set.rows() << endl;
    
    return 0;
}

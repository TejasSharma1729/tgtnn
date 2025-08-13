#include "GTnn/optimized_linscan.hpp"
#include <chrono>
#include <iostream>

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <data_file> <query_file>" << endl;
        return 1;
    }
    
    // Load data
    cout << "Loading data..." << endl;
    auto [data, data_dim] = GTnn::read_sparse_matrix(argv[1]);
    auto [queries, query_dim] = GTnn::read_sparse_matrix(argv[2]);
    
    cout << "Data: " << data.size() << " vectors, " << data_dim << " dimensions" << endl;
    cout << "Queries: " << queries.size() << " vectors, " << query_dim << " dimensions" << endl;
    
    // Method 1: Optimized LINSCAN approach
    cout << "\n=== Testing OptimizedLinscan ===" << endl;
    GTnn::OptimizedLinscan linscan(0.8);
    
    auto start = high_resolution_clock::now();
    linscan.fit(data);
    auto stop = high_resolution_clock::now();
    auto fit_time = duration_cast<milliseconds>(stop - start);
    cout << "Fit time: " << fit_time.count() << " ms" << endl;
    
    // Warmup
    linscan.search(vector<GTnn::sparse_vec_t>(queries.begin(), queries.begin() + min(10UL, queries.size())));
    
    // Benchmark
    start = high_resolution_clock::now();
    auto results = linscan.search(queries);
    stop = high_resolution_clock::now();
    auto search_time = duration_cast<microseconds>(stop - start);
    
    double avg_time_per_query = static_cast<double>(search_time.count()) / queries.size() / 1000.0;
    double qps = 1000.0 / avg_time_per_query;
    
    cout << "Total search time: " << search_time.count() / 1000.0 << " ms" << endl;
    cout << "Average time per query: " << avg_time_per_query << " ms" << endl;
    cout << "QPS: " << qps << endl;
    
    size_t total_results = 0;
    for (const auto& result : results) {
        total_results += result.size();
    }
    cout << "Total results found: " << total_results << endl;
    
    // Method 2: Batch processing approach
    cout << "\n=== Testing BatchProcessor ===" << endl;
    
    start = high_resolution_clock::now();
    auto batch_results = GTnn::BatchProcessor::processBatch(queries, data, 0.8);
    stop = high_resolution_clock::now();
    auto batch_time = duration_cast<microseconds>(stop - start);
    
    double batch_avg_time = static_cast<double>(batch_time.count()) / queries.size() / 1000.0;
    double batch_qps = 1000.0 / batch_avg_time;
    
    cout << "Total batch time: " << batch_time.count() / 1000.0 << " ms" << endl;
    cout << "Average time per query: " << batch_avg_time << " ms" << endl;
    cout << "QPS: " << batch_qps << endl;
    
    // Single-threaded comparison
    cout << "\n=== Testing Single-threaded ===" << endl;
    
    start = high_resolution_clock::now();
    auto single_results = linscan.search(queries, 1);
    stop = high_resolution_clock::now();
    auto single_time = duration_cast<microseconds>(stop - start);
    
    double single_avg_time = static_cast<double>(single_time.count()) / queries.size() / 1000.0;
    double single_qps = 1000.0 / single_avg_time;
    
    cout << "Single-threaded time: " << single_time.count() / 1000.0 << " ms" << endl;
    cout << "Average time per query: " << single_avg_time << " ms" << endl;
    cout << "QPS: " << single_qps << endl;
    
    return 0;
}

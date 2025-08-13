#include "GTnn/naive_vs_linscan.hpp"
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <data_file> <query_file>" << endl;
        return 1;
    }
    
    try {
        // Load data
        cout << "Loading data from " << argv[1] << "..." << endl;
        auto [data, data_dim] = GTnn::read_sparse_matrix(argv[1]);
        
        cout << "Loading queries from " << argv[2] << "..." << endl;
        auto [queries, query_dim] = GTnn::read_sparse_matrix(argv[2]);
        
        // Use full dataset but limit queries to 100 for reasonable testing time
        cout << "Using full dataset: " << data.size() << " vectors" << endl;
        
        if (queries.size() > 100) {
            queries.resize(100);
            cout << "Limited queries to 100 vectors for testing" << endl;
        } else {
            cout << "Using all " << queries.size() << " queries" << endl;
        }
        
        // Analyze threshold distribution first
        GTnn::analyze_threshold_distribution(data, queries, 1000);
        
        // Test single query performance to verify manual calculation
        cout << "\n=== Single Query Performance Test ===" << endl;
        GTnn::NaiveSearch single_test(0.6);
        
        auto single_start = chrono::high_resolution_clock::now();
        auto single_result = single_test.search({queries[0]}, data);  // Just first query
        auto single_stop = chrono::high_resolution_clock::now();
        auto single_duration = chrono::duration_cast<chrono::microseconds>(single_stop - single_start);
        
        cout << "Single query time: " << single_duration.count() / 1000.0 << " ms" << endl;
        cout << "Single query results: " << single_result[0].size() << " matches" << endl;
        
        // Run comparison with threshold 0.6
        cout << "\n=== Testing with threshold 0.6 ===" << endl;
        GTnn::compare_approaches(data, queries, 0.6);
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}

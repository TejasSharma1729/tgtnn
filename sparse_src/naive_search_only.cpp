#include "GTnn/header.hpp"
#include <thread>
#include <mutex>
#include <atomic>

using namespace std;
using namespace std::chrono;

// Global variables for thread synchronization
mutex result_mutex;
atomic<size_t> total_dot_products(0);

// Function to perform naive search for a subset of queries
void naive_search_thread(const GTnn::sparse_mat_t& dataset, 
                        const GTnn::sparse_mat_t& queries, 
                        size_t start_query, 
                        size_t end_query,
                        double threshold,
                        vector<vector<size_t>>& results) {
    
    size_t local_dot_products = 0;
    
    for (size_t q = start_query; q < end_query; q++) {
        vector<size_t> query_results;
        
        for (size_t d = 0; d < dataset.size(); d++) {
            local_dot_products++;
            double similarity = GTnn::dot_product(
                const_cast<GTnn::sparse_vec_t&>(queries[q]), 
                const_cast<GTnn::sparse_vec_t&>(dataset[d])
            );
            
            if (similarity >= threshold) {
                query_results.push_back(d);
            }
        }
        
        // Store results safely
        lock_guard<mutex> lock(result_mutex);
        results[q] = move(query_results);
    }
    
    total_dot_products += local_dot_products;
}

// Function to run naive search on a dataset and report results
void run_naive_search(const string& dataset_name, 
                     const string& dataset_path, 
                     const GTnn::sparse_mat_t& queries, 
                     double threshold) {
    
    cout << "\n" << string(50, '=') << endl;
    cout << "TESTING " << dataset_name << endl;
    cout << string(50, '=') << endl;
    cout << "Loading dataset from: " << dataset_path << endl;
    
    try {
        // Load dataset
        auto start_load = high_resolution_clock::now();
        auto [dataset, dataset_dim] = GTnn::read_sparse_matrix(dataset_path);
        auto stop_load = high_resolution_clock::now();
        auto load_duration = duration_cast<milliseconds>(stop_load - start_load);
        
        cout << "Dataset loaded: " << dataset.size() << " vectors, dimension: " << dataset_dim << endl;
        cout << "Dataset load time: " << load_duration.count() << " ms" << endl;
        
        // Verify expected sizes for known datasets
        if (dataset_name == "1M Dataset") {
            if (dataset.size() != 1000000) {
                cerr << "Warning: Expected 1M vectors, got " << dataset.size() << endl;
            }
        }
        if (dataset_dim != 30109) {
            cerr << "Warning: Expected dimension 30109, got " << dataset_dim << endl;
        }
        
        // Prepare results storage
        vector<vector<size_t>> results(queries.size());
        
        cout << "\nStarting naive search with " << NUM_THREADS << " threads..." << endl;
        cout << "Threshold: " << threshold << endl;
        
        // Reset counters
        total_dot_products = 0;
        
        // Start timing
        auto search_start = high_resolution_clock::now();
        
        // Create threads
        vector<thread> threads;
        size_t queries_per_thread = queries.size() / NUM_THREADS;
        size_t remaining_queries = queries.size() % NUM_THREADS;
        
        for (int t = 0; t < NUM_THREADS; t++) {
            size_t start_query = t * queries_per_thread;
            size_t end_query = start_query + queries_per_thread;
            
            // Last thread takes remaining queries
            if (t == NUM_THREADS - 1) {
                end_query += remaining_queries;
            }
            
            threads.emplace_back(naive_search_thread, 
                               ref(dataset), 
                               ref(queries), 
                               start_query, 
                               end_query, 
                               threshold, 
                               ref(results));
        }
        
        // Wait for all threads to complete
        for (auto& t : threads) {
            t.join();
        }
        
        auto search_stop = high_resolution_clock::now();
        auto search_duration = duration_cast<microseconds>(search_stop - search_start);
        
        // Calculate statistics
        double total_search_time_ms = search_duration.count() / 1000.0;
        double avg_search_time_per_query_ms = total_search_time_ms / queries.size();
        
        size_t total_results = 0;
        for (const auto& query_results : results) {
            total_results += query_results.size();
        }
        
        // Report results
        cout << "\n=== " << dataset_name << " NAIVE SEARCH RESULTS ===" << endl;
        cout << "Total search time: " << total_search_time_ms << " ms" << endl;
        cout << "Average time per query: " << avg_search_time_per_query_ms << " ms" << endl;
        cout << "Total dot products computed: " << total_dot_products.load() << endl;
        cout << "Expected dot products: " << dataset.size() * queries.size() << endl;
        cout << "Total results found: " << total_results << endl;
        cout << "Average results per query: " << (double)total_results / queries.size() << endl;
        
        // Throughput calculations
        double queries_per_second = queries.size() * 1000.0 / total_search_time_ms;
        double dot_products_per_second = total_dot_products.load() * 1000.0 / total_search_time_ms;
        
        cout << "Throughput: " << queries_per_second << " queries/second" << endl;
        cout << "Dot product rate: " << dot_products_per_second / 1e6 << " M dot products/second" << endl;
        
    } catch (const exception& e) {
        cerr << "Error processing " << dataset_name << ": " << e.what() << endl;
    }
}

int main(int argc, char** argv) {
    // Default file paths
    string dataset_1m_path = "/home/tejassharma/big-ann-benchmarks/data/sparse/base_1M.csr";
    string dataset_full_path = "/home/tejassharma/big-ann-benchmarks/data/sparse/base_full.csr";
    string query_path = "/home/tejassharma/big-ann-benchmarks/data/sparse/queries.dev.csr";
    
    // Allow command line override
    if (argc >= 2) {
        dataset_1m_path = argv[1];
    }
    if (argc >= 3) {
        dataset_full_path = argv[2];
    }
    if (argc >= 4) {
        query_path = argv[3];
    }
    
    cout << "Loading queries from: " << query_path << endl;
    
    try {
        // Load queries first (shared between both datasets)
        auto start_load = high_resolution_clock::now();
        auto [queries, query_dim] = GTnn::read_sparse_matrix(query_path);
        auto stop_load = high_resolution_clock::now();
        auto load_duration = duration_cast<milliseconds>(stop_load - start_load);
        
        cout << "Queries loaded: " << queries.size() << " vectors, dimension: " << query_dim << endl;
        cout << "Queries load time: " << load_duration.count() << " ms" << endl;
        
        // Verify query set size
        if (queries.size() != 6980) {
            cerr << "Warning: Expected 6980 query vectors, got " << queries.size() << endl;
        }
        if (query_dim != 30109) {
            cerr << "Warning: Expected query dimension 30109, got " << query_dim << endl;
        }
        
        // Use threshold of 0.5 (commonly used threshold)
        double threshold = 0.5;
        
        // Test 1M dataset first
        run_naive_search("1M Dataset", dataset_1m_path, queries, threshold);
        
        // Test full dataset second
        run_naive_search("FULL Dataset", dataset_full_path, queries, threshold);
        
        cout << "\n" << string(50, '=') << endl;
        cout << "ALL TESTS COMPLETED" << endl;
        cout << string(50, '=') << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

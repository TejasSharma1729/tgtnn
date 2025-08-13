#include "GTnn/header.hpp"
#include <thread>
#include <mutex>
#include <iomanip>

#define K_VAL 10
#define THRESHOLD 0.8
#define NUM_THREADS 16

// Thread-safe structure to collect results
struct ThreadResult {
    long query_id;
    vector<uint> above_threshold;
    array<uint, K_VAL> top_k;
};

// Function to process a range of queries
void process_queries(
    const GTnn::matrix_t& query_set,
    const GTnn::matrix_t& data_set,
    long start_query,
    long end_query,
    vector<ThreadResult>& results,
    mutex& results_mutex
) {
    vector<ThreadResult> local_results;
    
    for (long i = start_query; i < end_query; i++) {
        GTnn::vector_t query = query_set.row(i);
        
        // Compute all dot products manually (naive search)
        vector<pair<double, uint>> scores;
        vector<uint> above_tr;
        
        for (Eigen::Index j = 0; j < data_set.rows(); j++) {
            double score = query.dot(data_set.row(j));
            scores.push_back(make_pair(score, j));
            
            if (score >= THRESHOLD) {
                above_tr.push_back(j);
            }
        }
        
        // Sort by score (descending) and take top k
        sort(scores.begin(), scores.end(), greater<pair<double, uint>>());
        
        array<uint, K_VAL> top_k;
        for (size_t idx = 0; idx < K_VAL; idx++) {
            top_k[idx] = scores[idx].second;
        }
        sort(top_k.begin(), top_k.end());
        
        ThreadResult result;
        result.query_id = i;
        result.above_threshold = above_tr;
        result.top_k = top_k;
        local_results.push_back(result);
    }
    
    // Add local results to global results with mutex protection
    lock_guard<mutex> lock(results_mutex);
    results.insert(results.end(), local_results.begin(), local_results.end());
}

// Run multithreaded naive search
map<long, pair<vector<uint>, array<uint, K_VAL>>> run_multithreaded_naive_search(
    GTnn::matrix_t& query_set,
    GTnn::matrix_t& data_set,
    double& search_time
) {
    map<long, pair<vector<uint>, array<uint, K_VAL>>> ground_truth;
    vector<ThreadResult> all_results;
    mutex results_mutex;
    
    cout << "Running multithreaded naive search with " << NUM_THREADS << " threads..." << endl;
    auto start = high_resolution_clock::now();
    
    // Calculate work distribution
    long total_queries = query_set.rows();
    long queries_per_thread = total_queries / NUM_THREADS;
    long remaining_queries = total_queries % NUM_THREADS;
    
    vector<thread> threads;
    threads.reserve(NUM_THREADS);
    
    long current_start = 0;
    
    // Create and launch threads
    for (int t = 0; t < NUM_THREADS; t++) {
        long current_end = current_start + queries_per_thread;
        
        // Distribute remaining queries among first threads
        if (t < remaining_queries) {
            current_end++;
        }
        
        if (current_start < total_queries) {
            threads.emplace_back(
                process_queries,
                ref(query_set),
                ref(data_set),
                current_start,
                current_end,
                ref(all_results),
                ref(results_mutex)
            );
            
            cout << "Thread " << t << " processing queries " << current_start 
                 << " to " << (current_end - 1) << endl;
        }
        
        current_start = current_end;
    }
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    auto stop = high_resolution_clock::now();
    search_time = duration_cast<microseconds>(stop - start).count() / 1.0e+3;
    
    // Organize results by query ID
    for (const auto& result : all_results) {
        ground_truth[result.query_id] = {result.above_threshold, result.top_k};
    }
    
    double avg_time_per_query = search_time / query_set.rows();
    cout << "Multithreaded naive search completed." << endl;
    cout << "Total time: " << search_time << " ms" << endl;
    cout << "Average time per query: " << avg_time_per_query << " ms" << endl;
    cout << "Processed " << all_results.size() << " queries" << endl;
    
    return ground_truth;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dataset_path>" << endl;
        return 1;
    }
    
    // Load query set
    GTnn::matrix_t query_set;
    if (!GTnn::extract_matrix(GTnn::path_append(argv[1], "Q.txt"), query_set)) {
        cerr << "Error extracting query matrix." << endl;
        return 1;
    }
    
    // Load data set
    GTnn::matrix_t data_set;
    if (!GTnn::extract_matrix(GTnn::path_append(argv[1], "X.txt"), data_set)) {
        cerr << "Error extracting data matrix." << endl;
        return 1;
    }
    
    cout << "=== MULTITHREADED NAIVE SEARCH ===" << endl;
    cout << "Dataset: " << argv[1] << endl;
    cout << "Query matrix dimensions: " << query_set.rows() << " x " << query_set.cols() << endl;
    cout << "Data matrix dimensions: " << data_set.rows() << " x " << data_set.cols() << endl;
    cout << "Number of threads: " << NUM_THREADS << endl;
    cout << "K value: " << K_VAL << endl;
    cout << "Threshold: " << THRESHOLD << endl << endl;
    
    // Run multithreaded naive search
    double search_time = 0.0;
    auto ground_truth = run_multithreaded_naive_search(query_set, data_set, search_time);
    
    // Display some statistics
    cout << endl << "=== STATISTICS ===" << endl;
    
    // Count total results above threshold and average
    long total_above_threshold = 0;
    long total_queries_processed = 0;
    
    for (const auto& [query_id, results] : ground_truth) {
        total_above_threshold += results.first.size();
        total_queries_processed++;
    }
    
    double avg_results_per_query = static_cast<double>(total_above_threshold) / total_queries_processed;
    
    cout << "Total queries processed: " << total_queries_processed << endl;
    cout << "Total results above threshold: " << total_above_threshold << endl;
    cout << "Average results per query above threshold: " << fixed << setprecision(2) 
         << avg_results_per_query << endl;
    
    // Show sample of first few results
    cout << endl << "=== SAMPLE RESULTS (First 5 queries) ===" << endl;
    for (int i = 0; i < min(5L, static_cast<long>(ground_truth.size())); i++) {
        if (ground_truth.count(i)) {
            const auto& [above_threshold, top_k] = ground_truth[i];
            cout << "Query " << i << ": " << above_threshold.size() 
                 << " results above threshold, Top-K: ";
            for (size_t j = 0; j < K_VAL; j++) {
                cout << top_k[j];
                if (j < K_VAL - 1) cout << ", ";
            }
            cout << endl;
        }
    }
    
    cout << endl << "=== NAIVE SEARCH COMPLETE ===" << endl;
    return 0;
}

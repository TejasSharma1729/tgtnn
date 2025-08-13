/**
 * Comparison: Naive Search vs LINSCAN
 * 
 * This file implements both approaches for direct performance comparison
 * and analysis of their differences.
 */

#include "header.hpp"
#include <unordered_map>
#include <chrono>
#include <thread>
#include <future>
#include <random>
#include <algorithm>

namespace GTnn {
    
    /**
     * APPROACH 1: Traditional Naive Search (Your current implementation)
     * 
     * Algorithm:
     * - For each query, iterate through ALL data vectors
     * - Compute dot product between query and each data vector
     * - If dot product >= threshold, add to results
     * 
     * Time Complexity: O(Q * N * min(d_q, d_n))
     * Space Complexity: O(results)
     */
    class NaiveSearch {
    public:
        NaiveSearch(double threshold = 0.8);
        vector<vector<uint>> search(const sparse_mat_t& queries, const sparse_mat_t& data);
        
    private:
        double threshold_;
        double dot_product_naive(const sparse_vec_t& q, const sparse_vec_t& d);
    };
    
    /**
     * APPROACH 2: LINSCAN-style Implementation
     * 
     * Algorithm:
     * - Build inverted index: term_id -> list of (doc_id, value) pairs
     * - For each query:
     *   1. Initialize candidate scores (hash map)
     *   2. For each query term, accumulate scores for documents containing that term
     *   3. Filter candidates by threshold
     * 
     * Time Complexity: 
     * - Index build: O(N * d_avg)
     * - Query: O(Q * d_q * avg_posting_length)
     * Space Complexity: O(N * d_avg) for index
     */
    class LinscanStyle {
    public:
        LinscanStyle(double threshold = 0.8);
        void build_index(const sparse_mat_t& data);
        vector<vector<uint>> search(const sparse_mat_t& queries);
        vector<vector<uint>> search_parallel(const sparse_mat_t& queries, int num_threads = 8);
        
    private:
        double threshold_;
        unordered_map<uint, vector<pair<uint, float>>> inverted_index_;
        size_t data_size_;
        vector<uint> search_single_query(const sparse_vec_t& query);
    };
    
    /**
     * APPROACH 3: Optimized LINSCAN with better memory management
     */
    class OptimizedLinscan {
    public:
        OptimizedLinscan(double threshold = 0.8);
        void build_index(const sparse_mat_t& data);
        vector<vector<uint>> search(const sparse_mat_t& queries);
        
    private:
        double threshold_;
        unordered_map<uint, vector<pair<uint, float>>> inverted_index_;
        size_t data_size_;
        
        // Reused arrays to avoid allocations
        vector<double> candidate_scores_;
        vector<bool> candidate_flags_;
        vector<uint> active_candidates_;
        
        vector<uint> search_single_query_optimized(const sparse_vec_t& query);
    };
    
    /**
     * Comparison and analysis functions
     */
    void compare_approaches(const sparse_mat_t& data, const sparse_mat_t& queries, double threshold = 0.8);
    bool verify_results_match(const vector<vector<uint>>& results1, const vector<vector<uint>>& results2);
    void print_result_stats(const string& name, const vector<vector<uint>>& results);
    void analyze_threshold_distribution(const sparse_mat_t& data, const sparse_mat_t& queries, int num_samples = 1000);
}

// Implementation of all classes and functions

GTnn::NaiveSearch::NaiveSearch(double threshold) : threshold_(threshold) {}

vector<vector<uint>> GTnn::NaiveSearch::search(const sparse_mat_t& queries, const sparse_mat_t& data) {
    vector<vector<uint>> results(queries.size());
    
    auto start = chrono::high_resolution_clock::now();
    
    // Time individual queries to verify performance
    vector<double> query_times;
    query_times.reserve(queries.size());
    
    for (size_t q = 0; q < queries.size(); q++) {
        auto query_start = chrono::high_resolution_clock::now();
        
        for (size_t d = 0; d < data.size(); d++) {
            double score = dot_product_naive(queries[q], data[d]);
            if (score >= threshold_) {
                results[q].push_back(d);
            }
        }
        
        auto query_stop = chrono::high_resolution_clock::now();
        auto query_duration = chrono::duration_cast<chrono::microseconds>(query_stop - query_start);
        query_times.push_back(query_duration.count() / 1000.0); // Convert to ms
    }
    
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    
    cout << "Naive Search - Total time: " << duration.count() / 1000.0 << " ms" << endl;
    cout << "Naive Search - Avg time per query: " << 
            (duration.count() / 1000.0) / queries.size() << " ms" << endl;
    
    // Show first few individual query times for verification
    cout << "Individual query times (first 5): ";
    for (size_t i = 0; i < min(size_t(5), query_times.size()); i++) {
        cout << query_times[i] << "ms ";
    }
    cout << endl;
    
    // Show min/max query times
    auto [min_time, max_time] = minmax_element(query_times.begin(), query_times.end());
    cout << "Min query time: " << *min_time << "ms, Max query time: " << *max_time << "ms" << endl;
    
    return results;
}

double GTnn::NaiveSearch::dot_product_naive(const sparse_vec_t& q, const sparse_vec_t& d) {
    double result = 0.0;
    size_t i = 0, j = 0;
    
    while (i < q.size() && j < d.size()) {
        if (q[i].index == d[j].index) {
            result += q[i].value * d[j].value;
            i++; j++;
        } else if (q[i].index < d[j].index) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}

GTnn::LinscanStyle::LinscanStyle(double threshold) : threshold_(threshold) {}

void GTnn::LinscanStyle::build_index(const sparse_mat_t& data) {
    cout << "Building inverted index..." << endl;
    auto start = chrono::high_resolution_clock::now();
    
    inverted_index_.clear();
    data_size_ = data.size();
    
    for (size_t doc_id = 0; doc_id < data.size(); doc_id++) {
        for (const auto& elem : data[doc_id]) {
            inverted_index_[elem.index].emplace_back(doc_id, elem.value);
        }
    }
    
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    
    cout << "Index build time: " << duration.count() << " ms" << endl;
    cout << "Index size: " << inverted_index_.size() << " terms" << endl;
    
    // Statistics
    size_t total_postings = 0;
    size_t max_posting_length = 0;
    for (const auto& [term, postings] : inverted_index_) {
        total_postings += postings.size();
        max_posting_length = max(max_posting_length, postings.size());
    }
    cout << "Average posting length: " << (double)total_postings / inverted_index_.size() << endl;
    cout << "Max posting length: " << max_posting_length << endl;
}

vector<vector<uint>> GTnn::LinscanStyle::search(const sparse_mat_t& queries) {
    vector<vector<uint>> results(queries.size());
    
    auto start = chrono::high_resolution_clock::now();
    
    for (size_t q = 0; q < queries.size(); q++) {
        results[q] = search_single_query(queries[q]);
    }
    
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    
    cout << "LINSCAN Search - Total time: " << duration.count() / 1000.0 << " ms" << endl;
    cout << "LINSCAN Search - Avg time per query: " << 
            (duration.count() / 1000.0) / queries.size() << " ms" << endl;
    
    return results;
}

vector<vector<uint>> GTnn::LinscanStyle::search_parallel(const sparse_mat_t& queries, int num_threads) {
    vector<vector<uint>> results(queries.size());
    
    auto start = chrono::high_resolution_clock::now();
    
    vector<future<void>> futures;
    atomic<size_t> query_idx(0);
    
    for (int t = 0; t < num_threads; t++) {
        futures.push_back(async(launch::async, [&]() {
            size_t idx;
            while ((idx = query_idx.fetch_add(1)) < queries.size()) {
                results[idx] = search_single_query(queries[idx]);
            }
        }));
    }
    
    for (auto& f : futures) {
        f.wait();
    }
    
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    
    cout << "LINSCAN Parallel Search - Total time: " << duration.count() / 1000.0 << " ms" << endl;
    cout << "LINSCAN Parallel Search - Avg time per query: " << 
            (duration.count() / 1000.0) / queries.size() << " ms" << endl;
    
    return results;
}

vector<uint> GTnn::LinscanStyle::search_single_query(const sparse_vec_t& query) {
    // Use unordered_map for candidate accumulation
    unordered_map<uint, double> candidates;
    
    // Accumulate scores for all candidates
    for (const auto& q_elem : query) {
        uint term_id = q_elem.index;
        float q_value = q_elem.value;
        
        auto it = inverted_index_.find(term_id);
        if (it != inverted_index_.end()) {
            for (const auto& [doc_id, doc_value] : it->second) {
                candidates[doc_id] += q_value * doc_value;
            }
        }
    }
    
    // Filter by threshold and collect results
    vector<uint> results;
    results.reserve(candidates.size());
    
    for (const auto& [doc_id, score] : candidates) {
        if (score >= threshold_) {
            results.push_back(doc_id);
        }
    }
    
    return results;
}

GTnn::OptimizedLinscan::OptimizedLinscan(double threshold) : threshold_(threshold) {}

void GTnn::OptimizedLinscan::build_index(const sparse_mat_t& data) {
    cout << "Building optimized inverted index..." << endl;
    auto start = chrono::high_resolution_clock::now();
    
    inverted_index_.clear();
    data_size_ = data.size();
    
    // Pre-allocate candidate scores array (reused across queries)
    candidate_scores_.resize(data_size_, 0.0);
    candidate_flags_.resize(data_size_, false);
    
    for (size_t doc_id = 0; doc_id < data.size(); doc_id++) {
        for (const auto& elem : data[doc_id]) {
            inverted_index_[elem.index].emplace_back(doc_id, elem.value);
        }
    }
    
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    
    cout << "Optimized index build time: " << duration.count() << " ms" << endl;
}

vector<vector<uint>> GTnn::OptimizedLinscan::search(const sparse_mat_t& queries) {
    vector<vector<uint>> results(queries.size());
    
    auto start = chrono::high_resolution_clock::now();
    
    for (size_t q = 0; q < queries.size(); q++) {
        results[q] = search_single_query_optimized(queries[q]);
    }
    
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
    
    cout << "Optimized LINSCAN Search - Total time: " << duration.count() / 1000.0 << " ms" << endl;
    cout << "Optimized LINSCAN Search - Avg time per query: " << 
            (duration.count() / 1000.0) / queries.size() << " ms" << endl;
    
    return results;
}

vector<uint> GTnn::OptimizedLinscan::search_single_query_optimized(const sparse_vec_t& query) {
    active_candidates_.clear();
    
    // Accumulate scores using pre-allocated arrays
    for (const auto& q_elem : query) {
        uint term_id = q_elem.index;
        float q_value = q_elem.value;
        
        auto it = inverted_index_.find(term_id);
        if (it != inverted_index_.end()) {
            for (const auto& [doc_id, doc_value] : it->second) {
                if (!candidate_flags_[doc_id]) {
                    candidate_flags_[doc_id] = true;
                    active_candidates_.push_back(doc_id);
                }
                candidate_scores_[doc_id] += q_value * doc_value;
            }
        }
    }
    
    // Filter by threshold and collect results
    vector<uint> results;
    results.reserve(active_candidates_.size());
    
    for (uint doc_id : active_candidates_) {
        if (candidate_scores_[doc_id] >= threshold_) {
            results.push_back(doc_id);
        }
        // Reset for next query
        candidate_scores_[doc_id] = 0.0;
        candidate_flags_[doc_id] = false;
    }
    
    return results;
}

// Implementation of comparison functions
namespace GTnn {
    void compare_approaches(const sparse_mat_t& data, const sparse_mat_t& queries, double threshold) {
        cout << "\n=== COMPARISON: Naive Search vs LINSCAN ===" << endl;
        cout << "Data size: " << data.size() << " vectors" << endl;
        cout << "Query size: " << queries.size() << " vectors" << endl;
        cout << "Threshold: " << threshold << endl;
        
        // Calculate average sparsity
        double avg_data_sparsity = 0, avg_query_sparsity = 0;
        for (const auto& vec : data) avg_data_sparsity += vec.size();
        for (const auto& vec : queries) avg_query_sparsity += vec.size();
        avg_data_sparsity /= data.size();
        avg_query_sparsity /= queries.size();
        
        cout << "Average data vector sparsity: " << avg_data_sparsity << endl;
        cout << "Average query vector sparsity: " << avg_query_sparsity << endl;
        
        // Test 1: Naive Search
        cout << "\n--- Test 1: Naive Search ---" << endl;
        NaiveSearch naive(threshold);
        auto naive_results = naive.search(queries, data);
        
        // Test 2: LINSCAN Style
        cout << "\n--- Test 2: LINSCAN Style ---" << endl;
        LinscanStyle linscan(threshold);
        linscan.build_index(data);
        auto linscan_results = linscan.search(queries);
        
        // Test 3: Optimized LINSCAN
        cout << "\n--- Test 3: Optimized LINSCAN ---" << endl;
        OptimizedLinscan opt_linscan(threshold);
        opt_linscan.build_index(data);
        auto opt_results = opt_linscan.search(queries);
        
        // Test 4: Parallel LINSCAN
        cout << "\n--- Test 4: Parallel LINSCAN ---" << endl;
        auto parallel_results = linscan.search_parallel(queries);
        
        // Verify results match
        cout << "\n--- Verification ---" << endl;
        bool naive_vs_linscan = verify_results_match(naive_results, linscan_results);
        bool linscan_vs_opt = verify_results_match(linscan_results, opt_results);
        bool opt_vs_parallel = verify_results_match(opt_results, parallel_results);
        
        cout << "Naive vs LINSCAN results match: " << (naive_vs_linscan ? "YES" : "NO") << endl;
        cout << "LINSCAN vs Optimized match: " << (linscan_vs_opt ? "YES" : "NO") << endl;
        cout << "Optimized vs Parallel match: " << (opt_vs_parallel ? "YES" : "NO") << endl;
        
        // Result statistics
        cout << "\n--- Result Statistics ---" << endl;
        print_result_stats("Naive", naive_results);
        print_result_stats("LINSCAN", linscan_results);
        print_result_stats("Optimized", opt_results);
        print_result_stats("Parallel", parallel_results);
    }
    
    bool verify_results_match(const vector<vector<uint>>& results1, const vector<vector<uint>>& results2) {
        if (results1.size() != results2.size()) return false;
        
        for (size_t i = 0; i < results1.size(); i++) {
            set<uint> set1(results1[i].begin(), results1[i].end());
            set<uint> set2(results2[i].begin(), results2[i].end());
            if (set1 != set2) return false;
        }
        return true;
    }
    
    void print_result_stats(const string& name, const vector<vector<uint>>& results) {
        size_t total_results = 0;
        size_t max_results = 0;
        
        for (const auto& result : results) {
            total_results += result.size();
            max_results = max(max_results, result.size());
        }
        
        cout << name << " - Total results: " << total_results 
             << ", Avg per query: " << (double)total_results / results.size()
             << ", Max per query: " << max_results << endl;
    }
    
    void analyze_threshold_distribution(const sparse_mat_t& data, const sparse_mat_t& queries, int num_samples) {
        cout << "\n=== THRESHOLD DISTRIBUTION ANALYSIS ===" << endl;
        
        vector<double> sample_scores;
        sample_scores.reserve(num_samples);
        
        // Sample random query-document pairs
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> query_dist(0, queries.size() - 1);
        uniform_int_distribution<> data_dist(0, data.size() - 1);
        
        for (int i = 0; i < num_samples; i++) {
            int q_idx = query_dist(gen);
            int d_idx = data_dist(gen);
            
            // Compute dot product
            double score = 0.0;
            size_t qi = 0, di = 0;
            
            while (qi < queries[q_idx].size() && di < data[d_idx].size()) {
                if (queries[q_idx][qi].index == data[d_idx][di].index) {
                    score += queries[q_idx][qi].value * data[d_idx][di].value;
                    qi++; di++;
                } else if (queries[q_idx][qi].index < data[d_idx][di].index) {
                    qi++;
                } else {
                    di++;
                }
            }
            
            if (score > 0) {  // Only keep non-zero scores
                sample_scores.push_back(score);
            }
        }
        
        if (sample_scores.empty()) {
            cout << "No non-zero similarity scores found in sample!" << endl;
            return;
        }
        
        sort(sample_scores.begin(), sample_scores.end());
        
        cout << "Non-zero similarity scores (" << sample_scores.size() << " out of " << num_samples << " samples):" << endl;
        cout << "Min: " << sample_scores.front() << endl;
        cout << "Max: " << sample_scores.back() << endl;
        cout << "Median: " << sample_scores[sample_scores.size()/2] << endl;
        cout << "90th percentile: " << sample_scores[sample_scores.size() * 0.9] << endl;
        cout << "99th percentile: " << sample_scores[sample_scores.size() * 0.99] << endl;
        
        // Suggest reasonable thresholds
        double p95 = sample_scores[sample_scores.size() * 0.95];
        double p90 = sample_scores[sample_scores.size() * 0.90];
        double p80 = sample_scores[sample_scores.size() * 0.80];
        
        cout << "\nSuggested thresholds:" << endl;
        cout << "High selectivity (top 5%): " << p95 << endl;
        cout << "Medium selectivity (top 10%): " << p90 << endl;
        cout << "Low selectivity (top 20%): " << p80 << endl;
    }
}

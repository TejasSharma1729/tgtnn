/**
 * Comparison: Naive Search vs LINSCAN
 * 
 * This file implements both approaches for direct performance comparison
 * and analysis of their differences.
 */

#include "sparse_types.hpp"
#include <unordered_map>
#include <chrono>
#include <thread>
#include <future>
#include <random>
#include <algorithm>

using namespace std;


/**
 * APPROACH 1: Traditional Naive Search (Your current implementation)
 */
class ThresholdNaiveSearch {
public:
    ThresholdNaiveSearch(double threshold = 0.8);
    std::vector<std::vector<uint>> search(const SparseMat& queries, const SparseMat& data);

private:
    double threshold_;
    double dot_product_naive(const SparseVec& q, const SparseVec& d);
};

// Method implementations for ThresholdNaiveSearch
ThresholdNaiveSearch::ThresholdNaiveSearch(double threshold) : threshold_(threshold) {}

std::vector<std::vector<uint>> ThresholdNaiveSearch::search(const SparseMat& queries, const SparseMat& data) {
    std::vector<std::vector<uint>> results(queries.size());
    
    for (size_t q = 0; q < queries.size(); q++) {
        for (size_t d = 0; d < data.size(); d++) {
            if (dot_product_naive(queries[q], data[d]) >= threshold_) {
                results[q].push_back(d);
            }
        }
    }
    
    return results;
}

double ThresholdNaiveSearch::dot_product_naive(const SparseVec& q, const SparseVec& d) {
    double result = 0.0;
    size_t i = 0, j = 0;
    while (i < q.size() && j < d.size()) {
        if (q[i].index == d[j].index) {
            result += q[i].value * d[j].value;
            i++;
            j++;
        } else if (q[i].index < d[j].index) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}

/**
 * APPROACH 2: LINSCAN-style Implementation
 */
class ThresholdLinscanStyle {
public:
    ThresholdLinscanStyle(double threshold = 0.8);
    void build_index(const SparseMat& data);
    std::vector<std::vector<uint>> search(const SparseMat& queries);
    std::vector<std::vector<uint>> search_parallel(const SparseMat& queries, int num_threads);

private:
    double threshold_;
    std::unordered_map<uint, std::vector<std::pair<uint, float>>> inverted_index_;
    SparseMat data_;
    size_t data_size_;
    void buildInvertedIndex(const SparseMat& data);
    std::vector<uint> search_single_query(const SparseVec& query);
    std::vector<uint> search_single_query_optimized(const SparseVec& query);
};

class ThresholdLinscanOptimized {
public:
    ThresholdLinscanOptimized(double threshold = 0.8);
    void build_index(const SparseMat& data);
    std::vector<std::vector<uint>> search(const SparseMat& queries);

private:
    double threshold_;
    std::unordered_map<uint, std::vector<std::pair<uint, float>>> inverted_index_;
    size_t data_size_;
    
    // Reused arrays to avoid allocations
    std::vector<double> candidate_scores_;
    std::vector<bool> candidate_flags_;
    std::vector<uint> active_candidates_;
    
    std::vector<uint> search_single_query_optimized(const SparseVec& query);
};

/**
 * Comparison and analysis functions
 */
void compare_approaches(const SparseMat& data, const SparseMat& queries, double threshold = 0.8);
bool verify_results_match(const std::vector<std::vector<uint>>& results1, const std::vector<std::vector<uint>>& results2);
void print_result_stats(const std::string& name, const std::vector<std::vector<uint>>& results);
void analyze_threshold_distribution(const SparseMat& data, const SparseMat& queries, int num_samples = 1000);
    
ThresholdLinscanStyle::ThresholdLinscanStyle(double threshold) : threshold_(threshold) {}

void ThresholdLinscanStyle::build_index(const SparseMat& data) {
    cout << "Building inverted index..." << endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    inverted_index_.clear();
    data_size_ = data.size();
    
    for (size_t doc_id = 0; doc_id < data.size(); doc_id++) {
        for (const auto& elem : data[doc_id]) {
            inverted_index_[elem.index].emplace_back(doc_id, elem.value);
        }
    }
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    
    cout << "Index build time: " << duration.count() << " ms" << endl;
    cout << "Index size: " << inverted_index_.size() << " terms" << endl;
    
    // Statistics
    size_t total_postings = 0;
    size_t max_posting_length = 0;
    for (const auto& [term, postings] : inverted_index_) {
        total_postings += postings.size();
        max_posting_length = std::max(max_posting_length, postings.size());
    }
    cout << "Average posting length: " << (double)total_postings / inverted_index_.size() << endl;
    cout << "Max posting length: " << max_posting_length << endl;
}

std::vector<std::vector<uint>> ThresholdLinscanStyle::search(const SparseMat& queries) {
    std::vector<std::vector<uint>> results(queries.size());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t q = 0; q < queries.size(); q++) {
        results[q] = search_single_query(queries[q]);
    }
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    cout << "LINSCAN Search - Total time: " << duration.count() / 1000.0 << " ms" << endl;
    cout << "LINSCAN Search - Avg time per query: " << 
            (duration.count() / 1000.0) / queries.size() << " ms" << endl;
    
    return results;
}

std::vector<std::vector<uint>> ThresholdLinscanStyle::search_parallel(const SparseMat& queries, int num_threads) {
    std::vector<std::vector<uint>> results(queries.size());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::future<void>> futures;
    std::atomic<size_t> query_idx(0);
    
    for (int t = 0; t < num_threads; t++) {
        futures.push_back(std::async(std::launch::async, [&]() {
            size_t idx;
            while ((idx = query_idx.fetch_add(1)) < queries.size()) {
                results[idx] = search_single_query(queries[idx]);
            }
        }));
    }
    
    for (auto& f : futures) {
        f.wait();
    }
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    cout << "LINSCAN Parallel Search - Total time: " << duration.count() / 1000.0 << " ms" << endl;
    cout << "LINSCAN Parallel Search - Avg time per query: " << 
            (duration.count() / 1000.0) / queries.size() << " ms" << endl;
    
    return results;
}

std::vector<uint> ThresholdLinscanStyle::search_single_query(const SparseVec& query) {
    // Use unordered_map for candidate accumulation
    std::unordered_map<uint, double> candidates;
    
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
    std::vector<uint> results;
    results.reserve(candidates.size());
    
    for (const auto& [doc_id, score] : candidates) {
        if (score >= threshold_) {
            results.push_back(doc_id);
        }
    }
    
    return results;
}

ThresholdLinscanOptimized::ThresholdLinscanOptimized(double threshold) : threshold_(threshold) {}

void ThresholdLinscanOptimized::build_index(const SparseMat& data) {
    cout << "Building optimized inverted index..." << endl;
    auto start = std::chrono::high_resolution_clock::now();
    
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
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    
    cout << "Optimized index build time: " << duration.count() << " ms" << endl;
}

std::vector<std::vector<uint>> ThresholdLinscanOptimized::search(const SparseMat& queries) {
    std::vector<std::vector<uint>> results(queries.size());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t q = 0; q < queries.size(); q++) {
        results[q] = search_single_query_optimized(queries[q]);
    }
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    cout << "Optimized LINSCAN Search - Total time: " << duration.count() / 1000.0 << " ms" << endl;
    cout << "Optimized LINSCAN Search - Avg time per query: " << 
            (duration.count() / 1000.0) / queries.size() << " ms" << endl;
    
    return results;
}

std::vector<uint> ThresholdLinscanOptimized::search_single_query_optimized(const SparseVec& query) {
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
    std::vector<uint> results;
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


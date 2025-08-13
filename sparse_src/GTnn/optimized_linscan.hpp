#pragma once
#include "header.hpp"
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>

namespace GTnn {
    
    class OptimizedLinscan {
    public:
        OptimizedLinscan(double threshold = 0.8);
        void fit(const sparse_mat_t& data);
        vector<vector<uint>> search(const sparse_mat_t& queries, int num_threads = NUM_THREADS);
        
    private:
        double threshold_;
        unordered_map<uint, vector<pair<uint, float>>> inverted_index_;
        sparse_mat_t data_;
        
        void buildInvertedIndex(const sparse_mat_t& data);
        vector<uint> searchSingle(const sparse_vec_t& query);
    };
    
    // High-performance batch processor similar to LINSCAN's approach
    class BatchProcessor {
    public:
        static constexpr size_t BATCH_SIZE = 1000;
        
        static vector<vector<uint>> processBatch(
            const sparse_mat_t& queries,
            const sparse_mat_t& data,
            double threshold,
            int num_threads = NUM_THREADS
        );
        
    private:
        static vector<uint> searchLinearScan(
            const sparse_vec_t& query,
            const sparse_mat_t& data,
            double threshold
        );
        
        // Highly optimized inline dot product
        static inline double dot_product_optimized_inline(
            const sparse_vec_t& one,
            const sparse_vec_t& two
        );
    };
}

// OptimizedLinscan Implementation
GTnn::OptimizedLinscan::OptimizedLinscan(double threshold) : threshold_(threshold) {}

void GTnn::OptimizedLinscan::fit(const sparse_mat_t& data) {
    // Build inverted index similar to LINSCAN
    buildInvertedIndex(data);
}

vector<vector<uint>> GTnn::OptimizedLinscan::search(const sparse_mat_t& queries, int num_threads) {
    vector<vector<uint>> results(queries.size());
    
    if (num_threads == 1) {
        // Single-threaded version
        for (size_t i = 0; i < queries.size(); i++) {
            results[i] = searchSingle(queries[i]);
        }
    } else {
        // Multi-threaded version
        vector<thread> threads;
        atomic<size_t> query_idx(0);
        
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back([&]() {
                size_t idx;
                while ((idx = query_idx.fetch_add(1)) < queries.size()) {
                    results[idx] = searchSingle(queries[idx]);
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
    }
    
    return results;
}

void GTnn::OptimizedLinscan::buildInvertedIndex(const sparse_mat_t& data) {
    data_ = data;
    inverted_index_.clear();
    
    for (size_t doc_id = 0; doc_id < data.size(); doc_id++) {
        for (const auto& elem : data[doc_id]) {
            inverted_index_[elem.index].emplace_back(doc_id, elem.value);
        }
    }
}

vector<uint> GTnn::OptimizedLinscan::searchSingle(const sparse_vec_t& query) {
    unordered_map<uint, double> candidates;
    
    // Accumulate scores for candidate documents
    for (const auto& q_elem : query) {
        uint term_id = q_elem.index;
        float q_val = q_elem.value;
        
        auto it = inverted_index_.find(term_id);
        if (it != inverted_index_.end()) {
            for (const auto& [doc_id, doc_val] : it->second) {
                candidates[doc_id] += q_val * doc_val;
            }
        }
    }
    
    // Filter by threshold
    vector<uint> results;
    results.reserve(candidates.size());
    
    for (const auto& [doc_id, score] : candidates) {
        if (score >= threshold_) {
            results.push_back(doc_id);
        }
    }
    
    return results;
}

// BatchProcessor Implementation
vector<vector<uint>> GTnn::BatchProcessor::processBatch(
    const sparse_mat_t& queries,
    const sparse_mat_t& data,
    double threshold,
    int num_threads
) {
    vector<vector<uint>> all_results(queries.size());
    
    // Process in batches to optimize memory usage
    for (size_t start = 0; start < queries.size(); start += BATCH_SIZE) {
        size_t end = min(start + BATCH_SIZE, queries.size());
        
        vector<thread> threads;
        atomic<size_t> query_idx(start);
        
        for (int t = 0; t < num_threads; t++) {
            threads.emplace_back([&]() {
                size_t idx;
                while ((idx = query_idx.fetch_add(1)) < end) {
                    all_results[idx] = searchLinearScan(queries[idx], data, threshold);
                }
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
    }
    
    return all_results;
}

vector<uint> GTnn::BatchProcessor::searchLinearScan(
    const sparse_vec_t& query,
    const sparse_mat_t& data,
    double threshold
) {
    vector<uint> results;
    results.reserve(data.size() / 100); // Rough estimate
    
    for (size_t i = 0; i < data.size(); i++) {
        double score = dot_product_optimized_inline(query, data[i]);
        if (score >= threshold) {
            results.push_back(i);
        }
    }
    
    return results;
}

// Highly optimized inline dot product
inline double GTnn::BatchProcessor::dot_product_optimized_inline(
    const sparse_vec_t& one,
    const sparse_vec_t& two
) {
    double result = 0;
    size_t i = 0, j = 0;
    const size_t size_one = one.size();
    const size_t size_two = two.size();
    
    while (i < size_one && j < size_two) {
        const uint idx_one = one[i].index;
        const uint idx_two = two[j].index;
        
        if (idx_one == idx_two) {
            result += one[i].value * two[j].value;
            ++i;
            ++j;
        } else if (idx_one < idx_two) {
            ++i;
        } else {
            ++j;
        }
    }
    
    return result;
}

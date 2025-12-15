#include "sparse_types.hpp"
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>

class ThresholdLinscanOptimized {
public:
    ThresholdLinscanOptimized(double threshold = 0.8);
    void fit(const SparseMat& data);
    std::vector<std::vector<uint>> search(const SparseMat& queries, int num_threads = NUM_THREADS);

private:
    double threshold_;
    std::unordered_map<uint, std::vector<std::pair<uint, float>>> inverted_index_;
    SparseMat data_;

    void buildInvertedIndex(const SparseMat& data);
    std::vector<uint> searchSingle(const SparseVec& query);
};

// High-performance batch processor similar to LINSCAN's approach
class ThresholdBatchProcessor {
public:
    static constexpr size_t BATCH_SIZE = 1000;

    static std::vector<std::vector<uint>> processBatch(
        const SparseMat& queries,
        const SparseMat& data,
        double threshold,
        int num_threads = NUM_THREADS
    );

private:
    static std::vector<uint> searchLinearScan(
        const SparseVec& query,
        const SparseMat& data,
        double threshold
    );
        
    static inline double dot_product_optimized_inline(
        const SparseVec& one,
        const SparseVec& two
    );
};


// Method implementations for ThresholdLinscanOptimized
ThresholdLinscanOptimized::ThresholdLinscanOptimized(double threshold) : threshold_(threshold) {}

void ThresholdLinscanOptimized::fit(const SparseMat& data) {
    buildInvertedIndex(data);
}

std::vector<std::vector<uint>> ThresholdLinscanOptimized::search(const SparseMat& queries, int num_threads) {
    std::vector<std::vector<uint>> results(queries.size());
    
    if (num_threads == 1) {
        for (size_t i = 0; i < queries.size(); i++) {
            results[i] = searchSingle(queries[i]);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> query_idx(0);
        
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

void ThresholdLinscanOptimized::buildInvertedIndex(const SparseMat& data) {
    data_ = data;
    inverted_index_.clear();
    
    for (size_t doc_id = 0; doc_id < data.size(); doc_id++) {
        for (const auto& elem : data[doc_id]) {
            inverted_index_[elem.index].emplace_back(doc_id, elem.value);
        }
    }
}

std::vector<uint> ThresholdLinscanOptimized::searchSingle(const SparseVec& query) {
    std::unordered_map<uint, double> candidates;
    for (const auto& q_elem : query) {
        if (inverted_index_.find(q_elem.index) != inverted_index_.end()) {
            for (const auto& [doc_id, value] : inverted_index_[q_elem.index]) {
                candidates[doc_id] += q_elem.value * value;
            }
        }
    }
    std::vector<uint> results;
    for (const auto& [doc_id, score] : candidates) {
        if (score >= threshold_) {
            results.push_back(doc_id);
        }
    }
    return results;
}

// Method implementations for ThresholdBatchProcessor
std::vector<std::vector<uint>> ThresholdBatchProcessor::processBatch(
    const SparseMat& queries,
    const SparseMat& data,
    double threshold,
    int num_threads
) {
    std::vector<std::vector<uint>> results(queries.size());
    std::vector<std::thread> threads;
    std::atomic<size_t> query_idx(0);
    
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&]() {
            size_t idx;
            while ((idx = query_idx.fetch_add(1)) < queries.size()) {
                results[idx] = searchLinearScan(queries[idx], data, threshold);
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return results;
}

std::vector<uint> ThresholdBatchProcessor::searchLinearScan(
    const SparseVec& query,
    const SparseMat& data,
    double threshold
) {
    std::vector<uint> results;
    for (size_t i = 0; i < data.size(); i++) {
        double dot = dot_product(const_cast<SparseVec&>(query), const_cast<SparseMat&>(data)[i]);
        if (dot >= threshold) {
            results.push_back(i);
        }
    }
    return results;
}

// All GTnn and legacy typedefs removed. All types and methods are PascalCase and Python/numpy compatible.

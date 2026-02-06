#include "sparse_types.hpp"
#include <unordered_map>
#include <chrono>

/**
 * @brief LINSCAN-style threshold-based search using inverted indices
 * Builds inverted index on dataset for efficient threshold-based similarity search
 */
class Linscan {
public:
    /**
     * @brief Constructor - builds inverted index from dataset
     * @param dataset Sparse matrix containing data vectors
     * @param threshold Similarity threshold for search (default 0.8)
     */
    Linscan(SparseMat &dataset, double threshold = 0.8);
    
    /**
     * @brief Destructor
     */
    ~Linscan() = default;
    
    /**
     * @brief Search for vectors above threshold for a single query
     * @param query Query sparse vector
     * @return Pair of (result indices, number of dot products computed)
     */
    std::pair<std::vector<uint>, size_t> search(const SparseVec &query);
    
    /**
     * @brief Search for vectors above threshold for multiple queries
     * @param queries Batch of query sparse vectors
     * @return Pair of (results per query, total dot products computed)
     */
    std::pair<std::vector<std::vector<uint>>, size_t> search_multiple(const SparseMat &queries);
    
    /**
     * @brief Verify search results against brute-force and compute metrics
     * @param query Query sparse vector
     * @param result Indices returned by search algorithm
     * @return Array of [time_taken_ms, recall, precision]
     */
    std::array<double, 3> verify_results(const SparseVec &query, const std::vector<uint> &result);
    
private:
    SparseMat dataset_;
    double threshold_;
    std::unordered_map<uint, std::vector<std::pair<uint, float>>> inverted_index_;
    
    /**
     * @brief Build inverted index from dataset
     */
    void build_inverted_index();
    
    /**
     * @brief Search single query using inverted index
     */
    std::vector<uint> search_single(const SparseVec &query);
};

/**
 * Constructor
 */
Linscan::Linscan(SparseMat &dataset, double threshold) 
    : dataset_(dataset), threshold_(threshold) {
    build_inverted_index();
}

/**
 * Build inverted index
 */
void Linscan::build_inverted_index() {
    inverted_index_.clear();
    
    for (uint doc_id = 0; doc_id < dataset_.size(); doc_id++) {
        for (const auto &elem : dataset_[doc_id]) {
            inverted_index_[elem.index].push_back({doc_id, elem.value});
        }
    }
}

/**
 * Single query search using inverted index
 */
std::vector<uint> Linscan::search_single(const SparseVec &query) {
    std::unordered_map<uint, double> candidates;
    
    // Accumulate scores for all candidates from inverted index
    for (const auto &q_elem : query) {
        auto it = inverted_index_.find(q_elem.index);
        if (it != inverted_index_.end()) {
            for (const auto &[doc_id, value] : it->second) {
                candidates[doc_id] += q_elem.value * value;
            }
        }
    }
    
    // Filter by threshold
    std::vector<uint> result;
    for (const auto &[doc_id, score] : candidates) {
        if (score >= threshold_) {
            result.push_back(doc_id);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

/**
 * Single query search
 */
std::pair<std::vector<uint>, size_t> Linscan::search(const SparseVec &query) {
    size_t dot_count = 0;
    
    // Count actual dot products performed (num elements in query × postings per element)
    for (const auto &q_elem : query) {
        auto it = inverted_index_.find(q_elem.index);
        if (it != inverted_index_.end()) {
            dot_count += it->second.size();
        }
    }
    
    std::vector<uint> result = search_single(query);
    return {result, dot_count};
}

/**
 * Batch query search
 */
std::pair<std::vector<std::vector<uint>>, size_t> Linscan::search_multiple(const SparseMat &queries) {
    std::vector<std::vector<uint>> results;
    size_t total_dots = 0;
    
    for (const auto &query : queries) {
        auto [res, dots] = search(query);
        results.push_back(res);
        total_dots += dots;
    }
    
    return {results, total_dots};
}

/**
 * Verify results
 */
std::array<double, 3> Linscan::verify_results(const SparseVec &query, const std::vector<uint> &result) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<uint> true_result = search_single(query);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double time_ms = duration.count() / 1000.0;
    
    // Compute recall and precision
    std::set<uint> true_set(true_result.begin(), true_result.end());
    std::set<uint> res_set(result.begin(), result.end());
    std::set<uint> match_set;
    
    std::set_intersection(true_set.begin(), true_set.end(),
                         res_set.begin(), res_set.end(),
                         std::inserter(match_set, match_set.begin()));
    
    double recall = (true_set.size() == 0) ? 1.0 : 
                    (static_cast<double>(match_set.size()) / true_set.size());
    double precision = (res_set.size() == 0) ? 1.0 : 
                       (static_cast<double>(match_set.size()) / res_set.size());
    
    return {time_ms, recall, precision};
}

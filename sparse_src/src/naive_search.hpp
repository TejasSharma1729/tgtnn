#include "sparse_types.hpp"
#include <chrono>

/**
 * @brief Naive search implementation for both KNN and threshold-based search
 * Uses brute-force comparison against all dataset vectors
 * Supports both single and batch query processing
 */
class NaiveSearch {
public:
    /**
     * @brief Constructor - initializes search parameters
     * @param dataset Sparse matrix containing data vectors
     * @param threshold Threshold for threshold-based search (default 0.8)
     * @param k Number of nearest neighbors for KNN search (default 1)
     * @param is_knn If true, performs KNN search; if false, performs threshold search
     */
    NaiveSearch(SparseMat &dataset, double threshold = 0.8, size_t k = 1, bool is_knn = true);
    
    /**
     * @brief Destructor
     */
    ~NaiveSearch() = default;
    
    /**
     * @brief Search for neighbors of a single query vector
     * @param query Query sparse vector
     * @return Pair of (result indices, number of dot products computed)
     */
    std::pair<std::vector<uint>, size_t> search(SparseVec &query);
    
    /**
     * @brief Search for neighbors of multiple query vectors (batch)
     * @param queries Batch of query sparse vectors
     * @return Pair of (results per query, total dot products computed)
     */
    std::pair<std::vector<std::vector<uint>>, size_t> search_multiple(SparseMat &queries);
    
    /**
     * @brief Verify search results against brute-force and compute metrics
     * @param query Query sparse vector
     * @param result Indices returned by search algorithm
     * @return Array of [time_taken_ms, recall, precision]
     */
    std::array<double, 3> verify_results(SparseVec &query, std::vector<uint> &result);
    
private:
    SparseMat dataset_;
    double threshold_;
    size_t k_;
    bool is_knn_;
    
    /**
     * @brief Internal helper for KNN search
     */
    std::vector<uint> search_knn(SparseVec &query);
    
    /**
     * @brief Internal helper for threshold search
     */
    std::vector<uint> search_threshold(SparseVec &query);
};

/**
 * NaiveSearch Constructor
 */
NaiveSearch::NaiveSearch(SparseMat &dataset, double threshold, size_t k, bool is_knn) 
    : dataset_(dataset), threshold_(threshold), k_(k), is_knn_(is_knn) {}

/**
 * KNN search implementation
 */
std::vector<uint> NaiveSearch::search_knn(SparseVec &query) {
    std::vector<std::pair<double, uint>> scores;
    
    for (uint i = 0; i < dataset_.size(); i++) {
        double score = dot_product(query, dataset_[i]);
        scores.push_back({score, i});
    }
    
    // Sort descending by score
    std::sort(scores.begin(), scores.end(), 
              [](const auto &a, const auto &b) { return a.first > b.first; });
    
    // Return top-k indices
    std::vector<uint> result;
    for (size_t i = 0; i < k_ && i < scores.size(); i++) {
        result.push_back(scores[i].second);
    }
    std::sort(result.begin(), result.end());
    return result;
}

/**
 * Threshold search implementation
 */
std::vector<uint> NaiveSearch::search_threshold(SparseVec &query) {
    std::vector<uint> result;
    
    for (uint i = 0; i < dataset_.size(); i++) {
        if (dot_product(query, dataset_[i]) >= threshold_) {
            result.push_back(i);
        }
    }
    
    return result;
}

/**
 * Single query search
 */
std::pair<std::vector<uint>, size_t> NaiveSearch::search(SparseVec &query) {
    size_t dot_count = dataset_.size();
    
    std::vector<uint> result = is_knn_ ? search_knn(query) : search_threshold(query);
    
    return {result, dot_count};
}

/**
 * Batch query search
 */
std::pair<std::vector<std::vector<uint>>, size_t> NaiveSearch::search_multiple(SparseMat &queries) {
    std::vector<std::vector<uint>> results;
    size_t total_dots = 0;
    
    for (auto query : queries) {
        auto [res, dots] = search(query);
        results.push_back(res);
        total_dots += dots;
    }
    
    return {results, total_dots};
}

/**
 * Verify results
 */
std::array<double, 3> NaiveSearch::verify_results(SparseVec &query, std::vector<uint> &result) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<uint> true_result;
    if (is_knn_) {
        auto res = search_knn(query);
        true_result = res;
    } else {
        true_result = search_threshold(query);
    }
    
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

#pragma once

#include "sparse_types.hpp"
#include <thread>
#include <array>
#include <chrono>
#include <set>
#include <random>
#include <cmath>

#define THRESHOLD_LEVELS 7
#define INVERTED_LEVELS 7

/** @brief Type alias for a single threshold index structure - array of sparse vectors */
using ThresholdIndexSingle = std::array<SparseVec, (1 << THRESHOLD_LEVELS)>;

/** @brief Type alias for the complete threshold index - vector of single indices */
using ThresholdIndexIdx = std::vector<ThresholdIndexSingle>;

/**
 * @brief Build a threshold index from a sparse matrix
 * Creates a hierarchical binary tree structure for efficient threshold-based search
 * @param matrix Input sparse matrix to index
 * @return Vector of index structures ready for threshold search
 */
ThresholdIndexIdx build_threshold_index_dataset(SparseMat &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << THRESHOLD_LEVELS) - 1) / (1 << THRESHOLD_LEVELS);
    ThresholdIndexIdx data_index(num_indices);
    for (uint i = 0; i < num_indices; i++) {
        const uint offset = i * (1 << THRESHOLD_LEVELS);
        for (uint j = 0; j < (1 << THRESHOLD_LEVELS); j++) {
            const uint index = offset + j;
            if (index < num_vectors) {
                data_index[i][j] = matrix[index];
            }
        }
        for (uint n = 1; n <= THRESHOLD_LEVELS; n++) {
            const uint step = 1 << n;
            for (uint j = 0; j < (1 << THRESHOLD_LEVELS); j += step) {
                data_index[i][j] = add_sparse(data_index[i][j], data_index[i][j + step / 2]);
            }
        }
    }
    return data_index;
}

/**
 * @brief Index structure for efficient threshold-based search on sparse data
 * Performs hierarchical binary partitioning for pruned search
 */
class ThresholdIndexDataset {
public:
    /**
     * @brief Constructor - initializes with dataset and builds the index
     * @param dataset Sparse matrix reference to index
     * @param threshold Similarity threshold for search (default 0.8)
     * @param use_threading Enable multi-threaded search (default false)
     */
    ThresholdIndexDataset(SparseMat &dataset, double threshold = 0.8, bool use_threading = false);

    /**
     * @brief Search for all vectors above the threshold for a single query
     * @param query Query sparse vector
     * @return Pair of (matching vector indices, number of dot products computed)
     */
    std::pair<std::vector<uint>, size_t> search(SparseVec &query);

    /**
     * @brief Search for all vectors above the threshold for multiple queries
     * @param queries Multiple query sparse vectors
     * @return Pair of (results per query, total dot products computed)
     */
    std::pair<std::vector<std::vector<uint>>, size_t> search_multiple(SparseMat &queries);

    /**
     * @brief Verify search results against brute force and compute metrics
     * @param query Query sparse vector
     * @param result Indices returned by the search algorithm
     * @return Array of [time (ms), recall, precision]
     */
    std::array<double, 3> verify_results(SparseVec &query, std::vector<uint> &result);

private:
    SparseMat data_set;                     /**< Original sparse matrix */
    ThresholdIndexIdx data_index;           /**< Hierarchical index structure */
    SparseMat query_index;                  /**< Hierarchical index for queries (built at search time) */
    size_t dimention;                       /**< Dimensionality of vectors */
    double threshold;                       /**< Similarity threshold */
    bool use_threading;                     /**< Enable multi-threading */

    /**
     * @brief Search within a single index pool
     * @param pool Index structure for this pool
     * @param query Query sparse vector
     * @param pool_index Index of this pool in the full dataset
     * @return Pair of (matching indices in this pool, dot products computed)
     */
    std::pair<std::vector<uint>, size_t> search_pool(ThresholdIndexSingle &pool, SparseVec &query, uint &pool_index);
    
    // For batch search with dual hierarchies
    std::random_device rd;
    std::mt19937 gen;
    uint left_data;
    uint right_data;
    uint left_query;
    uint right_query;
    std::vector<std::vector<uint>> search_res;
    size_t num_dot_products;
    
    // Recursive dual-hierarchy search methods
    void search_single_data(double dot_prod);
    void search_single_query(double dot_prod);
    void search_subspans(double dot_prod);
};

ThresholdIndexDataset::ThresholdIndexDataset(SparseMat &dataset, double threshold_val, bool use_threading_flag)
    : threshold(threshold_val), use_threading(use_threading_flag), gen(rd()) {
    this->data_set = dataset;
    this->dimention = dataset.empty() ? 0 : dataset[0].size();
    this->data_index = build_threshold_index_dataset(this->data_set);
}

/**
 * @brief Search for all vectors above the threshold for a single query
 */
std::pair<std::vector<uint>, size_t> ThresholdIndexDataset::search(SparseVec &query) {
    std::vector<std::pair<std::vector<uint>, size_t>> async_results(this->data_index.size());
    std::array<std::thread, NUM_THREADS> threads;
    auto worker = [this, &async_results, &query] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            async_results[pool_index] = this->search_pool(this->data_index[pool_index], query, pool_index);
        }
    };
    
    if (use_threading && this->data_index.size() > 1) {
        for (uint i = 0; i < NUM_THREADS; i++) {
            uint start = i * (this->data_index.size() / NUM_THREADS);
            uint end = (i + 1) * (this->data_index.size() / NUM_THREADS);
            if (i == NUM_THREADS - 1) {
                end = this->data_index.size();
            }
            threads[i] = std::thread(worker, start, end);
        }
        for (uint i = 0; i < NUM_THREADS; i++) {
            threads[i].join();
        }
    } else {
        worker(0, this->data_index.size());
    }
    
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    for (uint i = 0; i < async_results.size(); i++) {
        result.first.insert(result.first.end(), async_results[i].first.begin(), async_results[i].first.end());
        result.second += async_results[i].second;
    }
    return result;
}

/**
 * @brief Search within a single pool for matches above threshold
 */
std::pair<std::vector<uint>, size_t> ThresholdIndexDataset::search_pool(ThresholdIndexSingle &pool, SparseVec &query, uint &pool_index) {
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    if (pool[0].size() == 0) {
        return result;
    }
    std::array<double, THRESHOLD_LEVELS + 1> dot_products;
    std::array<std::pair<uint, uint>, THRESHOLD_LEVELS + 1> ranges;
    ranges[0] = {0, (1 << THRESHOLD_LEVELS)};
    dot_products[0] = dot_product(query, pool[0]);
    uint dot_idx = 1;
    uint start = 0;
    uint end = 0;
    uint mid = 0;
    while (dot_idx) {
        if (dot_products[dot_idx - 1] < this->threshold) {
            dot_idx--;
            continue;
        }
        std::tie(start, end) = ranges[dot_idx - 1];
        mid = (start + end) / 2;
        if (start + 1 == end) {
            result.first.push_back(mid + pool_index * (1 << THRESHOLD_LEVELS));
            dot_idx--;
            continue;
        }
        result.second++;
        dot_products[dot_idx] = dot_product(query, pool[mid]);
        ranges[dot_idx] = {mid, end};
        dot_products[dot_idx - 1] -= dot_products[dot_idx];
        ranges[dot_idx - 1] = {start, mid};
        dot_idx++;
    }
    return result;
}

/**
 * @brief Search for multiple queries using dual-hierarchy traversal
 */
std::pair<std::vector<std::vector<uint>>, size_t> ThresholdIndexDataset::search_multiple(SparseMat &queries) {
    std::pair<std::vector<std::vector<uint>>, size_t> result;
    result.first.resize(queries.size());
    result.second = 0;
    
    if (this->data_set.size() == 0 || queries.size() == 0) {
        return result;
    }
    
    // Build hierarchical query index
    const uint num_queries = queries.size();
    const uint query_log_size = std::ceil(std::log2(num_queries));
    const uint query_upper_size = 1 << query_log_size;
    this->query_index.clear();
    this->query_index.resize(query_upper_size);
    for (uint i = 0; i < num_queries; i++) {
        this->query_index[i] = queries[i];
    }
    for (uint n = 1; n <= query_log_size; n++) {
        const uint step = 1 << n;
        for (uint i = 0; i < query_upper_size; i += step) {
            this->query_index[i] = add_sparse(this->query_index[i], this->query_index[i + step / 2]);
        }
    }
    
    // Initialize search results
    this->search_res.clear();
    this->search_res.resize(queries.size(), std::vector<uint>());
    this->num_dot_products = 0;
    
    // Get size of data hierarchy
    const uint num_data = this->data_set.size();
    const uint data_log_size = std::ceil(std::log2(num_data));
    const uint data_upper_size = 1 << data_log_size;
    
    // Coarse-grained dual-hierarchy traversal
    const uint dstep = 1 << (std::max(0, (int)data_log_size - 2));
    const uint qstep = 1 << (std::max(0, (int)query_log_size - 2));
    
    // We need to construct aggregates at the coarse level from data_index
    // Access data_index[pool][0] which contains aggregate of that pool
    for (uint pool = 0; pool < this->data_index.size(); pool++) {
        for (uint j = 0; j < query_upper_size; j += qstep) {
            this->left_data = pool * (1 << THRESHOLD_LEVELS);
            this->right_data = std::min((pool + 1) * (1 << THRESHOLD_LEVELS), (uint)this->data_set.size());
            this->left_query = j;
            this->right_query = std::min(j + qstep, query_upper_size);
            
            // Use aggregate from data_index pool
            double dot_prod = dot_product(this->data_index[pool][0], this->query_index[j]);
            this->num_dot_products++;
            
            if (dot_prod >= this->threshold) {
                this->search_subspans(dot_prod);
            }
        }
    }
    
    // Clear query index after search
    this->query_index.clear();
    
    result.first = this->search_res;
    result.second = this->num_dot_products;
    return result;
}

/**
 * @brief Search recursively in a single data span (when data is leaf)
 */
void ThresholdIndexDataset::search_single_data(double dot_prod) {
    if (this->left_query + 1 == this->right_query) {
        if (dot_prod >= this->threshold && this->left_query < this->search_res.size()) {
            this->search_res[this->left_query].push_back(this->left_data);
        }
        return;
    }
    const uint mid_query = (this->left_query + this->right_query) / 2;
    uint orig_left_query = this->left_query;
    uint orig_right_query = this->right_query;
    
    double right_dot_prod = dot_product(this->data_set[this->left_data], this->query_index[mid_query]);
    dot_prod -= right_dot_prod;
    this->num_dot_products++;
    
    if (dot_prod >= this->threshold) {
        this->right_query = mid_query;
        this->search_single_data(dot_prod);
        this->right_query = orig_right_query;
    }
    if (right_dot_prod >= this->threshold) {
        this->left_query = mid_query;
        this->search_single_data(right_dot_prod);
        this->left_query = orig_left_query;
    }
}

/**
 * @brief Search recursively in a single query span (when query is leaf)
 */
void ThresholdIndexDataset::search_single_query(double dot_prod) {
    if (this->left_data + 1 == this->right_data) {
        if (dot_prod >= this->threshold && this->left_query < this->search_res.size()) {
            this->search_res[this->left_query].push_back(this->left_data);
        }
        return;
    }
    const uint mid_data = (this->left_data + this->right_data) / 2;
    uint orig_left_data = this->left_data;
    uint orig_right_data = this->right_data;
    
    double right_dot_prod = dot_product(this->data_set[mid_data], this->query_index[this->left_query]);
    dot_prod -= right_dot_prod;
    this->num_dot_products++;
    
    if (dot_prod >= this->threshold) {
        this->right_data = mid_data;
        this->search_single_query(dot_prod);
        this->right_data = orig_right_data;
    }
    if (right_dot_prod >= this->threshold) {
        this->left_data = mid_data;
        this->search_single_query(right_dot_prod);
        this->left_data = orig_left_data;
    }
}

/**
 * @brief Recursively search both data and query subspans with randomized splitting
 */
void ThresholdIndexDataset::search_subspans(double dot_prod) {
    if (this->left_data + 1 == this->right_data) {
        return search_single_data(dot_prod);
    }
    if (this->left_query + 1 == this->right_query) {
        return search_single_query(dot_prod);
    }
    
    // Randomly choose to split query or data hierarchy
    if (this->gen() % 3) {
        // Split query hierarchy
        const uint mid_query = (this->left_query + this->right_query) / 2;
        uint orig_left_query = this->left_query;
        uint orig_right_query = this->right_query;

        double right_dot_prod = dot_product(this->data_set[this->left_data], this->query_index[mid_query]);
        dot_prod -= right_dot_prod;
        this->num_dot_products++;
        
        if (dot_prod >= this->threshold) {
            this->right_query = mid_query;
            this->search_subspans(dot_prod);
            this->right_query = orig_right_query;
        }
        if (right_dot_prod >= this->threshold) {
            this->left_query = mid_query;
            this->search_subspans(right_dot_prod);
            this->left_query = orig_left_query;
        }
    } else {
        // Split data hierarchy
        const uint mid_data = (this->left_data + this->right_data) / 2;
        uint orig_left_data = this->left_data;
        uint orig_right_data = this->right_data;

        double right_dot_prod = dot_product(this->data_set[mid_data], this->query_index[this->left_query]);
        dot_prod -= right_dot_prod;
        this->num_dot_products++;
        
        if (dot_prod >= this->threshold) {
            this->right_data = mid_data;
            this->search_subspans(dot_prod);
            this->right_data = orig_right_data;
        }
        if (right_dot_prod >= this->threshold) {
            this->left_data = mid_data;
            this->search_subspans(right_dot_prod);
            this->left_data = orig_left_data;
        }
    }
}

/**
 * @brief Verify search results against brute force and compute precision/recall
 */
std::array<double, 3> ThresholdIndexDataset::verify_results(SparseVec &query, std::vector<uint> &result) {
    std::vector<uint> true_result;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data_set.size(); i++) {
        if (dot_product(query, data_set[i]) >= this->threshold) {
            true_result.push_back(i);
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double time = duration.count() / 1.0e+3;
    std::set<uint> true_set(true_result.begin(), true_result.end());
    std::set<uint> res_set(result.begin(), result.end());
    std::set<uint> match_set;
    std::set_intersection(true_set.begin(), true_set.end(),
                   res_set.begin(), res_set.end(),
                   std::inserter(match_set, match_set.begin()));
    double precision = (res_set.size() == 0) ? 1.0 : (static_cast<double>(match_set.size()) / res_set.size());
    double recall = (true_set.size() == 0) ? 1.0 : (static_cast<double>(match_set.size()) / true_set.size());
    return {time, recall, precision};
}

#undef THRESHOLD_LEVELS
#undef INVERTED_LEVELS

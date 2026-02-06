#pragma once

#include "sparse_types.hpp"
#include <thread>
#include <array>
#include <chrono>
#include <numeric>

#define KNN_LEVELS 7
#define INVERTED_LEVELS 7
#define NUM_THREADS 16

/** @brief Template for single KNN dataset index - array of sparse vectors */
template <uint N = KNN_LEVELS> using KNNIndexSingle = std::array<SparseVec, (1 << (N + 1))>;
/** @brief Template for complete KNN dataset index - vector of single indices */
template <uint N = KNN_LEVELS> using KNNIndex = std::vector<KNNIndexSingle<N>>;
/** @brief Result element type for double-group search - array of result vectors */
using KNNIndexDoubleGroupResultElem = std::array<std::vector<uint>, (1 << INVERTED_LEVELS)>;

/**
 * @brief Build KNN index from sparse matrix using hierarchical partitioning
 */
template <uint N = KNN_LEVELS>
KNNIndex<N> build_knn_index(SparseMat &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    KNNIndex<N> data_index(num_indices);
    for (uint i = 0; i < num_indices; i++) {
        const uint offset = i * (1 << N);
        for (uint j = 0; j < (1 << N); j++) {
            const uint index = offset + j;
            if (index < num_vectors) {
                data_index[i][(1 << N) + j] = matrix[index];
            }
        }
        for (int n = N - 1; n >= 0; n--) {
            for (uint j = 0; j < static_cast<uint>(1 << n); j++) {
                const uint index = (1 << n) + j;
                data_index[i][index] = add_sparse(data_index[i][2 * index], data_index[i][2 * index + 1]);
            }
        }
    }
    return data_index;
}

/**
 * @brief Index structure for efficient KNN search on sparse data
 * Performs hierarchical binary partitioning for pruned KNN search
 */
class KNNIndexDataset {
public:
    /**
     * @brief Constructor - initializes with dataset and builds the index
     * @param dataset Sparse matrix reference to index
     * @param k Number of nearest neighbors to search for (default 1)
     * @param use_threading Enable multi-threaded search (default false)
     */
    KNNIndexDataset(SparseMat &dataset, size_t k = 1, bool use_threading = false);
    
    /**
     * @brief Search for k-nearest neighbors to a query vector
     * @param query Query sparse vector
     * @return Pair of (result indices, number of dot products computed)
     */
    std::pair<std::vector<uint>, size_t> search(SparseVec &query);

    /**
     * @brief Search for k-nearest neighbors for multiple queries using double-group testing
     * @param queries Multiple query sparse vectors
     * @return Pair of (vector of result indices per query, total dot products computed)
     */
    std::pair<std::vector<std::vector<uint>>, size_t> search_multiple(SparseMat &queries);
    
    /**
     * @brief Verify search results and compute distance metrics
     * For KNN: recall == precision (returns both as same value)
     * @param query Query sparse vector
     * @param result Result indices from search
     * @return Array of [time_ms, recall, precision]
     */
    std::array<double, 3> verify_results(SparseVec &query, std::vector<uint> &result);

protected:
    SparseMat data_set;
    KNNIndex<KNN_LEVELS> data_index;
    size_t dimention;
    size_t k_val;
    bool use_threading;

    std::pair<std::vector<uint>, size_t> search_threshold(SparseVec &query, double threshold);
    std::pair<std::vector<uint>, size_t> search_pool(KNNIndexSingle<KNN_LEVELS> &pool, SparseVec &query, uint pool_index, double threshold);

    // Double-group testing for batch queries
    std::pair<std::vector<std::vector<uint>>, size_t> search_threshold_batch(const SparseMat &queries, double threshold);
    std::pair<KNNIndexDoubleGroupResultElem, size_t> search_pool_batch(
        KNNIndexSingle<KNN_LEVELS> &pool,
        KNNIndexSingle<INVERTED_LEVELS> &qpool,
        uint pool_index,
        double threshold
    );
};

KNNIndexDataset::KNNIndexDataset(SparseMat &dataset, size_t k, bool use_threading_flag) 
    : k_val(k), use_threading(use_threading_flag) {
    this->data_set = dataset;
    this->dimention = dataset.empty() ? 0 : dataset[0].size();
    this->data_index = build_knn_index<KNN_LEVELS>(this->data_set);
}

std::pair<std::vector<uint>, size_t> KNNIndexDataset::search(SparseVec &query) {
    if (this->data_index.size() == 0) {
        return {std::vector<uint>(), 0};
    }
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    for (double threshold = 0.60; threshold >= 0.0; threshold -= 0.20) {
        auto [res, net_num_dots] = this->search_threshold(query, threshold);
        result.second += net_num_dots;
        if (res.size() >= this->k_val) {
            result.first = res;
            break;
        }
    }
    std::vector<std::pair<double, uint>> sorted_results;
    for (uint i = 0; i < result.first.size(); i++) {
        sorted_results.push_back({dot_product(query, this->data_set[result.first[i]]), result.first[i]});
    }
    std::sort(sorted_results.begin(), sorted_results.end(), std::greater<std::pair<double, uint>>());
    result.first.clear();
    for (uint i = 0; i < this->k_val && i < sorted_results.size(); i++) {
        result.first.push_back(sorted_results[i].second);
    }
    std::sort(result.first.begin(), result.first.end());
    return result;
}

std::pair<std::vector<std::vector<uint>>, size_t> KNNIndexDataset::search_multiple(SparseMat &queries) {
    std::pair<std::vector<std::vector<uint>>, size_t> result;
    result.first.resize(queries.size());
    result.second = 0;
    
    if (this->data_index.size() == 0 || queries.size() == 0) {
        return result;
    }
    
    // Double-group testing: iteratively search with decreasing thresholds
    std::vector<uint> pending_queries(queries.size());
    std::iota(pending_queries.begin(), pending_queries.end(), 0);
    
    SparseMat current_queries;
    for (double threshold : {0.6, 0.4, 0.2, 1.0e-6}) {
        // Extract pending queries
        current_queries.clear();
        current_queries.resize(pending_queries.size());
        for (uint i = 0; i < pending_queries.size(); i++) {
            current_queries[i] = queries[pending_queries[i]];
        }
        
        auto [batch_res, num_dots] = this->search_threshold_batch(current_queries, threshold);
        result.second += num_dots;
        
        // Update results and filter pending queries
        std::vector<uint> still_pending;
        for (uint i = 0; i < batch_res.size(); i++) {
            uint query_idx = pending_queries[i];
            if (batch_res[i].size() >= this->k_val) {
                result.first[query_idx] = batch_res[i];
            } else {
                still_pending.push_back(query_idx);
            }
        }
        pending_queries = still_pending;
        if (pending_queries.empty()) break;
    }
    
    // Sort results by distance and keep top-k for each query
    for (uint i = 0; i < queries.size(); i++) {
        std::vector<std::pair<double, uint>> scored;
        for (uint idx : result.first[i]) {
            scored.push_back({dot_product(queries[i], this->data_set[idx]), idx});
        }
        std::sort(scored.begin(), scored.end(),
                 [](const auto &a, const auto &b) { return a.first > b.first; });
        
        result.first[i].clear();
        for (uint j = 0; j < this->k_val && j < scored.size(); j++) {
            result.first[i].push_back(scored[j].second);
        }
        std::sort(result.first[i].begin(), result.first[i].end());
    }
    
    return result;
}

std::pair<std::vector<uint>, size_t> KNNIndexDataset::search_threshold(SparseVec &query, double threshold) {
    std::vector<std::pair<std::vector<uint>, size_t>> async_results(this->data_index.size());
    std::array<std::thread, NUM_THREADS> threads;
    
    auto worker = [this, &async_results, &query, threshold] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            async_results[pool_index] = this->search_pool(this->data_index[pool_index], query, pool_index, threshold);
        }
    };

    if (use_threading && this->data_index.size() > 1) {
        for (uint i = 0; i < NUM_THREADS; i++) {
            uint start = (i * this->data_index.size()) / NUM_THREADS;
            uint end = ((i + 1) * this->data_index.size()) / NUM_THREADS;
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

std::pair<std::vector<uint>, size_t> KNNIndexDataset::search_pool(KNNIndexSingle<KNN_LEVELS> &pool, SparseVec &query, uint pool_index, double threshold) {
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    if (pool[1].size() == 0) {
        return result;
    }
    std::array<double, KNN_LEVELS + 1> dots;
    std::array<uint, KNN_LEVELS + 1> idxs;
    idxs[0] = 1;
    dots[0] = dot_product(query, pool[1]);
    result.second = 1;
    uint dot_idx = 1;
    uint idx = 1;
    double dot_val = dots[0];
    while (dot_idx > 0) {
        dot_val = dots[dot_idx - 1];
        idx = idxs[dot_idx - 1];
        if (dot_val < threshold) {
            dot_idx--;
            continue;
        }
        if (idx >= static_cast<uint>(1 << KNN_LEVELS)) {
            uint actual_idx = idx - (1 << KNN_LEVELS) + pool_index * (1 << KNN_LEVELS);
            result.first.push_back(actual_idx);
            dot_idx--;
            continue;
        }
        result.second++;
        dot_idx++;
        dots[dot_idx - 1] = dot_product(query, pool[2 * idx + 1]);
        dots[dot_idx - 2] -= dots[dot_idx - 1];
        idxs[dot_idx - 1] = 2 * idx + 1;
        idxs[dot_idx - 2] = 2 * idx;
    }
    return result;
}

std::array<double, 3> KNNIndexDataset::verify_results(SparseVec &query, std::vector<uint> &result) {
    std::vector<std::pair<double, uint>> true_results;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data_set.size(); i++) {
        true_results.push_back({dot_product(query, data_set[i]), i});
    }
    std::sort(true_results.begin(), true_results.end(), std::greater<std::pair<double, uint>>());
    std::vector<uint> true_result;
    for (size_t i = 0; i < this->k_val && i < true_results.size(); i++) {
        true_result.push_back(true_results[i].second);
    }
    if (true_result.size() < this->k_val) {
        true_result.resize(this->k_val, 0);
    }
    std::sort(true_result.begin(), true_result.end());
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double time = duration.count() / 1.0e+3;
    uint match_count = 0;
    uint i = 0;
    uint j = 0;
    while (i < this->k_val && j < this->k_val) {
        if (result[i] == true_result[j]) {
            match_count++;
            i++;
            j++;
        } else if (result[i] < true_result[j]) {
            i++;
        } else {
            j++;
        }
    }
    // For KNN: recall == precision
    double metric = static_cast<double>(match_count) / this->k_val;
    return {time, metric, metric};
}

std::pair<std::vector<std::vector<uint>>, size_t> KNNIndexDataset::search_threshold_batch(
    const SparseMat &queries, double threshold) {
    KNNIndex<INVERTED_LEVELS> query_pools = build_knn_index<INVERTED_LEVELS>(const_cast<SparseMat&>(queries));
    std::vector<std::pair<KNNIndexDoubleGroupResultElem, size_t>> async_results(
        this->data_index.size() * query_pools.size());
    std::array<std::thread, NUM_THREADS> threads;
    
    auto worker = [this, &async_results, &query_pools, threshold] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            for (uint qpool_index = 0; qpool_index < query_pools.size(); qpool_index++) {
                async_results[pool_index * query_pools.size() + qpool_index] = this->search_pool_batch(
                    this->data_index[pool_index], query_pools[qpool_index], pool_index, threshold
                );
            }
        }
    };
    
    if (use_threading) {
        for (uint i = 0; i < NUM_THREADS; i++) {
            uint start = (i * this->data_index.size()) / NUM_THREADS;
            uint end = ((i + 1) * this->data_index.size()) / NUM_THREADS;
            threads[i] = std::thread(worker, start, end);
        }
        for (uint i = 0; i < NUM_THREADS; i++) {
            threads[i].join();
        }
    } else {
        worker(0, this->data_index.size());
    }
    
    std::pair<std::vector<std::vector<uint>>, size_t> result = {std::vector<std::vector<uint>>(queries.size()), 0};
    for (uint pool_index = 0; pool_index < this->data_index.size(); pool_index++) {
        for (uint qpool_index = 0; qpool_index < query_pools.size(); qpool_index++) {
            auto &async_result = async_results[pool_index * query_pools.size() + qpool_index].first;
            result.second += async_results[pool_index * query_pools.size() + qpool_index].second;
            for (uint i = 0; i < async_result.size(); i++) {
                uint qidx = i + qpool_index * (1 << INVERTED_LEVELS);
                if (qidx >= queries.size()) {
                    break;
                }
                for (uint match : async_result[i]) {
                    result.first[qidx].push_back(match);
                }
            }
        }
    }
    return result;
}

std::pair<KNNIndexDoubleGroupResultElem, size_t> KNNIndexDataset::search_pool_batch(
    KNNIndexSingle<KNN_LEVELS> &pool,
    KNNIndexSingle<INVERTED_LEVELS> &qpool,
    uint pool_index,
    double threshold
) {
    
    std::pair<KNNIndexDoubleGroupResultElem, size_t> result = {KNNIndexDoubleGroupResultElem(), 1};
    std::array<double, 2 * (KNN_LEVELS + INVERTED_LEVELS) + 1> dots;
    std::array<uint, 2 * (KNN_LEVELS + INVERTED_LEVELS) + 1> idxs;
    std::array<uint, 2 * (KNN_LEVELS + INVERTED_LEVELS) + 1> qidxs;
    uint dot_idx = 1;
    dots[0] = dot_product(qpool[1], pool[1]);
    idxs[0] = 1;
    qidxs[0] = 1;
    uint idx = 1;
    uint qidx = 1;
    double dot_val = 0;

    while (dot_idx > 0) {
        idx = idxs[dot_idx - 1];
        qidx = qidxs[dot_idx - 1];
        dot_val = dots[dot_idx - 1];

        if (dot_val < threshold) {
            dot_idx--;
            continue;
        }
        if (idx >= static_cast<uint>(1 << KNN_LEVELS)) {
            if (qidx >= static_cast<uint>(1 << INVERTED_LEVELS)) {
                uint data_idx = idx - (1 << KNN_LEVELS) + pool_index * (1 << KNN_LEVELS);
                if (data_idx >= this->data_set.size()) {
                    dot_idx--;
                    continue;
                }
                uint query_idx = qidx - (1 << INVERTED_LEVELS);
                result.first[query_idx].push_back(data_idx);
                dot_idx--;
                continue;
            }
            dot_idx++;
            result.second++;
            dots[dot_idx - 1] = dot_product(qpool[2 * qidx + 1], pool[idx]);
            dots[dot_idx - 2] -= dots[dot_idx - 1];
            
            idxs[dot_idx - 1] = idx;
            idxs[dot_idx - 2] = idx;
            qidxs[dot_idx - 1] = 2 * qidx + 1;
            qidxs[dot_idx - 2] = 2 * qidx;
            continue;
        }
        if (qidx >= static_cast<uint>(1 << INVERTED_LEVELS)) {
            dot_idx++;
            result.second++;
            dots[dot_idx - 1] = dot_product(qpool[qidx], pool[2 * idx + 1]);
            dots[dot_idx - 2] -= dots[dot_idx - 1];
            
            idxs[dot_idx - 1] = 2 * idx + 1;
            idxs[dot_idx - 2] = 2 * idx;
            qidxs[dot_idx - 1] = qidx;
            qidxs[dot_idx - 2] = qidx;
            continue;
        }
        dot_idx += 3;
        result.second += 3;
        dots[dot_idx - 1] = dot_product(qpool[2 * qidx + 1], pool[2 * idx + 1]);
        dots[dot_idx - 2] = dot_product(qpool[2 * qidx + 1], pool[2 * idx]);
        dots[dot_idx - 3] = dot_product(qpool[2 * qidx], pool[2 * idx + 1]);
        dots[dot_idx - 4] -= (dots[dot_idx - 3] + dots[dot_idx - 2] + dots[dot_idx - 1]);

        idxs[dot_idx - 1] = 2 * idx + 1;
        idxs[dot_idx - 2] = 2 * idx;
        idxs[dot_idx - 3] = 2 * idx + 1;
        idxs[dot_idx - 4] = 2 * idx;
        qidxs[dot_idx - 1] = 2 * qidx + 1;
        qidxs[dot_idx - 2] = 2 * qidx + 1;
        qidxs[dot_idx - 3] = 2 * qidx;
        qidxs[dot_idx - 4] = 2 * qidx;
    }
    return result;
}

#undef KNN_LEVELS
#undef INVERTED_LEVELS
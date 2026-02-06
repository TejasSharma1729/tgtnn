#pragma once

#include "sparse_types.hpp"
#include <thread>
#include <chrono>

#define INNER_NUM 8
#define NUM_LEVELS 4
#define OUTER_NUM (1 << NUM_LEVELS)
#define PAGE_SIZE (INNER_NUM * OUTER_NUM)
#define NUM_THREADS 16

struct ThresholdIndexRandomizedElem {
    SparseVec cum_sum;
    size_t base;
    std::array<uint8_t, INNER_NUM> offset;
};

using ThresholdIndexRandomizedSingle = std::array<ThresholdIndexRandomizedElem, OUTER_NUM>;
using ThresholdIndexRandomizedIndex = std::vector<ThresholdIndexRandomizedSingle>;

/**
 * @brief Build randomized index from sparse matrix
 */
ThresholdIndexRandomizedIndex build_threshold_index_randomized(SparseMat& matrix) {
    ThresholdIndexRandomizedIndex index;
    size_t pages = (matrix.size() + PAGE_SIZE - 1) / PAGE_SIZE;
    index.resize(pages);
    
    for (size_t page = 0; page < pages; page++) {
        for (uint outer = 0; outer < OUTER_NUM; outer++) {
            index[page][outer].base = page * PAGE_SIZE;
            index[page][outer].cum_sum.clear();
            
            for (uint inner = 0; inner < INNER_NUM; inner++) {
                size_t idx = page * PAGE_SIZE + outer * INNER_NUM + inner;
                if (idx < matrix.size()) {
                    index[page][outer].offset[inner] = inner;
                    index[page][outer].cum_sum = add_sparse(index[page][outer].cum_sum, matrix[idx]);
                }
            }
        }
    }
    return index;
}

/**
 * @brief Randomized threshold-based search using page-based partitioning
 * Organizes dataset into pages with cumulative vectors for early pruning
 * Single query only
 */
class ThresholdIndexRandomized {
public:
    /**
     * @brief Constructor - builds randomized index from dataset
     * @param dataset Sparse matrix containing data vectors
     * @param threshold Similarity threshold for search (default 0.8)
     */
    ThresholdIndexRandomized(SparseMat &dataset, double threshold = 0.8);
    
    /**
     * @brief Search for vectors above threshold for a single query
     * @param query Query sparse vector
     * @return Pair of (result indices, number of dot products computed)
     */
    std::pair<std::vector<uint>, size_t> search(SparseVec &query);
    
    /**
     * @brief Verify search results against brute-force and compute metrics
     * @param query Query sparse vector
     * @param result Indices returned by search algorithm
     * @return Array of [time_taken_ms, recall, precision]
     */
    std::array<double, 3> verify_results(SparseVec &query, std::vector<uint> &result);

private:
    SparseMat data_set;
    ThresholdIndexRandomizedIndex data_index;
    uint size;
    uint dimention;
    double threshold;
    
    /**
     * @brief Multi-threaded search worker
     */
    void search_thread(SparseVec &query, std::vector<uint> &result, size_t &num_dots, uint thread_id);
    
    /**
     * @brief Search individual page elements
     */
    void individual_search(SparseVec &query, std::vector<uint> &result, size_t &num_dots, 
                          double dot, ThresholdIndexRandomizedElem &pool);
};

ThresholdIndexRandomized::ThresholdIndexRandomized(SparseMat &dataset, double threshold_val)
    : data_set(dataset), threshold(threshold_val) {
    this->dimention = dataset.empty() ? 0 : dataset[0].size();
    this->size = (this->data_set.size() + PAGE_SIZE - 1) / PAGE_SIZE;
    this->data_index = build_threshold_index_randomized(this->data_set);
}

std::pair<std::vector<uint>, size_t> ThresholdIndexRandomized::search(SparseVec &query) {
    std::array<std::thread, NUM_THREADS> threads;
    std::array<std::pair<std::vector<uint>, size_t>, NUM_THREADS> ranges;
    std::pair<std::vector<uint>, size_t> result;
    
    for (uint t = 0; t < NUM_THREADS; t++) {
        ranges[t].first = std::vector<uint>();
        ranges[t].second = 0;
        threads[t] = std::thread(&ThresholdIndexRandomized::search_thread, this, std::ref(query), 
                std::ref(ranges[t].first), std::ref(ranges[t].second), t);
    }
    
    for (uint t = 0; t < NUM_THREADS; t++) {
        threads[t].join();
        result.first.insert(result.first.end(), ranges[t].first.begin(), ranges[t].first.end());
        result.second += ranges[t].second;
    }
    std::sort(result.first.begin(), result.first.end());
    return result;
}

std::array<double, 3> ThresholdIndexRandomized::verify_results(SparseVec &query, std::vector<uint> &result) {
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

void ThresholdIndexRandomized::search_thread(SparseVec &query, std::vector<uint> &result, 
        size_t &num_dots, uint thread_id) {
    for (uint base = thread_id; base < this->size; base += NUM_THREADS) {
        std::array<double, (1 + NUM_LEVELS)> dot_vals;
        std::array<std::pair<uint, uint>, (1 + NUM_LEVELS)> ranges;
        
        dot_vals[0] = dot_product(query, this->data_index[base][0].cum_sum);
        ranges[0] = {0, OUTER_NUM};
        uint pos = 1;
        
        while (pos > 0) {
            if (dot_vals[pos - 1] < this->threshold) {
                pos--;
                continue;
            }
            uint start = ranges[pos - 1].first;
            uint end = ranges[pos - 1].second;
            uint mid = (start + end) / 2;
            
            if (start + 1 == end) {
                this->individual_search(query, result, num_dots, dot_vals[pos - 1], this->data_index[base][mid]);
                pos--;
                continue;
            }
            num_dots++;
            dot_vals[pos] = dot_product(query, this->data_index[base][mid].cum_sum);
            ranges[pos] = {mid, end};
            dot_vals[pos - 1] -= dot_vals[pos];
            ranges[pos - 1] = {start, mid};
            pos++;
        }
    }
}

void ThresholdIndexRandomized::individual_search(SparseVec &query, std::vector<uint> &result, 
                                                  size_t &num_dots, double dot, ThresholdIndexRandomizedElem &pool) {
    double net_dot = 0;
    for (uint i = 1; i < INNER_NUM; i++) {
        num_dots++;
        double dot_i = dot_product(query, this->data_set[pool.base + pool.offset[i]]);
        if (dot_i >= this->threshold) {
            result.push_back(pool.base + pool.offset[i]);
        }
        net_dot += dot_i;
    }
    if (dot - net_dot >= this->threshold) {
        result.push_back(pool.base + pool.offset[0]);
    }
}

#undef INNER_NUM
#undef NUM_LEVELS
#undef OUTER_NUM
#undef PAGE_SIZE
#undef NUM_THREADS

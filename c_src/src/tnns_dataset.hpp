#ifndef TNNS_DATASET_HPP
#define TNNS_DATASET_HPP

#include "header.hpp"
#include <thread>
#include <mutex>
#include <numeric>
#include <map>
#include <set>
#include <iostream>

#define TNNS_SINGLE_NUM_LEVELS 6
#define TNNS_DOUBLE_NUM_LEVELS 7
#define TNNS_DOUBLE_INVERTED_LEVELS 5


class ThresholdIndexDataset {
public:
    using single_elem_t = array<vector_t, (1 << (TNNS_SINGLE_NUM_LEVELS))>;
    using single_index_t = vector<single_elem_t>;

    using double_elem_t = array<vector_t, (1 << (TNNS_DOUBLE_NUM_LEVELS + 1))>; 
    using double_index_t = vector<double_elem_t>;

    using double_inverted_elem_t = array<vector_t, (1 << (TNNS_DOUBLE_INVERTED_LEVELS + 1))>;
    using double_inverted_index_t = vector<double_inverted_elem_t>;
    
    using result_elem_t = array<vector<uint>, (1 << TNNS_DOUBLE_INVERTED_LEVELS)>;

    ThresholdIndexDataset(matrix_t &mat, double threshold = 0.8) : data_set(mat), threshold(threshold) {
        this->single_index = build_single_index(this->data_set);
        this->double_index = build_double_index<TNNS_DOUBLE_NUM_LEVELS, double_index_t>(this->data_set);
    }

    pair<vector<uint>, size_t> search(vector_t &query) {
        vector<pair<vector<uint>, size_t>> async_results(this->single_index.size());
        vector<thread> threads(NUM_THREADS);
        auto worker = [this, &async_results, &query] (uint start, uint end) {
            for (uint i = start; i < end; i++)
                async_results[i] = this->search_pool_single(this->single_index[i], query, i);
        };
        uint batch = this->single_index.size();
        for (uint i = 0; i < NUM_THREADS; i++) {
            uint start = (i * batch) / NUM_THREADS;
            uint end = ((i + 1) * batch) / NUM_THREADS;
            if (i == NUM_THREADS - 1) end = batch;
            threads[i] = thread(worker, start, end);
        }
        for (auto &t : threads) if (t.joinable()) t.join();
        
        pair<vector<uint>, size_t> result = {vector<uint>(), 0};
        for (auto &res : async_results) {
            result.first.insert(result.first.end(), res.first.begin(), res.first.end());
            result.second += res.second;
        }
        return result;
    }

    pair<vector<vector<uint>>, size_t> search_multiple(matrix_t &queries) { 
            double_inverted_index_t query_pools = build_double_index<TNNS_DOUBLE_INVERTED_LEVELS, double_inverted_index_t>(queries);
            vector<pair<result_elem_t, size_t>> async_results(this->double_index.size() * query_pools.size());
            
            vector<thread> threads(NUM_THREADS);
            auto worker = [this, &async_results, &query_pools] (uint start, uint end) {
                for (uint i = start; i < end; i++) {
                    uint pool_idx = i / query_pools.size();
                    uint qpool_idx = i % query_pools.size();
                    async_results[i] = this->search_pool_double(this->double_index[pool_idx], query_pools[qpool_idx], pool_idx);
                }
            };
            
            size_t total = async_results.size();
            for (uint i = 0; i < NUM_THREADS; i++) {
                uint start = (i * total) / NUM_THREADS;
                uint end = ((i + 1) * total) / NUM_THREADS;
                if (i == NUM_THREADS - 1) end = total;
                threads[i] = thread(worker, start, end);
            }
            for (auto &t : threads) if (t.joinable()) t.join();

            pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(queries.rows()), 0};
            for (size_t i = 0; i < total; i++) {
                result.second += async_results[i].second;
                uint qpool_idx = i % query_pools.size();
                for (uint j = 0; j < (1 << TNNS_DOUBLE_INVERTED_LEVELS); j++) {
                    uint qidx = j + qpool_idx * (1 << TNNS_DOUBLE_INVERTED_LEVELS);
                    if (qidx >= queries.rows()) break;
                    for (uint match : async_results[i].first[j]) result.first[qidx].push_back(match);
                }
            }
            return result;
    }

    array<double, 3> verify_results(vector_t &query, vector<uint> &results) {
        vector<uint> true_result;
        auto start = high_resolution_clock::now();
        vector_t dots = this->data_set * query;
        for (long i = 0; i < dots.rows(); i++) {
            if (dots(i) >= this->threshold) true_result.push_back(i);
        }
        auto stop = high_resolution_clock::now();
        double time = duration_cast<microseconds>(stop - start).count() / 1.0e+3;
        
        set<uint> true_set(true_result.begin(), true_result.end());
        set<uint> res_set(results.begin(), results.end());
        set<uint> match_set;
        set_intersection(true_set.begin(), true_set.end(), res_set.begin(), res_set.end(), inserter(match_set, match_set.begin()));
        
        double precision = (res_set.empty()) ? 1.0 : (double)match_set.size() / res_set.size();
        double recall = (true_set.empty()) ? 1.0 : (double)match_set.size() / true_set.size();
        return {time, precision, recall};
    }

private:
    matrix_t data_set;
    double threshold;
    single_index_t single_index;
    double_index_t double_index;

    single_index_t build_single_index(matrix_t &matrix) {
        const uint num_vectors = matrix.rows();
        const uint num_indices = (num_vectors + (1 << TNNS_SINGLE_NUM_LEVELS) - 1) / (1 << TNNS_SINGLE_NUM_LEVELS);
        single_index_t idx(num_indices);
        for (uint i = 0; i < num_indices; i++) {
            const uint offset = i * (1 << TNNS_SINGLE_NUM_LEVELS);
            for (uint j = 0; j < (1 << TNNS_SINGLE_NUM_LEVELS); j++) {
                const uint index = offset + j;
                if (index < num_vectors) idx[i][j] = matrix.row(index);
                else idx[i][j] = vector_t::Zero(matrix.cols());
            }
            for (uint n = 1; n <= TNNS_SINGLE_NUM_LEVELS; n++) {
                const uint step = 1 << n;
                for (uint j = 0; j < (1 << TNNS_SINGLE_NUM_LEVELS); j += step) {
                    idx[i][j] = idx[i][j] + idx[i][j + step / 2];
                }
            }
        }
        return idx;
    }

    template <uint N, typename IndexT>
    IndexT build_double_index(matrix_t &matrix) {
        const uint num_vectors = matrix.rows();
        const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
        IndexT idx(num_indices);
        for (uint i = 0; i < num_indices; i++) {
            const uint offset = i * (1 << N);
            for (uint j = 0; j < (1 << N); j++) {
                const uint index = offset + j;
                if (index < num_vectors) idx[i][(1 << N) + j] = matrix.row(index);
                else idx[i][(1 << N) + j] = vector_t::Zero(matrix.cols());
            }
            for (int n = N - 1; n >= 0; n--) {
                for (uint j = 0; j < static_cast<uint>(1 << n); j++) {
                    const uint index = (1 << n) + j;
                    idx[i][index] = idx[i][2 * index] + idx[i][2 * index + 1];
                }
            }
        }
        return idx;
    }

    pair<vector<uint>, size_t> search_pool_single(single_elem_t &pool, vector_t &query, uint &pool_index) {
        pair<vector<uint>, size_t> result = {vector<uint>(), 0};
        array<double, TNNS_SINGLE_NUM_LEVELS + 1> dots;
        array<pair<uint, uint>, TNNS_SINGLE_NUM_LEVELS + 1> ranges;
        ranges[0] = {0, (1 << TNNS_SINGLE_NUM_LEVELS)};
        dots[0] = query.dot(pool[0]);
        uint dot_idx = 1; 
        while (dot_idx) {
            if (dots[dot_idx - 1] < this->threshold) { dot_idx--; continue; }
            auto [start, end] = ranges[dot_idx - 1];
            uint mid = (start + end) / 2;
            if (start + 1 == end) {
                uint idx = mid + pool_index * (1 << TNNS_SINGLE_NUM_LEVELS);
                if (idx < this->data_set.rows()) result.first.push_back(idx);
                dot_idx--; continue;
            }
            result.second++;
            dots[dot_idx] = query.dot(pool[mid]);
            ranges[dot_idx] = {mid, end};
            dots[dot_idx - 1] -= dots[dot_idx];
            ranges[dot_idx - 1] = {start, mid};
            dot_idx++;
        }
        return result;
    }

    pair<result_elem_t, size_t> search_pool_double(double_elem_t &pool, double_inverted_elem_t &qpool, uint &pool_index) {
        pair<result_elem_t, size_t> result = {result_elem_t(), 1};
        array<double, 2 * (TNNS_DOUBLE_NUM_LEVELS + TNNS_DOUBLE_INVERTED_LEVELS) + 1> dots;
        array<uint, 2 * (TNNS_DOUBLE_NUM_LEVELS + TNNS_DOUBLE_INVERTED_LEVELS) + 1> idxs;
        array<uint, 2 * (TNNS_DOUBLE_NUM_LEVELS + TNNS_DOUBLE_INVERTED_LEVELS) + 1> qidxs;
        uint dot_idx = 1; dots[0] = qpool[1].dot(pool[1]); idxs[0] = 1; qidxs[0] = 1;
        while (dot_idx > 0) {
            double dot_val = dots[dot_idx - 1];
            if (dot_val < this->threshold) { dot_idx--; continue; }
            uint idx = idxs[dot_idx - 1]; uint qidx = qidxs[dot_idx - 1];
            bool idx_leaf = idx >= (1 << TNNS_DOUBLE_NUM_LEVELS);
            bool qidx_leaf = qidx >= (1 << TNNS_DOUBLE_INVERTED_LEVELS);
            if (idx_leaf && qidx_leaf) {
                uint data_idx = idx - (1 << TNNS_DOUBLE_NUM_LEVELS) + pool_index * (1 << TNNS_DOUBLE_NUM_LEVELS);
                if (data_idx < this->data_set.rows()) result.first[qidx - (1 << TNNS_DOUBLE_INVERTED_LEVELS)].push_back(data_idx);
                dot_idx--; continue;
            }
            if (idx_leaf) {
                dot_idx++; result.second++;
                dots[dot_idx - 1] = qpool[2 * qidx + 1].dot(pool[idx]); dots[dot_idx - 2] -= dots[dot_idx - 1];
                idxs[dot_idx - 1] = idx; idxs[dot_idx - 2] = idx;
                qidxs[dot_idx - 1] = 2 * qidx + 1; qidxs[dot_idx - 2] = 2 * qidx; continue;
            }
            if (qidx_leaf) {
                dot_idx++; result.second++;
                dots[dot_idx - 1] = qpool[qidx].dot(pool[2 * idx + 1]); dots[dot_idx - 2] -= dots[dot_idx - 1];
                idxs[dot_idx - 1] = 2 * idx + 1; idxs[dot_idx - 2] = 2 * idx;
                qidxs[dot_idx - 1] = qidx; qidxs[dot_idx - 2] = qidx; continue;
            }
            dot_idx += 3; result.second += 3;
            dots[dot_idx - 1] = qpool[2 * qidx + 1].dot(pool[2 * idx + 1]);
            dots[dot_idx - 2] = qpool[2 * qidx + 1].dot(pool[2 * idx]);
            dots[dot_idx - 3] = qpool[2 * qidx].dot(pool[2 * idx + 1]);
            dots[dot_idx - 4] -= (dots[dot_idx - 3] + dots[dot_idx - 2] + dots[dot_idx - 1]);
            idxs[dot_idx - 1] = 2 * idx + 1; idxs[dot_idx - 2] = 2 * idx; idxs[dot_idx - 3] = 2 * idx + 1; idxs[dot_idx - 4] = 2 * idx;
            qidxs[dot_idx - 1] = 2 * qidx + 1; qidxs[dot_idx - 2] = 2 * qidx + 1; qidxs[dot_idx - 3] = 2 * qidx; qidxs[dot_idx - 4] = 2 * qidx;
        }
        return result;
    }
};

#endif

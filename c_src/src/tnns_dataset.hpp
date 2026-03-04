#ifndef TNNS_DATASET_HPP
#define TNNS_DATASET_HPP

#include "header.hpp"
#include <thread>
#include <mutex>
#include <numeric>
#include <map>
#include <set>
#include <iostream>

#define TNNS_NUM_LEVELS 10
#define TNNS_INVERTED_LEVELS 5


class ThresholdIndexDataset {
public:
    using index_elem_t = array<vector_t, (1 << (TNNS_NUM_LEVELS + 1))>;
    using index_t = vector<index_elem_t>;

    using inverted_elem_t = array<vector_t, (1 << (TNNS_INVERTED_LEVELS + 1))>;
    using inverted_index_t = vector<inverted_elem_t>;
    
    using result_elem_t = array<vector<uint>, (1 << TNNS_INVERTED_LEVELS)>;

    ThresholdIndexDataset(matrix_t &mat, double threshold = 0.8) : data_set(mat), threshold(threshold) {
        this->index = build_index<TNNS_NUM_LEVELS, index_t>(this->data_set);
    }

    void streaming_update(matrix_t &mat) {
        uint old_size = this->data_set.rows();
        this->data_set.conservativeResize(old_size + mat.rows(), Eigen::NoChange);
        for (long i = 0; i < mat.rows(); i++) {
            uint new_idx = old_size + i;
            this->data_set.row(new_idx) = mat.row(i);
            this->update_index_for_point(mat.row(i), new_idx);
        }
    }

    pair<vector<uint>, size_t> search(vector_t &query, bool use_threading = true) {
        if (this->index.empty()) return {vector<uint>(), 0};
        
        vector<pair<vector<uint>, size_t>> results(this->index.size());
        for (uint i = 0; i < this->index.size(); i++) {
            results[i] = this->search_pool_single(this->index[i], query, i);
        }

        pair<vector<uint>, size_t> result = {vector<uint>(), 0};
        for (auto &res : results) {
            result.first.insert(result.first.end(), res.first.begin(), res.first.end());
            result.second += res.second;
        }
        return result;
    }

    pair<vector<vector<uint>>, size_t> search_batch_binary(matrix_t &queries, bool use_threading = true) {
        if (this->index.empty() || queries.rows() == 0) 
            return {vector<vector<uint>>(queries.rows()), 0};
        
        const uint num_queries = queries.rows();
        vector<pair<vector<uint>, size_t>> all_results(num_queries);

        if (!use_threading || num_queries < 2000) {
            for (uint i = 0; i < num_queries; i++) {
                vector_t q = queries.row(i);
                all_results[i] = this->search(q, false);
            }
        } else {
            vector<thread> threads(NUM_THREADS);
            auto worker = [this, &all_results, &queries] (uint start, uint end) {
                for (uint i = start; i < end; i++) {
                    vector_t q = queries.row(i);
                    all_results[i] = this->search(q, false);
                }
            };
            for (uint i = 0; i < NUM_THREADS; i++) {
                uint start = (i * num_queries) / NUM_THREADS;
                uint end = ((i + 1) * num_queries) / NUM_THREADS;
                if (i == NUM_THREADS - 1) end = num_queries;
                threads[i] = thread(worker, start, end);
            }
            for (auto &t : threads) if (t.joinable()) t.join();
        }

        pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(num_queries), 0};
        for (uint i = 0; i < num_queries; i++) {
            result.first[i] = std::move(all_results[i].first);
            result.second += all_results[i].second;
        }
        return result;
    }

    pair<vector<vector<uint>>, size_t> search_multiple(matrix_t &queries, bool use_threading = true) { 
        if (this->index.empty() || queries.rows() == 0) 
            return {vector<vector<uint>>(queries.rows()), 0};
        
        inverted_index_t query_pools = build_index<TNNS_INVERTED_LEVELS, inverted_index_t>(queries);
        const uint num_pools = query_pools.size();
        const uint num_queries = queries.rows();

        if (!use_threading || num_queries < 2000) {
            return this->search_multiple_internal(query_pools, num_queries);
        }

        vector<pair<vector<vector<uint>>, size_t>> async_results(num_pools);
        vector<thread> threads(NUM_THREADS);

        auto worker = [this, &async_results, &query_pools] (uint start, uint end) {
            for (uint i = start; i < end; i++) {
                inverted_index_t single_pool = {query_pools[i]};
                async_results[i] = this->search_multiple_internal(single_pool, 1 << TNNS_INVERTED_LEVELS);
            }
        };

        for (uint i = 0; i < NUM_THREADS; i++) {
            uint start = (i * num_pools) / NUM_THREADS;
            uint end = ((i + 1) * num_pools) / NUM_THREADS;
            if (i == NUM_THREADS - 1) end = num_pools;
            threads[i] = thread(worker, start, end);
        }
        for (auto &t : threads) if (t.joinable()) t.join();

        pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(num_queries), 0};
        for (uint p_idx = 0; p_idx < num_pools; p_idx++) {
            result.second += async_results[p_idx].second;
            for (uint j = 0; j < (1 << TNNS_INVERTED_LEVELS); j++) {
                uint q_idx = (p_idx * (1 << TNNS_INVERTED_LEVELS)) + j;
                if (q_idx >= num_queries) break;
                result.first[q_idx] = std::move(async_results[p_idx].first[j]);
            }
        }
        return result;
    }

private:
    pair<vector<vector<uint>>, size_t> search_multiple_internal(
        const inverted_index_t &query_pools, 
        uint num_queries_to_fill,
        uint pool_offset = 0
    ) {
        pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(num_queries_to_fill), 0};
        
        for (uint tree_idx = 0; tree_idx < this->index.size(); tree_idx++) {
            auto &tree = this->index[tree_idx];
            for (uint qtree_idx = 0; qtree_idx < query_pools.size(); qtree_idx++) {
                auto &qtree = query_pools[qtree_idx];
                auto res = this->search_pool_double(tree, qtree, tree_idx);
                result.second += res.second;
                for (uint j = 0; j < (1 << TNNS_INVERTED_LEVELS); j++) {
                    uint q_idx = qtree_idx * (1 << TNNS_INVERTED_LEVELS) + j;
                    if (q_idx < num_queries_to_fill) {
                        result.first[q_idx].insert(result.first[q_idx].end(), res.first[j].begin(), res.first[j].end());
                    }
                }
            }
        }
        return result;
    }

public:
    array<double, 3> verify_results(vector_t &query, vector<uint> &results) {
        auto start = high_resolution_clock::now();
        vector_t dots = this->data_set * query;
        uint true_count = 0;
        for (long i = 0; i < dots.rows(); i++) {
            if (dots(i) >= this->threshold - 1e-9) true_count++;
        }
        auto stop = high_resolution_clock::now();
        double time = duration_cast<microseconds>(stop - start).count() / 1.0e+3;
        
        uint match_count = 0;
        for (uint idx : results) {
            if (dots(idx) >= this->threshold - 1e-9) match_count++;
        }
        
        double precision = (results.empty()) ? 1.0 : (double)match_count / results.size();
        double recall = (true_count == 0) ? 1.0 : (double)match_count / true_count;
        return {time, min(1.0, precision), min(1.0, recall)};
    }

private:
    matrix_t data_set;
    double threshold;
    index_t index;

    template <uint N, typename IndexT>
    IndexT build_index(const matrix_t &matrix) {
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

    void update_index_for_point(const vector_t &new_point, uint new_idx) {
        // Add to index
        uint tree_idx = new_idx / (1 << TNNS_NUM_LEVELS);
        uint node_idx = (new_idx % (1 << TNNS_NUM_LEVELS)) + (1 << TNNS_NUM_LEVELS);
        if (tree_idx >= this->index.size()) {
            index_elem_t new_tree;
            for (uint j = 0; j < (1 << TNNS_NUM_LEVELS); j++) {
                const uint index = tree_idx * (1 << TNNS_NUM_LEVELS) + j;
                if (index < this->data_set.rows()) {
                    new_tree[(1 << TNNS_NUM_LEVELS) + j] = this->data_set.row(index);
                } else {
                    new_tree[(1 << TNNS_NUM_LEVELS) + j] = vector_t::Zero(this->data_set.cols());
                }
            }
            for (int n = TNNS_NUM_LEVELS - 1; n >= 0; n--) {
                for (uint j = 0; j < static_cast<uint>(1 << n); j++) {
                    const uint index = (1 << n) + j;
                    new_tree[index] = new_tree[2 * index] + new_tree[2 * index + 1];
                }
            }
            this->index.push_back(new_tree);
        } else {
            this->index[tree_idx][node_idx] = new_point;
            for (int n = node_idx / 2; n >= 1; n /= 2) {
                this->index[tree_idx][n] = this->index[tree_idx][2 * n] + this->index[tree_idx][2 * n + 1];
            }
        }
    }

    pair<vector<uint>, size_t> search_pool_single(index_elem_t &pool, vector_t &query, uint &pool_index) {
        pair<vector<uint>, size_t> result = {vector<uint>(), 0};
        array<double, TNNS_NUM_LEVELS + 1> dots;
        array<uint, TNNS_NUM_LEVELS + 1> idxs;
        
        uint dot_idx = 1; 
        dots[0] = query.dot(pool[1]);
        idxs[0] = 1;
        
        while (dot_idx > 0) {
            double dot_val = dots[dot_idx - 1];
            if (dot_val < this->threshold - 1e-9) { dot_idx--; continue; }
            uint idx = idxs[dot_idx - 1];
            
            if (idx >= (1 << TNNS_NUM_LEVELS)) {
                uint data_idx = idx - (1 << TNNS_NUM_LEVELS) + pool_index * (1 << TNNS_NUM_LEVELS);
                if (data_idx < this->data_set.rows()) result.first.push_back(data_idx);
                dot_idx--; continue;
            }
            
            result.second++;
            double right_dot = query.dot(pool[2 * idx + 1]);
            double left_dot = dot_val - right_dot;
            
            dots[dot_idx - 1] = left_dot;
            idxs[dot_idx - 1] = 2 * idx;
            
            dots[dot_idx] = right_dot;
            idxs[dot_idx] = 2 * idx + 1;
            
            dot_idx++;
        }
        return result;
    }

    pair<result_elem_t, size_t> search_pool_double(const index_elem_t &pool, const inverted_elem_t &qpool, uint pool_index) {
        pair<result_elem_t, size_t> result = {result_elem_t(), 1};
        array<double, 2 * (TNNS_NUM_LEVELS + TNNS_INVERTED_LEVELS) + 1> dots;
        array<uint, 2 * (TNNS_NUM_LEVELS + TNNS_INVERTED_LEVELS) + 1> idxs;
        array<uint, 2 * (TNNS_NUM_LEVELS + TNNS_INVERTED_LEVELS) + 1> qidxs;
        uint dot_idx = 1; dots[0] = qpool[1].dot(pool[1]); idxs[0] = 1; qidxs[0] = 1;
        while (dot_idx > 0) {
            double dot_val = dots[dot_idx - 1];
            if (dot_val < this->threshold - 1e-9) { dot_idx--; continue; }
            uint idx = idxs[dot_idx - 1]; uint qidx = qidxs[dot_idx - 1];
            bool idx_leaf = idx >= (1 << TNNS_NUM_LEVELS);
            bool qidx_leaf = qidx >= (1 << TNNS_INVERTED_LEVELS);
            if (idx_leaf && qidx_leaf) {
                uint data_idx = idx - (1 << TNNS_NUM_LEVELS) + pool_index * (1 << TNNS_NUM_LEVELS);
                if (data_idx < this->data_set.rows()) result.first[qidx - (1 << TNNS_INVERTED_LEVELS)].push_back(data_idx);
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

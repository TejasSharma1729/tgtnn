#ifndef KNNS_DATASET_HPP
#define KNNS_DATASET_HPP

#include "header.hpp"
#include <thread>
#include <mutex>
#include <numeric>
#include <map>
#include <iostream>

#define KNNS_TREE_LEVELS 10
#define KNNS_INVERTED_LEVELS 5

struct KNNSNode {
    double dot_val;
    uint tree_idx;
    uint node_idx;

    bool operator == (const KNNSNode &other) const {
        return tie(dot_val, tree_idx, node_idx) == tie(other.dot_val, other.tree_idx, other.node_idx);
    }
    bool operator < (const KNNSNode &other) const {
        return tie(dot_val, tree_idx, node_idx) < tie(other.dot_val, other.tree_idx, other.node_idx);
    } 
    bool operator > (const KNNSNode &other) const {
        return tie(dot_val, tree_idx, node_idx) > tie(other.dot_val, other.tree_idx, other.node_idx);
    }
};
using KNNSSearchIndex = priority_queue<KNNSNode, vector<KNNSNode>, less<KNNSNode>>;

struct KNNSDoubleNode {
    double dot_val;
    uint tree_idx;
    uint node_idx;
    uint qtree_idx;
    uint qnode_idx;

    bool operator == (const KNNSDoubleNode &other) const {
        return tie(dot_val, tree_idx, node_idx, qtree_idx, qnode_idx) == 
               tie(other.dot_val, other.tree_idx, other.node_idx, other.qtree_idx, other.qnode_idx);
    }
    bool operator < (const KNNSDoubleNode &other) const {
        return tie(dot_val, tree_idx, node_idx, qtree_idx, qnode_idx) < 
               tie(other.dot_val, other.tree_idx, other.node_idx, other.qtree_idx, other.qnode_idx);
    }
    bool operator > (const KNNSDoubleNode &other) const {
        return tie(dot_val, tree_idx, node_idx, qtree_idx, qnode_idx) > 
               tie(other.dot_val, other.tree_idx, other.node_idx, other.qtree_idx, other.qnode_idx);
    }
};
using KNNSDoubleSearchIndex = priority_queue<KNNSDoubleNode, vector<KNNSDoubleNode>, less<KNNSDoubleNode>>;

class KNNSIndexDataset {
public:
    using knns_elem_t = array<vector_t, (1 << (KNNS_TREE_LEVELS + 1))>;
    using knns_index_t = vector<knns_elem_t>;

    using knns_inverted_elem_t = array<vector_t, (1 << (KNNS_INVERTED_LEVELS + 1))>;
    using knns_inverted_index_t = vector<knns_inverted_elem_t>;

    using knnsd_result_elem_t = array<vector<uint>, (1 << KNNS_INVERTED_LEVELS)>;

    KNNSIndexDataset(matrix_t &mat, size_t k = 10) : data_set(mat), k_val(k) {
        Eigen::setNbThreads(1);
        this->index = build_index<KNNS_TREE_LEVELS, knns_index_t>(data_set);
    }

    void streaming_update(const matrix_t &mat) {
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

        const uint num_trees = this->index.size();
        // If we have many trees and use_threading is true, parallelize across trees
        if (use_threading && num_trees >= NUM_THREADS) {
            vector<pair<vector<pair<double, uint>>, size_t>> async_results(num_trees);
            vector<thread> threads(NUM_THREADS);
            auto worker = [this, &async_results, &query] (uint start, uint end) {
                for (uint i = start; i < end; i++)
                    async_results[i] = this->search_pool_single(this->index[i], query, i);
            };

            for (uint i = 0; i < NUM_THREADS; i++) {
                uint start = (i * num_trees) / NUM_THREADS;
                uint end = ((i + 1) * num_trees) / NUM_THREADS;
                if (i == NUM_THREADS - 1) end = num_trees;
                threads[i] = thread(worker, start, end);
            }
            for (auto &t : threads) if (t.joinable()) t.join();

            vector<pair<double, uint>> all_candidates;
            size_t total_dots = 0;
            for (auto &res : async_results) {
                all_candidates.insert(all_candidates.end(), res.first.begin(), res.first.end());
                total_dots += res.second;
            }
            sort(all_candidates.begin(), all_candidates.end(), greater<pair<double, uint>>());
            
            vector<uint> final_results;
            for (uint i = 0; i < min((size_t)all_candidates.size(), this->k_val); i++) {
                final_results.push_back(all_candidates[i].second);
            }
            sort(final_results.begin(), final_results.end());
            return {final_results, total_dots};
        }

        // Original serial search
        pair<vector<uint>, size_t> result = {vector<uint>(), 0};
        KNNSSearchIndex search_index;
        for (uint tree_idx = 0; tree_idx < num_trees; tree_idx++) {
            const auto &tree = this->index[tree_idx];
            double dot_val = query.dot(tree[1]);
            search_index.push({dot_val, tree_idx, 1});
            result.second++;
        }

        while (!search_index.empty()) {
            auto [dot_val, tree_idx, node_idx] = search_index.top();
            
            // If we already have k neighbors and this branch's maximum possible recall 
            // is less than our worst neighbor found so far, we can safely stop.
            if (result.first.size() >= this->k_val) {
                double min_found_dot = query.dot(this->data_set.row(result.first.back()));
                if (dot_val < min_found_dot - 1e-9) break;
            }
            
            search_index.pop();
            const auto &tree = this->index[tree_idx];
            if (node_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS)) {
                result.first.push_back(
                    node_idx - (1 << KNNS_TREE_LEVELS) + tree_idx * (1 << KNNS_TREE_LEVELS)
                );
                // Keep the results sorted by dot product for efficient bound checking
                sort(result.first.begin(), result.first.end(), [&](uint a, uint b) {
                    return query.dot(this->data_set.row(a)) > query.dot(this->data_set.row(b));
                });
                if (result.first.size() > this->k_val) result.first.pop_back();
                continue;
            }
            uint left_child = 2 * node_idx;
            uint right_child = 2 * node_idx + 1;
            double left_dot = query.dot(tree[left_child]);
            double right_dot = query.dot(tree[right_child]);

            search_index.push({left_dot, tree_idx, left_child});
            search_index.push({right_dot, tree_idx, right_child});
            result.second += 2;
        }
        sort(result.first.begin(), result.first.end());
        return result;
    }

    pair<vector<vector<uint>>, size_t> search_batch_binary(matrix_t &query_set, bool use_threading = true) {
        if (this->index.empty() || query_set.rows() == 0) 
            return {vector<vector<uint>>(query_set.rows()), 0};
        
        const uint num_queries = query_set.rows();
        const uint num_trees = this->index.size();
        vector<pair<vector<uint>, size_t>> all_results(num_queries);

        // If batch is large, parallelize across queries (Batch-Parallel)
        // Adjust NUM_THREADS based on workload to reduce overhead
        uint work_threads = NUM_THREADS;
        if (num_queries < NUM_THREADS * 5) work_threads = max(1u, num_queries / 8);

        if (use_threading && work_threads > 1) {
            vector<thread> threads;
            auto worker = [this, &all_results, &query_set] (uint start, uint end) {
                for (uint i = start; i < end; i++) {
                    vector_t q = query_set.row(i);
                    all_results[i] = this->search(q, false);
                }
            };
            for (uint i = 0; i < work_threads; i++) {
                uint start = (i * num_queries) / work_threads;
                uint end = ((i + 1) * num_queries) / work_threads;
                if (i == work_threads - 1) end = num_queries;
                if (start < end) threads.emplace_back(worker, start, end);
            }
            for (auto &t : threads) t.join();
        } 
        else {
            for (uint i = 0; i < num_queries; i++) {
                vector_t q = query_set.row(i);
                all_results[i] = this->search(q, use_threading);
            }
        }

        pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(num_queries), 0};
        for (uint i = 0; i < num_queries; i++) {
            result.first[i] = std::move(all_results[i].first);
            result.second += all_results[i].second;
        }
        return result;
    }

    pair<vector<vector<uint>>, size_t> search_multiple(matrix_t &query_set, bool use_threading = true) {
        if (this->index.empty() || query_set.rows() == 0) 
            return {vector<vector<uint>>(query_set.rows()), 0};
        const uint num_queries = query_set.rows();
        
        // Strategy Decision: Double GT is more efficient than Binary only for large query batches.
        // For small batches (Q < 256), just use the faster Binary algorithm.
        // if (num_queries < 256) {
        //     return this->search_batch_binary(query_set, use_threading);
        // }

        knns_inverted_index_t query_pools = build_index<KNNS_INVERTED_LEVELS, knns_inverted_index_t>(query_set);
        const uint num_pools = query_pools.size();

        // Serial mode
        if (!use_threading || num_pools < 4) {
             return this->search_multiple_internal(query_pools, num_queries);
        }

        // Parallel mode (Strategy A: Many query pools)
        vector<pair<vector<vector<uint>>, size_t>> async_results(num_pools);
        vector<thread> threads;
        uint work_threads = min((uint)NUM_THREADS, (uint)num_pools);
        
        auto worker = [this, &async_results, &query_pools] (uint start_p, uint end_p) {
            for (uint p = start_p; p < end_p; p++) {
                knns_inverted_index_t single_pool = {query_pools[p]};
                async_results[p] = this->search_multiple_internal(single_pool, 1 << KNNS_INVERTED_LEVELS);
            }
        };

        for (uint i = 0; i < work_threads; i++) {
            uint start = (i * num_pools) / work_threads;
            uint end = ((i + 1) * num_pools) / work_threads;
            if (i == work_threads - 1) end = num_pools;
            if (start < end) threads.emplace_back(worker, start, end);
        }
        for (auto &t : threads) if (t.joinable()) t.join();

        pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(num_queries), 0};
        for (uint p = 0; p < num_pools; p++) {
            result.second += async_results[p].second;
            for (uint j = 0; j < (1 << KNNS_INVERTED_LEVELS); j++) {
                uint q_idx = p * (1 << KNNS_INVERTED_LEVELS) + j;
                if (q_idx < num_queries) result.first[q_idx] = std::move(async_results[p].first[j]);
            }
        }
        return result;
    }

private:
    pair<vector<vector<uint>>, size_t> search_multiple_internal(
        const knns_inverted_index_t &query_pools, 
        uint num_queries_to_fill,
        uint pool_offset = 0 // Offset is now only for leaf index calculation if needed
    ) {
        pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(num_queries_to_fill), 0};
        
        KNNSDoubleSearchIndex search_index;
        for (uint tree_idx = 0; tree_idx < this->index.size(); tree_idx++) {
            const auto &tree = this->index[tree_idx];
            for (uint qtree_idx = 0; qtree_idx < query_pools.size(); qtree_idx++) {
                const auto &qtree = query_pools[qtree_idx];
                double dot_val = qtree[1].dot(tree[1]);
                search_index.push({dot_val, tree_idx, 1, qtree_idx, 1});
                result.second++;
            }
        }
        
        uint finished_queries = 0;
        uint active_queries_count = num_queries_to_fill; // Simplified threshold

        while (finished_queries < active_queries_count && !search_index.empty()) {
            auto [dot_val, tree_idx, node_idx, qtree_idx, qnode_idx] = search_index.top();
            search_index.pop();
            const auto &tree = this->index[tree_idx];
            const auto &qtree = query_pools[qtree_idx];
            
            if (node_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS) && 
                qnode_idx >= static_cast<uint>(1 << KNNS_INVERTED_LEVELS)) {
                uint q_idx = qtree_idx * (1 << KNNS_INVERTED_LEVELS) + (qnode_idx - (1 << KNNS_INVERTED_LEVELS));
                uint data_idx = node_idx - (1 << KNNS_TREE_LEVELS) + tree_idx * (1 << KNNS_TREE_LEVELS);
                if (q_idx < num_queries_to_fill && result.first[q_idx].size() < this->k_val) {
                    result.first[q_idx].push_back(data_idx);
                    if (result.first[q_idx].size() == this->k_val) finished_queries++;
                }
                continue;
            }
            if (node_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS)) {
                uint left_qchild = 2 * qnode_idx;
                uint right_qchild = 2 * qnode_idx + 1;
                double left_dot = qtree[left_qchild].dot(tree[node_idx]);
                double right_dot = qtree[right_qchild].dot(tree[node_idx]);
                search_index.push({left_dot, tree_idx, node_idx, qtree_idx, left_qchild});
                search_index.push({right_dot, tree_idx, node_idx, qtree_idx, right_qchild});
                result.second += 2;
            } else if (qnode_idx >= static_cast<uint>(1 << KNNS_INVERTED_LEVELS)) {
                uint q_idx = qtree_idx * (1 << KNNS_INVERTED_LEVELS) + (qnode_idx - (1 << KNNS_INVERTED_LEVELS));
                if (q_idx < num_queries_to_fill && result.first[q_idx].size() >= this->k_val) continue;
                uint left_child = 2 * node_idx;
                uint right_child = 2 * node_idx + 1;
                double left_dot = qtree[qnode_idx].dot(tree[left_child]);
                double right_dot = qtree[qnode_idx].dot(tree[right_child]);
                search_index.push({left_dot, tree_idx, left_child, qtree_idx, qnode_idx});
                search_index.push({right_dot, tree_idx, right_child, qtree_idx, qnode_idx});
                result.second += 2;
            } else {
                uint left_child = 2 * node_idx; uint right_child = 2 * node_idx + 1;
                uint left_qchild = 2 * qnode_idx; uint right_qchild = 2 * qnode_idx + 1;
                double ll_dot = qtree[left_qchild].dot(tree[left_child]);
                double lr_dot = qtree[right_qchild].dot(tree[left_child]);
                double rl_dot = qtree[left_qchild].dot(tree[right_child]);
                double rr_dot = qtree[right_qchild].dot(tree[right_child]);
                search_index.push({ll_dot, tree_idx, left_child, qtree_idx, left_qchild});
                search_index.push({lr_dot, tree_idx, left_child, qtree_idx, right_qchild});
                search_index.push({rl_dot, tree_idx, right_child, qtree_idx, left_qchild});
                search_index.push({rr_dot, tree_idx, right_child, qtree_idx, right_qchild});
                result.second += 4;
            }
        }
        for (uint i = 0; i < num_queries_to_fill; i++) {
            if (!result.first[i].empty()) sort(result.first[i].begin(), result.first[i].end());
        }
        return result;
    }

public:
    array<double, 2> verify_results(vector_t query, vector<uint> &result) {
        vector<pair<double, uint>> true_results;
        auto start = high_resolution_clock::now();
        for (long i = 0; i < data_set.rows(); i++) {
            true_results.push_back({query.dot(data_set.row(i)), (uint)i});
        }
        sort(true_results.begin(), true_results.end(), greater<pair<double, uint>>());
        
        double min_true_dot = -1e18;
        if (!true_results.empty() && this->k_val > 0) {
            size_t effective_k = min(this->k_val, (size_t)true_results.size());
            min_true_dot = true_results[effective_k - 1].first;
        }

        auto stop = high_resolution_clock::now();
        double time = duration_cast<microseconds>(stop - start).count() / 1.0e+3;

        uint match_count = 0;
        for (uint idx : result) {
            if (query.dot(data_set.row(idx)) >= min_true_dot - 1e-9) {
                match_count++;
            }
        }
        
        double recall = (this->k_val == 0) ? 1.0 : (double)match_count / this->k_val;
        return {time, min(1.0, recall)};
    }

private:
    matrix_t data_set;
    size_t k_val;
    knns_index_t index;

    pair<vector<pair<double, uint>>, size_t> search_pool_single(
        const knns_elem_t &tree, 
        const vector_t &query, 
        uint tree_idx
    ) {
        pair<vector<pair<double, uint>>, size_t> result = {vector<pair<double, uint>>(), 0};
        KNNSSearchIndex search_pool_index;
        search_pool_index.push({query.dot(tree[1]), tree_idx, 1});
        result.second++;
        while (result.first.size() < this->k_val && !search_pool_index.empty()) {
            auto [dot_val, t_idx, n_idx] = search_pool_index.top(); search_pool_index.pop();
            if (n_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS)) {
                result.first.push_back({dot_val, n_idx - (1 << KNNS_TREE_LEVELS) + tree_idx * (1 << KNNS_TREE_LEVELS)});
                continue;
            }
            uint left = 2 * n_idx; uint right = 2 * n_idx + 1;
            double l_dot = query.dot(tree[left]); double r_dot = query.dot(tree[right]);
            search_pool_index.push({l_dot, t_idx, left}); search_pool_index.push({r_dot, t_idx, right});
            result.second += 2;
        }
        return result;
    }

    pair<vector<vector<pair<double, uint>>>, size_t> search_pool_double(
        const knns_elem_t &tree, 
        const knns_inverted_elem_t &qtree, 
        uint tree_idx
    ) {
        vector<vector<pair<double, uint>>> pool_results(1 << KNNS_INVERTED_LEVELS);
        size_t dot_products = 0;
        KNNSDoubleSearchIndex search_index;
        search_index.push({qtree[1].dot(tree[1]), tree_idx, 1, 1, 1});
        dot_products++;
        uint finished_queries = 0;
        const uint num_local_queries = (1 << KNNS_INVERTED_LEVELS);
        while (finished_queries < num_local_queries && !search_index.empty()) {
            auto [dot_val, t_idx, n_idx, qt_idx, qn_idx] = search_index.top(); search_index.pop();
            if (n_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS) && qn_idx >= static_cast<uint>(1 << KNNS_INVERTED_LEVELS)) {
                uint local_qidx = qn_idx - (1 << KNNS_INVERTED_LEVELS);
                if (pool_results[local_qidx].size() < this->k_val) {
                    pool_results[local_qidx].push_back({dot_val, n_idx - (1 << KNNS_TREE_LEVELS) + tree_idx * (1 << KNNS_TREE_LEVELS)});
                    if (pool_results[local_qidx].size() == this->k_val) finished_queries++;
                }
                continue;
            }
            if (n_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS)) {
                uint left_qchild = 2 * qn_idx; uint right_qchild = 2 * qn_idx + 1;
                double left_dot = qtree[left_qchild].dot(tree[n_idx]); double right_dot = qtree[right_qchild].dot(tree[n_idx]);
                search_index.push({left_dot, t_idx, n_idx, 1, left_qchild}); search_index.push({right_dot, t_idx, n_idx, 1, right_qchild});
                dot_products += 2;
            } else if (qn_idx >= static_cast<uint>(1 << KNNS_INVERTED_LEVELS)) {
                uint local_qidx = qn_idx - (1 << KNNS_INVERTED_LEVELS);
                if (pool_results[local_qidx].size() >= this->k_val) continue;
                uint left_child = 2 * n_idx; uint right_child = 2 * n_idx + 1;
                double left_dot = qtree[qn_idx].dot(tree[left_child]); double right_dot = qtree[qn_idx].dot(tree[right_child]);
                search_index.push({left_dot, t_idx, left_child, 1, qn_idx}); search_index.push({right_dot, t_idx, right_child, 1, qn_idx});
                dot_products += 2;
            } else {
                uint left_child = 2 * n_idx; uint right_child = 2 * n_idx + 1;
                uint left_qchild = 2 * qn_idx; uint right_qchild = 2 * qn_idx + 1;
                double ll_dot = qtree[left_qchild].dot(tree[left_child]); double lr_dot = qtree[right_qchild].dot(tree[left_child]);
                double rl_dot = qtree[left_qchild].dot(tree[right_child]); double rr_dot = qtree[right_qchild].dot(tree[right_child]);
                search_index.push({ll_dot, t_idx, left_child, 1, left_qchild}); search_index.push({lr_dot, t_idx, left_child, 1, right_qchild});
                search_index.push({rl_dot, t_idx, right_child, 1, left_qchild}); search_index.push({rr_dot, t_idx, right_child, 1, right_qchild});
                dot_products += 4;
            }
        }
        return {pool_results, dot_products};
    }

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
        uint tree_idx = new_idx / (1 << KNNS_TREE_LEVELS);
        uint node_idx = (new_idx % (1 << KNNS_TREE_LEVELS)) + (1 << KNNS_TREE_LEVELS);
        if (tree_idx >= this->index.size()) {
            // Need to add a new tree
            knns_elem_t new_tree;
            for (uint j = 0; j < (1 << KNNS_TREE_LEVELS); j++) {
                const uint index = tree_idx * (1 << KNNS_TREE_LEVELS) + j;
                if (index < this->data_set.rows()) {
                    new_tree[(1 << KNNS_TREE_LEVELS) + j] = this->data_set.row(index);
                } else {
                    new_tree[(1 << KNNS_TREE_LEVELS) + j] = vector_t::Zero(this->data_set.cols());
                }
            }
            for (int n = KNNS_TREE_LEVELS - 1; n >= 0; n--) {
                for (uint j = 0; j < static_cast<uint>(1 << n); j++) {
                    const uint index = (1 << n) + j;
                    new_tree[index] = new_tree[2 * index] + new_tree[2 * index + 1];
                }
            }
            this->index.push_back(new_tree);
        } else {
            // Just need to update existing tree
            auto &tree = this->index[tree_idx];
            tree[node_idx] = new_point; // Fixed: replaced += with =
            for (int n = node_idx / 2; n >= 1; n /= 2) {
                tree[n] = tree[2 * n] + tree[2 * n + 1];
            }
        }
    }
};

#endif

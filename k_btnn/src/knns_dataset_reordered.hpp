#ifndef DC39D49B_AD0F_4907_AA56_3302836D7745
#define DC39D49B_AD0F_4907_AA56_3302836D7745

#include "header.hpp"
#include <thread>
#include <mutex>
#include <numeric>
#include <map>
#include <iostream>
#include <queue>
#include <algorithm>
#include <limits>
#include <atomic>

#ifndef KNNS_TREE_LEVELS
#define KNNS_TREE_LEVELS 10
#endif
#ifndef KNNS_INVERTED_LEVELS
#define KNNS_INVERTED_LEVELS 5
#endif

struct KNNSReorderedNode {
    double dot_val;
    uint tree_idx;
    uint node_idx;

    bool operator == (const KNNSReorderedNode &other) const {
        return tie(dot_val, tree_idx, node_idx) == tie(other.dot_val, other.tree_idx, other.node_idx);
    }
    bool operator < (const KNNSReorderedNode &other) const {
        return tie(dot_val, tree_idx, node_idx) < tie(other.dot_val, other.tree_idx, other.node_idx);
    } 
    bool operator > (const KNNSReorderedNode &other) const {
        return tie(dot_val, tree_idx, node_idx) > tie(other.dot_val, other.tree_idx, other.node_idx);
    }
};
using KNNSSearchReorderedIndex = priority_queue<KNNSReorderedNode, vector<KNNSReorderedNode>, less<KNNSReorderedNode>>;

struct KNNSReorderedDoubleNode {
    double dot_val;
    uint tree_idx;
    uint node_idx;
    uint qtree_idx;
    uint qnode_idx;

    bool operator == (const KNNSReorderedDoubleNode &other) const {
        return tie(dot_val, tree_idx, node_idx, qtree_idx, qnode_idx) == 
               tie(other.dot_val, other.tree_idx, other.node_idx, other.qtree_idx, other.qnode_idx);
    }
    bool operator < (const KNNSReorderedDoubleNode &other) const {
        return tie(dot_val, tree_idx, node_idx, qtree_idx, qnode_idx) < 
               tie(other.dot_val, other.tree_idx, other.node_idx, other.qtree_idx, other.qnode_idx);
    }
    bool operator > (const KNNSReorderedDoubleNode &other) const {
        return tie(dot_val, tree_idx, node_idx, qtree_idx, qnode_idx) > 
               tie(other.dot_val, other.tree_idx, other.node_idx, other.qtree_idx, other.qnode_idx);
    }
};
using KNNSReorderedDoubleSearchIndex = priority_queue<KNNSReorderedDoubleNode, vector<KNNSReorderedDoubleNode>, less<KNNSReorderedDoubleNode>>;


template <uint L> using knns_reordered_elem_t = array<vector_t, (1 << (L + 1))>;
template <uint L> struct knns_reordered_index_t {
    vector<knns_reordered_elem_t<L>> trees;
    vector<array<uint, (1 << L)>> reorder_maps; // Maps from relative leaf index to original absolute index

    knns_reordered_index_t() {}
    knns_reordered_index_t(const matrix_t &matrix) {
        const uint num_vectors = matrix.rows();
        const uint K = (1 << L);
        const uint num_indices = (num_vectors + K - 1) / K;
        this->trees.resize(num_indices);
        this->reorder_maps.resize(num_indices);
        
        vector<uint> order(num_vectors);
        iota(order.begin(), order.end(), 0);
        
        vector<uint> argMaxIndices(num_vectors);
        for (uint i = 0; i < num_vectors; i++) {
            Eigen::Index argmax;
            matrix.row(i).maxCoeff(&argmax);
            argMaxIndices[i] = static_cast<uint>(argmax);
        }
        
        std::sort(order.begin(), order.end(), [&argMaxIndices](uint a, uint b) {
            return argMaxIndices[a] < argMaxIndices[b];
        });

        for (uint i = 0; i < num_indices * K; i++) {
            uint tree_idx = i / K;
            uint rel_idx = i % K;
            if (i < num_vectors) {
                this->trees[tree_idx][K + rel_idx] = matrix.row(order[i]);
                this->reorder_maps[tree_idx][rel_idx] = order[i];
            } else {
                this->trees[tree_idx][K + rel_idx] = vector_t::Zero(matrix.cols());
                this->reorder_maps[tree_idx][rel_idx] = std::numeric_limits<uint>::max();
            }
        }

        // Build tree summaries
        for (uint i = 0; i < num_indices; i++) {
            for (int n = L - 1; n >= 0; n--) {
                for (uint j = 0; j < static_cast<uint>(1 << n); j++) {
                    const uint index = (1 << n) + j;
                    this->trees[i][index] = this->trees[i][2 * index] + this->trees[i][2 * index + 1];
                }
            }
        }
    }
};

class KNNReorderedIndexDataset {
public:
    using knns_elem_t = knns_reordered_elem_t<KNNS_TREE_LEVELS>;
    using knns_index_t = knns_reordered_index_t<KNNS_TREE_LEVELS>;

    using knns_inverted_index_t = knns_reordered_index_t<KNNS_INVERTED_LEVELS>;

    KNNReorderedIndexDataset(matrix_t &mat, size_t k = 10) : data_set(mat), k_val(k), index(mat) {
        Eigen::setNbThreads(1);
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
        if (this->index.trees.empty()) return {vector<uint>(), 0};

        const uint num_trees = this->index.trees.size();
        if (use_threading && num_trees >= NUM_THREADS) {
            vector<pair<vector<pair<double, uint>>, size_t>> async_results(num_trees);
            vector<thread> threads(NUM_THREADS);
            auto worker = [this, &async_results, &query] (uint start, uint end) {
                for (uint i = start; i < end; i++)
                    async_results[i] = this->search_pool_single(this->index.trees[i], this->index.reorder_maps[i], query, i);
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

        // Serial Best-First Search (Global Priority Queue across all trees)
        pair<vector<uint>, size_t> result = {vector<uint>(), 0};
        KNNSSearchReorderedIndex search_index;
        for (uint tree_idx = 0; tree_idx < num_trees; tree_idx++) {
            const auto &tree = this->index.trees[tree_idx];
            double dot_val = query.dot(tree[1]);
            search_index.push({dot_val, tree_idx, 1});
            result.second++;
        }

        while (!search_index.empty()) {
            auto [dot_val, tree_idx, node_idx] = search_index.top();
            if (result.first.size() >= this->k_val) {
                double min_found_dot = query.dot(this->data_set.row(result.first.back()));
                if (dot_val < min_found_dot - 1e-9) break;
            }
            search_index.pop();
            const auto &tree = this->index.trees[tree_idx];
            const auto &rmap = this->index.reorder_maps[tree_idx];
            if (node_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS)) {
                uint abs_idx = rmap[node_idx - (1 << KNNS_TREE_LEVELS)];
                if (abs_idx != std::numeric_limits<uint>::max()) {
                    result.first.push_back(abs_idx);
                    sort(result.first.begin(), result.first.end(), [&](uint a, uint b) {
                        return query.dot(this->data_set.row(a)) > query.dot(this->data_set.row(b));
                    });
                    if (result.first.size() > this->k_val) result.first.pop_back();
                }
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
        if (this->index.trees.empty() || query_set.rows() == 0) 
            return {vector<vector<uint>>(query_set.rows()), 0};
        
        const uint num_queries = query_set.rows();
        vector<pair<vector<uint>, size_t>> all_results(num_queries);

        if (use_threading) {
            #pragma omp parallel for num_threads(NUM_THREADS)
            for (int i = 0; i < (int)num_queries; i++) {
                vector_t q = query_set.row(i);
                all_results[i] = this->search(q, false); 
            }
        } 
        else {
            for (uint i = 0; i < num_queries; i++) {
                vector_t q = query_set.row(i);
                all_results[i] = this->search(q, false);
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
        if (this->index.trees.empty() || query_set.rows() == 0) 
            return {vector<vector<uint>>(query_set.rows()), 0};
        const uint num_queries = query_set.rows();
        
        if (num_queries < 256) {
            return this->search_batch_binary(query_set, use_threading);
        }

        knns_inverted_index_t query_pools(query_set);
        const uint num_pools = query_pools.trees.size();

        if (!use_threading) {
             auto raw_results = this->search_multiple_internal(query_pools, num_queries);
             pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(num_queries), raw_results.second};
             for (uint p = 0; p < num_pools; p++) {
                 const auto &qrmap = query_pools.reorder_maps[p];
                 for (uint j = 0; j < (1 << KNNS_INVERTED_LEVELS); j++) {
                     uint q_abs_idx = qrmap[j];
                     if (q_abs_idx < num_queries) {
                         result.first[q_abs_idx] = std::move(raw_results.first[p * (1 << KNNS_INVERTED_LEVELS) + j]);
                     }
                 }
             }
             return result;
        }

        vector<pair<vector<vector<uint>>, size_t>> async_results(num_pools);
        
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int p = 0; p < (int)num_pools; p++) {
            knns_inverted_index_t single_pool;
            single_pool.trees.push_back(query_pools.trees[p]);
            single_pool.reorder_maps.push_back(query_pools.reorder_maps[p]);
            async_results[p] = this->search_multiple_internal(single_pool, 1 << KNNS_INVERTED_LEVELS);
        }

        pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(num_queries), 0};
        for (uint p = 0; p < num_pools; p++) {
            result.second += async_results[p].second;
            const auto &qrmap = query_pools.reorder_maps[p];
            for (uint j = 0; j < (1 << KNNS_INVERTED_LEVELS); j++) {
                uint q_abs_idx = qrmap[j];
                if (q_abs_idx < num_queries) {
                    result.first[q_abs_idx] = std::move(async_results[p].first[j]);
                }
            }
        }
        return result;
    }

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
        const array<uint, (1 << KNNS_TREE_LEVELS)> &rmap,
        const vector_t &query, 
        uint tree_idx
    ) {
        pair<vector<pair<double, uint>>, size_t> result = {vector<pair<double, uint>>(), 0};
        KNNSSearchReorderedIndex search_pool_index;
        search_pool_index.push({query.dot(tree[1]), tree_idx, 1});
        result.second++;
        while (result.first.size() < this->k_val && !search_pool_index.empty()) {
            auto [dot_val, t_idx, n_idx] = search_pool_index.top(); search_pool_index.pop();
            if (n_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS)) {
                uint abs_idx = rmap[n_idx - (1 << KNNS_TREE_LEVELS)];
                if (abs_idx != std::numeric_limits<uint>::max()) {
                    result.first.push_back({dot_val, abs_idx});
                }
                continue;
            }
            uint left = 2 * n_idx; uint right = 2 * n_idx + 1;
            double l_dot = query.dot(tree[left]); double r_dot = query.dot(tree[right]);
            search_pool_index.push({l_dot, t_idx, left}); search_pool_index.push({r_dot, t_idx, right});
            result.second += 2;
        }
        return result;
    }

    pair<vector<vector<uint>>, size_t> search_multiple_internal(
        const knns_inverted_index_t &query_pools, 
        uint num_queries_to_fill
    ) {
        pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(num_queries_to_fill), 0};
        
        KNNSReorderedDoubleSearchIndex search_index;
        for (uint tree_idx = 0; tree_idx < this->index.trees.size(); tree_idx++) {
            const auto &tree = this->index.trees[tree_idx];
            for (uint qtree_idx = 0; qtree_idx < query_pools.trees.size(); qtree_idx++) {
                const auto &qtree = query_pools.trees[qtree_idx];
                double dot_val = qtree[1].dot(tree[1]);
                search_index.push({dot_val, tree_idx, 1, qtree_idx, 1});
                result.second++;
            }
        }
        
        uint finished_queries = 0;
        uint active_queries_count = num_queries_to_fill;

        while (finished_queries < active_queries_count && !search_index.empty()) {
            auto [dot_val, tree_idx, node_idx, qtree_idx, qnode_idx] = search_index.top();
            search_index.pop();
            const auto &tree = this->index.trees[tree_idx];
            const auto &rmap = this->index.reorder_maps[tree_idx];
            const auto &qtree = query_pools.trees[qtree_idx];
            
            if (node_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS) && 
                qnode_idx >= static_cast<uint>(1 << KNNS_INVERTED_LEVELS)) {
                uint q_rel_idx_reordered = qnode_idx - (1 << KNNS_INVERTED_LEVELS);
                uint data_idx = rmap[node_idx - (1 << KNNS_TREE_LEVELS)];
                
                uint local_q_idx = qtree_idx * (1 << KNNS_INVERTED_LEVELS) + q_rel_idx_reordered;
                if (data_idx != std::numeric_limits<uint>::max() && local_q_idx < num_queries_to_fill && result.first[local_q_idx].size() < this->k_val) {
                    result.first[local_q_idx].push_back(data_idx);
                    if (result.first[local_q_idx].size() == this->k_val) finished_queries++;
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
                uint q_rel_idx_reordered = qnode_idx - (1 << KNNS_INVERTED_LEVELS);
                uint local_q_idx = qtree_idx * (1 << KNNS_INVERTED_LEVELS) + q_rel_idx_reordered;
                if (local_q_idx < num_queries_to_fill && result.first[local_q_idx].size() >= this->k_val) continue;
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

    void update_index_for_point(const vector_t &new_point, uint new_idx) {
        uint K = (1 << KNNS_TREE_LEVELS);
        uint tree_idx = new_idx / K;
        
        if (tree_idx >= this->index.trees.size()) {
            this->index.trees.emplace_back();
            this->index.reorder_maps.emplace_back();
        }
        
        uint offset = tree_idx * K;
        uint num_vectors = this->data_set.rows();
        vector<pair<uint, uint>> argmax_vals;
        for (uint j = 0; j < K; j++) {
            uint idx = offset + j;
            if (idx < num_vectors) {
                Eigen::Index argmax;
                this->data_set.row(idx).maxCoeff(&argmax);
                argmax_vals.push_back({static_cast<uint>(argmax), idx});
            } else {
                argmax_vals.push_back({std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max()});
            }
        }
        sort(argmax_vals.begin(), argmax_vals.end());

        auto &tree = this->index.trees[tree_idx];
        auto &rmap = this->index.reorder_maps[tree_idx];
        for (uint j = 0; j < K; j++) {
            uint original_idx = argmax_vals[j].second;
            if (original_idx != std::numeric_limits<uint>::max()) {
                tree[K + j] = this->data_set.row(original_idx);
                rmap[j] = original_idx;
            } else {
                tree[K + j] = vector_t::Zero(this->data_set.cols());
                rmap[j] = std::numeric_limits<uint>::max();
            }
        }
        for (int n = KNNS_TREE_LEVELS - 1; n >= 0; n--) {
            for (uint j = 0; j < static_cast<uint>(1 << n); j++) {
                const uint index = (1 << n) + j;
                tree[index] = tree[2 * index] + tree[2 * index + 1];
            }
        }
    }
};

#endif /* DC39D49B_AD0F_4907_AA56_3302836D7745 */

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
        this->index = build_index<KNNS_TREE_LEVELS, knns_index_t>(data_set);
    }

    pair<vector<uint>, size_t> search(vector_t &query) {
        if (this->index.size() == 0) return {vector<uint>(), 0};
        pair<vector<uint>, size_t> result = {vector<uint>(), 0};
        
        KNNSSearchIndex search_index;
        /*
        // DEBUG: True Max Scan
        double true_max = -1.0;
        uint true_idx = 0;
        */
        
        for (uint tree_idx = 0; tree_idx < this->index.size(); tree_idx++) {
            const auto &tree = this->index[tree_idx];
            double dot_val = query.dot(tree[1]);
            search_index.push({dot_val, tree_idx, 1});
            result.second++;
        }

        while (result.first.size() < this->k_val && !search_index.empty()) {
            auto [dot_val, tree_idx, node_idx] = search_index.top();
            search_index.pop();
            const auto &tree = this->index[tree_idx];
            if (node_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS)) {
                result.first.push_back(
                    node_idx - (1 << KNNS_TREE_LEVELS) + tree_idx * (1 << KNNS_TREE_LEVELS)
                );
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

    pair<vector<vector<uint>>, size_t> search_multiple(matrix_t &query_set) {
        pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(query_set.rows()), 0};
        if (this->index.size() == 0 || query_set.rows() == 0) return result;
        const uint num_queries = query_set.rows();
        
        knns_inverted_index_t query_pools = build_index<KNNS_INVERTED_LEVELS, knns_inverted_index_t>(query_set);
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
        
        uint flag = 0;
        while (flag < num_queries && !search_index.empty()) {
            auto [dot_val, tree_idx, node_idx, qtree_idx, qnode_idx] = search_index.top();
            search_index.pop();
            const auto &tree = this->index[tree_idx];
            const auto &qtree = query_pools[qtree_idx];
            if (
                node_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS) && 
                qnode_idx >= static_cast<uint>(1 << KNNS_INVERTED_LEVELS)
            ) {
                uint qbase_idx = qtree_idx * (1 << KNNS_INVERTED_LEVELS) + (qnode_idx - (1 << KNNS_INVERTED_LEVELS));
                uint data_idx = node_idx - (1 << KNNS_TREE_LEVELS) + tree_idx * (1 << KNNS_TREE_LEVELS);
                if (result.first[qbase_idx].size() < this->k_val) {
                    result.first[qbase_idx].push_back(data_idx);
                    if (result.first[qbase_idx].size() == this->k_val) {
                        flag++;
                    }
                }
                continue;
            }
            if (node_idx >= static_cast<uint>(1 << KNNS_TREE_LEVELS)) {
                // Only expand query tree
                uint left_qchild = 2 * qnode_idx;
                uint right_qchild = 2 * qnode_idx + 1;
                double left_dot = qtree[left_qchild].dot(tree[node_idx]);
                double right_dot = qtree[right_qchild].dot(tree[node_idx]);

                search_index.push({left_dot, tree_idx, node_idx, qtree_idx, left_qchild});
                search_index.push({right_dot, tree_idx, node_idx, qtree_idx, right_qchild});
                result.second += 2;
            } 
            else if (qnode_idx >= static_cast<uint>(1 << KNNS_INVERTED_LEVELS)) {
                // Check if we already have k results for this query
                uint qbase_idx = qtree_idx * (1 << KNNS_INVERTED_LEVELS) + (qnode_idx - (1 << KNNS_INVERTED_LEVELS));
                if (result.first[qbase_idx].size() >= this->k_val) {
                    continue;
                }
                // Only expand data tree
                uint left_child = 2 * node_idx;
                uint right_child = 2 * node_idx + 1;
                double left_dot = qtree[qnode_idx].dot(tree[left_child]);
                double right_dot = qtree[qnode_idx].dot(tree[right_child]);

                search_index.push({left_dot, tree_idx, left_child, qtree_idx, qnode_idx});
                search_index.push({right_dot, tree_idx, right_child, qtree_idx, qnode_idx});
                result.second += 2;
            } 
            else {
                // Expand both trees
                uint left_child = 2 * node_idx;
                uint right_child = 2 * node_idx + 1;
                uint left_qchild = 2 * qnode_idx;
                uint right_qchild = 2 * qnode_idx + 1;

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
        for (uint qidx = 0; qidx < num_queries; qidx++) {
            sort(result.first[qidx].begin(), result.first[qidx].end());
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
        vector<uint> true_result;
        for (size_t i = 0; i < this->k_val && i < true_results.size(); i++) {
            true_result.push_back(true_results[i].second);
        }
        if (true_result.size() < this->k_val) true_result.resize(this->k_val, 0);
        sort(true_result.begin(), true_result.end());
        auto stop = high_resolution_clock::now();
        double time = duration_cast<microseconds>(stop - start).count() / 1.0e+3;
        uint match_count = 0;
        uint i = 0, j = 0;
        while (i < result.size() && j < true_result.size()) {
            if (result[i] == true_result[j]) { match_count++; i++; j++; }
            else if (result[i] < true_result[j]) i++;
            else j++;
        }
        return {time, (this->k_val == 0) ? 0 : static_cast<double>(match_count) / this->k_val};
    }

private:
    matrix_t data_set;
    size_t k_val;
    knns_index_t index;

    template <uint N, typename IndexT>
    IndexT build_index(matrix_t &matrix) {
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
};

#endif

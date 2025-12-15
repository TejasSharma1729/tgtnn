#include "sparse_types.hpp"
#include <pybind11/pybind11.h>
#include <thread>
#include <map>
#include <algorithm>
#include <numeric>
#include <chrono>

namespace py = pybind11;

#define NUM_LEVELS 7
#define INVERTED_LEVELS 7

template <uint N = NUM_LEVELS> using KNNIndexDoubleGroupSingle = std::array<SparseVec, (1 << (N + 1))>;
template <uint N = NUM_LEVELS> using KNNIndexDoubleGroupIndex = std::vector<KNNIndexDoubleGroupSingle<N>>;
using KNNIndexDoubleGroupResultElem = std::array<std::vector<uint>, (1 << INVERTED_LEVELS)>;

template <uint N = NUM_LEVELS>
KNNIndexDoubleGroupIndex<N> build_knn_index_double_group(SparseMat &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    KNNIndexDoubleGroupIndex<N> data_index(num_indices);
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

class KNNIndexDoubleGroup {
public:
    KNNIndexDoubleGroup(const std::string &file_name, double k_val);
    std::pair<std::map<uint, std::vector<uint>>, size_t> search_cpp(const SparseMat &query_set);
    std::array<double, 2> verify_results_cpp(const SparseVec &query, const std::vector<uint> &result);

protected:
    SparseMat data_set;
    KNNIndexDoubleGroupIndex<NUM_LEVELS> data_index;
    size_t dimention;
    size_t k_val;
    std::pair<std::vector<std::vector<uint>>, size_t> search_threshold(const SparseMat &queries, double threshold);
    std::pair<KNNIndexDoubleGroupResultElem, size_t> search_pool(
        KNNIndexDoubleGroupSingle<NUM_LEVELS> &pool, 
        KNNIndexDoubleGroupSingle<INVERTED_LEVELS> &qpool, 
        uint pool_index, 
        double threshold
    );
};

KNNIndexDoubleGroup::KNNIndexDoubleGroup(const std::string &file_name, double k_val) {
    std::tie(this->data_set, this->dimention) = read_sparse_matrix(file_name);
    this->k_val = k_val;
    this->data_index = build_knn_index_double_group<NUM_LEVELS>(this->data_set);
}

std::pair<std::map<uint, std::vector<uint>>, size_t> KNNIndexDoubleGroup::search_cpp(const SparseMat &query_set) {
    std::pair<std::map<uint, std::vector<uint>>, size_t> result;
    if (this->data_index.size() == 0 || query_set.size() == 0) {
        return result;
    }
    std::vector<uint> pending_queries(query_set.size(), 0);
    std::iota(pending_queries.begin(), pending_queries.end(), 0);
    SparseMat queries;
    for (double threshold : {0.6, 0.4, 0.2, 1.0e-6}) {
        queries.clear();
        queries.resize(pending_queries.size());
        for (uint i = 0; i < pending_queries.size(); i++) {
            queries[i] = query_set[pending_queries[i]];
        }
        auto [res, num_dots] = this->search_threshold(queries, threshold);
        std::vector<uint> still_pending_queries;
        for (uint i = 0; i < res.size(); i++) {
            if (res[i].size() >= this->k_val) {
                result.first[pending_queries[i]] = res[i];
            } else {
                still_pending_queries.push_back(pending_queries[i]);
            }
        }
        result.second += num_dots;
        pending_queries = still_pending_queries;
    }
    for (uint i = 0; i < query_set.size(); i++) {
        std::vector<std::pair<double, uint>> sorted_results;
        for (uint j : result.first[i]) {
            sorted_results.push_back({dot_product(query_set[i], this->data_set[j]), j});
        }
        std::sort(sorted_results.begin(), sorted_results.end(), 
                 [](const auto &a, const auto &b) { return a.first > b.first; });
        result.first[i].clear();
        for (uint j = 0; j < this->k_val && j < sorted_results.size(); j++) {
            result.first[i].push_back(sorted_results[j].second);
        }
        std::sort(result.first[i].begin(), result.first[i].end());
    }
    return result;
}

std::pair<std::vector<std::vector<uint>>, size_t> KNNIndexDoubleGroup::search_threshold(
    const SparseMat &queries, double threshold
) {
    KNNIndexDoubleGroupIndex<INVERTED_LEVELS> query_pools = build_knn_index_double_group<INVERTED_LEVELS>(const_cast<SparseMat&>(queries));
    std::vector<std::pair<KNNIndexDoubleGroupResultElem, size_t>> async_results(
        this->data_index.size() * query_pools.size());
    std::array<std::thread, NUM_THREADS> threads;
    auto worker = [this, &async_results, &query_pools, threshold] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            for (uint qpool_index = 0; qpool_index < query_pools.size(); qpool_index++) {
                async_results[pool_index * query_pools.size() + qpool_index] = this->search_pool(
                    this->data_index[pool_index], query_pools[qpool_index], pool_index, threshold
                );
            }
        }
    };
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

std::pair<KNNIndexDoubleGroupResultElem, size_t> KNNIndexDoubleGroup::search_pool(
    KNNIndexDoubleGroupSingle<NUM_LEVELS> &pool, 
    KNNIndexDoubleGroupSingle<INVERTED_LEVELS> &qpool, 
    uint pool_index, 
    double threshold
) {
    std::pair<KNNIndexDoubleGroupResultElem, size_t> result = {KNNIndexDoubleGroupResultElem(), 1};
    std::array<double, 2 * (NUM_LEVELS + INVERTED_LEVELS) + 1> dots;
    std::array<uint, 2 * (NUM_LEVELS + INVERTED_LEVELS) + 1> idxs;
    std::array<uint, 2 * (NUM_LEVELS + INVERTED_LEVELS) + 1> qidxs;
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
        if (idx >= static_cast<uint>(1 << NUM_LEVELS)) {
            if (qidx >= static_cast<uint>(1 << INVERTED_LEVELS)) {
                uint data_idx = idx - (1 << NUM_LEVELS) + pool_index * (1 << NUM_LEVELS);
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

std::array<double, 2> KNNIndexDoubleGroup::verify_results_cpp(const SparseVec &query, const std::vector<uint> &result) {
    std::vector<std::pair<double, uint>> true_results;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < this->data_set.size(); i++) {
        true_results.push_back({dot_product(query, this->data_set[i]), i});
    }
    std::sort(true_results.begin(), true_results.end(), 
             [](const auto &a, const auto &b) { return a.first > b.first; });
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
    double recall = static_cast<double>(match_count) / this->k_val;
    return {time, recall};
}

PYBIND11_MODULE(knn_index_double_group, m) {
    m.doc() = "K-NN with quad-tree (data × query) hierarchical structure";
    
    py::class_<KNNIndexDoubleGroup>(m, "KNNIndexDoubleGroup")
        .def(py::init<std::string, double>())
        .def("search_cpp", &KNNIndexDoubleGroup::search_cpp, "Search for k-nearest neighbors");
}

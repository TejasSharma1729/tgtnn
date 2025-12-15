
#include "sparse_types.hpp"
#include <pybind11/pybind11.h>
#include <thread>
#include <set>
#include <algorithm>
#include <chrono>

namespace py = pybind11;

#define NUM_LEVELS 7
#define INVERTED_LEVELS 7

template <uint N = NUM_LEVELS> using ThresholdIndexDoubleGroupSingle = std::array<SparseVec, (1 << (N + 1))>;
template <uint N = NUM_LEVELS> using ThresholdIndexDoubleGroupIndex = std::vector<ThresholdIndexDoubleGroupSingle<N>>;
using ThresholdIndexDoubleGroupResultElem = std::array<std::vector<uint>, (1 << INVERTED_LEVELS)>;

template <uint N = NUM_LEVELS>
ThresholdIndexDoubleGroupIndex<N> build_threshold_index_double_group(SparseMat &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    ThresholdIndexDoubleGroupIndex<N> data_index(num_indices);
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

class ThresholdIndexDoubleGroup {
public:
    ThresholdIndexDoubleGroup(const std::string &file_name, double threshold);
    std::pair<std::vector<std::vector<uint>>, size_t> search_cpp(const SparseMat &queries);
    std::array<double, 3> verify_results_cpp(const SparseVec &query, const std::vector<uint> &result);

protected:
    SparseMat data_set;
    ThresholdIndexDoubleGroupIndex<NUM_LEVELS> data_index;
    size_t dimention;
    double threshold;
    std::pair<ThresholdIndexDoubleGroupResultElem, size_t> search_pool(
        ThresholdIndexDoubleGroupSingle<NUM_LEVELS> &pool, 
        ThresholdIndexDoubleGroupSingle<INVERTED_LEVELS> &qpool, 
        uint pool_index);
};

ThresholdIndexDoubleGroup::ThresholdIndexDoubleGroup(const std::string &file_name, double threshold) {
    std::tie(this->data_set, this->dimention) = read_sparse_matrix(file_name);
    this->threshold = threshold;
    this->data_index = build_threshold_index_double_group<NUM_LEVELS>(this->data_set);
}

std::pair<std::vector<std::vector<uint>>, size_t> ThresholdIndexDoubleGroup::search_cpp(const SparseMat &queries) {
    ThresholdIndexDoubleGroupIndex<INVERTED_LEVELS> query_pools = build_threshold_index_double_group<INVERTED_LEVELS>(const_cast<SparseMat&>(queries));
    std::vector<std::pair<ThresholdIndexDoubleGroupResultElem, size_t>> async_results(this->data_index.size() * query_pools.size());
    std::array<std::thread, NUM_THREADS> threads;
    auto worker = [this, &async_results, &query_pools] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            for (uint qpool_index = 0; qpool_index < query_pools.size(); qpool_index++) {
                async_results[pool_index * query_pools.size() + qpool_index] = this->search_pool(
                    this->data_index[pool_index], query_pools[qpool_index], pool_index);
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

std::pair<ThresholdIndexDoubleGroupResultElem, size_t> ThresholdIndexDoubleGroup::search_pool(
        ThresholdIndexDoubleGroupSingle<NUM_LEVELS> &pool, 
        ThresholdIndexDoubleGroupSingle<INVERTED_LEVELS> &qpool, 
        uint pool_index) 
{
    std::pair<ThresholdIndexDoubleGroupResultElem, size_t> result = {ThresholdIndexDoubleGroupResultElem(), 1};
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

        if (dot_val < this->threshold) {
            dot_idx--;
            continue;
        }
        if (idx >= static_cast<uint>(1 << NUM_LEVELS)) {
            if (qidx >= static_cast<uint>(1 << INVERTED_LEVELS)) {
                uint data_idx = idx - (1 << NUM_LEVELS) + pool_index * (1 << NUM_LEVELS);
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

std::array<double, 3> ThresholdIndexDoubleGroup::verify_results_cpp(const SparseVec &query, const std::vector<uint> &result) {
    std::vector<uint> true_result;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < this->data_set.size(); i++) {
        if (dot_product(query, this->data_set[i]) >= this->threshold) {
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
    return {time, precision, recall};
}

PYBIND11_MODULE(threshold_index_double_group, m) {
    m.doc() = "Threshold search with quad-tree (data × query) hierarchical structure";
    
    py::class_<ThresholdIndexDoubleGroup>(m, "ThresholdIndexDoubleGroup")
        .def(py::init<std::string, double>(), 
            py::arg("file_name"), 
            py::arg("threshold") = 0.8)
        .def("search_cpp", &ThresholdIndexDoubleGroup::search_cpp, "Search for vectors above threshold");
}

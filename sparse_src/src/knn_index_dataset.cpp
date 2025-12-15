#include "sparse_types.hpp"

#define NUM_LEVELS 7


template <uint N = NUM_LEVELS> using KNNIndexDatasetSingle = std::array<SparseVec, (1 << (N + 1))>;
template <uint N = NUM_LEVELS> using KNNIndexDatasetIndex = std::vector<KNNIndexDatasetSingle<N>>;
template <uint N = NUM_LEVELS> KNNIndexDatasetIndex<N> build_knn_index_dataset(SparseMat &matrix);

class KNNIndexDataset {
public:
    KNNIndexDataset(std::string file_name, size_t __k_val = 1);
    size_t size() { return this->data_set.size(); }
    std::pair<std::vector<uint>, size_t> search(SparseVec &query);
    std::array<double, 2> verify_results(SparseVec &query, std::vector<uint> &src_result);

protected:
    SparseMat data_set;
    KNNIndexDatasetIndex<NUM_LEVELS> data_index;
    size_t dimention;
    size_t k_val;
    std::pair<std::vector<uint>, size_t> search_threshold(SparseVec &query, double threshold);
    std::pair<std::vector<uint>, size_t> search_pool(KNNIndexDatasetSingle<NUM_LEVELS> &pool, SparseVec &query, uint &pool_index, double threshold);
};

// ...existing code...

// Restored method implementations for KNNIndexDataset
template <uint N = NUM_LEVELS>
KNNIndexDatasetIndex<N> build_knn_index_dataset(SparseMat &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    KNNIndexDatasetIndex<N> data_index(num_indices);
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

KNNIndexDataset::KNNIndexDataset(std::string file_name, size_t k_val) {
    std::tie(this->data_set, this->dimention) = read_sparse_matrix(file_name);
    this->k_val = k_val;
    this->data_index = build_knn_index_dataset<NUM_LEVELS>(this->data_set);
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

std::pair<std::vector<uint>, size_t> KNNIndexDataset::search_threshold(SparseVec &query, double threshold) {
    std::vector<std::pair<std::vector<uint>, size_t>> async_results(this->data_index.size());
    std::array<std::thread, NUM_THREADS> threads;
    auto worker = [this, &async_results, &query, threshold] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            async_results[pool_index] = this->search_pool(this->data_index[pool_index], query, pool_index, threshold);
        }
    };
    for (uint i = 0; i < NUM_THREADS; i++) {
        uint start = (i * this->data_index.size()) / NUM_THREADS;
        uint end = ((i + 1) * this->data_index.size()) / NUM_THREADS;
        threads[i] = std::thread(worker, start, end);
    }
    for (uint i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    for (uint i = 0; i < async_results.size(); i++) {
        result.first.insert(result.first.end(), async_results[i].first.begin(), async_results[i].first.end());
        result.second += async_results[i].second;
    }
    return result;
}

std::pair<std::vector<uint>, size_t> KNNIndexDataset::search_pool(KNNIndexDatasetSingle<NUM_LEVELS> &pool, SparseVec &query, uint &pool_index, double threshold) {
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    if (pool[1].size() == 0) {
        return result;
    }
    std::array<double, NUM_LEVELS + 1> dots;
    std::array<uint, NUM_LEVELS + 1> idxs;
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
        if (idx >= static_cast<uint>(1 << NUM_LEVELS)) {
            uint actual_idx = idx - (1 << NUM_LEVELS) + pool_index * (1 << NUM_LEVELS);
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

std::array<double, 2> KNNIndexDataset::verify_results(SparseVec &query, std::vector<uint> &result) {
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
    double recall = static_cast<double>(match_count) / this->k_val;
    return {time, recall};
}


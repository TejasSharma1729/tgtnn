#include "sparse_types.hpp"

#define NUM_LEVELS 6


using ThresholdIndexDatasetSingle = std::array<SparseVec, (1 << NUM_LEVELS)>;
using ThresholdIndexDatasetIndex = std::vector<ThresholdIndexDatasetSingle>;
ThresholdIndexDatasetIndex build_threshold_index_dataset(SparseMat &matrix);

class ThresholdIndexDataset {
public:
    ThresholdIndexDataset(std::string file_name, double threshold = 0.8);
    std::pair<std::vector<uint>, size_t> search(SparseVec &query);
    std::array<double, 3> verify_results(SparseVec &query, std::vector<uint> &src_result);

    // protected:
    SparseMat data_set;
    ThresholdIndexDatasetIndex data_index;
    size_t dimention;
    double threshold;
    std::pair<std::vector<uint>, size_t> search_pool(ThresholdIndexDatasetSingle &pool, SparseVec &query, uint &pool_index);
};

ThresholdIndexDatasetIndex build_threshold_index_dataset(SparseMat &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << NUM_LEVELS) - 1) / (1 << NUM_LEVELS);
    ThresholdIndexDatasetIndex data_index(num_indices);
    for (uint i = 0; i < num_indices; i++) {
        const uint offset = i * (1 << NUM_LEVELS);
        for (uint j = 0; j < (1 << NUM_LEVELS); j++) {
            const uint index = offset + j;
            if (index < num_vectors) {
                data_index[i][j] = matrix[index];
            }
        }
        for (uint n = 1; n <= NUM_LEVELS; n++) {
            const uint step = 1 << n;
            for (uint j = 0; j < (1 << NUM_LEVELS); j += step) {
                data_index[i][j] = add_sparse(data_index[i][j], data_index[i][j + step / 2]);
            }
        }
    }
    return data_index;
}

ThresholdIndexDataset::ThresholdIndexDataset(std::string file_name, double threshold) {
    std::tie(this->data_set, this->dimention) = read_sparse_matrix(file_name);
    this->threshold = threshold;
    this->data_index = build_threshold_index_dataset(this->data_set);
}

std::pair<std::vector<uint>, size_t> ThresholdIndexDataset::search(SparseVec &query) {
    std::vector<std::pair<std::vector<uint>, size_t>> async_results(this->data_index.size());
    std::array<std::thread, NUM_THREADS> threads;
    auto worker = [this, &async_results, &query] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            async_results[pool_index] = this->search_pool(this->data_index[pool_index], query, pool_index);
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
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    for (uint i = 0; i < async_results.size(); i++) {
        result.first.insert(result.first.end(), async_results[i].first.begin(), async_results[i].first.end());
        result.second += async_results[i].second;
    }
    return result;
}

std::pair<std::vector<uint>, size_t> ThresholdIndexDataset::search_pool(ThresholdIndexDatasetSingle &pool, SparseVec &query, uint &pool_index) {
    std::pair<std::vector<uint>, size_t> result = {std::vector<uint>(), 0};
    if (pool[0].size() == 0) {
        return result;
    }
    std::array<double, NUM_LEVELS + 1> dot_products;
    std::array<std::pair<uint, uint>, NUM_LEVELS + 1> ranges;
    ranges[0] = {0, (1 << NUM_LEVELS)};
    dot_products[0] = dot_product(query, pool[0]);
    uint dot_idx = 1;
    uint start = 0;
    uint end = 0;
    uint mid = 0;
    while (dot_idx) {
        if (dot_products[dot_idx - 1] < this->threshold) {
            dot_idx--;
            continue;
        }
        std::tie(start, end) = ranges[dot_idx - 1];
        mid = (start + end) / 2;
        if (start + 1 == end) {
            result.first.push_back(mid + pool_index * (1 << NUM_LEVELS));
            dot_idx--;
            continue;
        }
        result.second++;
        dot_products[dot_idx] = dot_product(query, pool[mid]);
        ranges[dot_idx] = {mid, end};
        dot_products[dot_idx - 1] -= dot_products[dot_idx];
        ranges[dot_idx - 1] = {start, mid};
        dot_idx++;
    }
    return result;
}

std::array<double, 3> ThresholdIndexDataset::verify_results(SparseVec &query, std::vector<uint> &result) {
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
    return {time, precision, recall};
}


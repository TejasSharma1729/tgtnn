#include "header.hpp"

#define KNNS_NUM_LEVELS 7

namespace GTnn {
    template <uint N = KNNS_NUM_LEVELS> using knns_single_elem_t = array<matrix_t, (1 << (N + 1))>;
    template <uint N = KNNS_NUM_LEVELS> using knns_single_index_t = vector<knns_single_elem_t<N>>;
    template <uint N = KNNS_NUM_LEVELS> knns_single_index_t<N> build_knns_index(matrix_t &matrix);

    class KnnsNN {
    public:
        KnnsNN(string file_name, size_t __k_val = 1);
        size_t size() { return this->data_set.rows(); }
        pair<vector<uint>, size_t> search(vector_t &query);
        array<double, 2> verify_results(vector_t &query, vector<uint> &src_result);

    protected:
        matrix_t data_set;
        knns_single_index_t<KNNS_NUM_LEVELS> data_index;
        size_t dimention;
        size_t k_val;
        pair<vector<uint>, size_t> search_threshold(vector_t &query, double threshold);
        pair<vector<uint>, size_t> search_pool(knns_single_elem_t<KNNS_NUM_LEVELS> &pool, 
            vector_t &query, uint &pool_index, double threshold);
    };
};

template <uint N>
GTnn::knns_single_index_t<N> GTnn::build_knns_index(GTnn::matrix_t &matrix) {
    const uint num_vectors = matrix.rows();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    GTnn::knns_single_index_t<N> data_index(num_indices);
    for (uint i = 0; i < num_indices; i++) {
        const uint offset = i * (1 << N);
        for (uint j = 0; j < (1 << N); j++) {
            const uint index = offset + j;
            if (index < num_vectors) {
                data_index[i][(1 << N) + j] = matrix.row(index);
            } else {
                data_index[i][(1 << N) + j] = matrix_t::Zero(1, matrix.cols());
            }
        }
        for (int n = N - 1; n >= 0; n--) {
            for (uint j = 0; j < static_cast<uint>(1 << n); j++) {
                const uint index = (1 << n) + j;
                data_index[i][index] = data_index[i][2 * index] + data_index[i][2 * index + 1];
            }
        }
    }
    return data_index;
}

GTnn::KnnsNN::KnnsNN(string file_name, size_t __k_val) {
    this->dimention = GTnn::extract_matrix(file_name, this->data_set);
    this->k_val = __k_val;
    this->data_index = GTnn::build_knns_index<KNNS_NUM_LEVELS>(this->data_set);
}

pair<vector<uint>, size_t> GTnn::KnnsNN::search(GTnn::vector_t &query) {
    if (this->data_index.size() == 0) {
        return {vector<uint>(), 0};
    }
    pair<vector<uint>, size_t> result = {vector<uint>(), 0};
    for (double threshold = 0.60; threshold >= 0.0; threshold -= 0.20) {
        auto [res, net_num_dots] = this->search_threshold(query, threshold);
        result.second += net_num_dots;
        if (res.size() >= this->k_val) {
            result.first = res;
            break;
        }
    }
    vector<pair<double, uint>> sorted_results;
    for (uint i = 0; i < result.first.size(); i++) {
        sorted_results.push_back({query.dot(this->data_set.row(result.first[i])), result.first[i]});
    }
    sort(sorted_results.begin(), sorted_results.end(), greater<pair<double, uint>>());
    result.first.clear();
    for (uint i = 0; i < this->k_val && i < sorted_results.size(); i++) {
        result.first.push_back(sorted_results[i].second);
    }
    assert (result.first.size() == this->k_val);
    if (result.first.size() < this->k_val) {
        result.first.resize(this->k_val, 0);
    }
    sort(result.first.begin(), result.first.end());
    return result;
}

pair<vector<uint>, size_t> GTnn::KnnsNN::search_threshold(GTnn::vector_t &query, double threshold) {
    vector<pair<vector<uint>, size_t>> async_results(this->data_index.size());
    array<thread, NUM_THREADS> threads;
    auto worker = [this, &async_results, &query, threshold] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            async_results[pool_index] = this->search_pool(this->data_index[pool_index], query, pool_index, threshold);
        }
    };
    for (uint i = 0; i < NUM_THREADS; i++) {
        uint start = (i * this->data_index.size()) / NUM_THREADS;
        uint end = ((i + 1) * this->data_index.size()) / NUM_THREADS;
        threads[i] = thread(worker, start, end);
    }
    for (uint i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
    pair<vector<uint>, size_t> result = {vector<uint>(), 0};
    for (uint i = 0; i < async_results.size(); i++) {
        result.first.insert(result.first.end(), async_results[i].first.begin(), async_results[i].first.end());
        result.second += async_results[i].second;
    }
    return result;
}

pair<vector<uint>, size_t> GTnn::KnnsNN::search_pool(
    GTnn::knns_single_elem_t<KNNS_NUM_LEVELS> &pool, GTnn::vector_t &query, uint &pool_index, double threshold
) {
    pair<vector<uint>, size_t> result = {vector<uint>(), 0};
    if (pool[1].rows() == 0) {
        return result;
    }
    array<double, KNNS_NUM_LEVELS + 1> dots;
    array<uint, KNNS_NUM_LEVELS + 1> idxs;
    idxs[0] = 1;
    dots[0] = query.dot(pool[1].row(0));
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
        if (idx >= static_cast<uint>(1 << KNNS_NUM_LEVELS)) {
            uint actual_idx = idx - (1 << KNNS_NUM_LEVELS) + pool_index * (1 << KNNS_NUM_LEVELS);
            result.first.push_back(actual_idx);
            dot_idx--;
            continue;
        }
        result.second++;
        dot_idx++;
        dots[dot_idx - 1] = query.dot(pool[2 * idx + 1].row(0));
        dots[dot_idx - 2] -= dots[dot_idx - 1];
        idxs[dot_idx - 1] = 2 * idx + 1;
        idxs[dot_idx - 2] = 2 * idx;
    }
    return result;
}

array<double, 2> GTnn::KnnsNN::verify_results(GTnn::vector_t &query, vector<uint> &result) {
    vector<pair<double, uint>> true_results;
    auto start = high_resolution_clock::now();
    for (long i = 0; i < data_set.rows(); i++) {
        true_results.push_back({query.dot(data_set.row(i)), i});
    }
    sort(true_results.begin(), true_results.end(), greater<pair<double, uint>>());
    vector<uint> true_result;
    for (size_t i = 0; i < this->k_val && i < true_results.size(); i++) {
        true_result.push_back(true_results[i].second);
    }
    if (true_result.size() < this->k_val) {
        true_result.resize(this->k_val, 0);
    }
    sort(true_result.begin(), true_result.end());
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
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

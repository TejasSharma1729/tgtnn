#include "header.hpp"

#define SINGLE_NUM_LEVELS 6

namespace GTnn {
    using single_elem_t = array<vector_t, (1 << SINGLE_NUM_LEVELS)>;
    using single_index_t = vector<single_elem_t>;
    single_index_t build_optimized_single_index(matrix_t &matrix);

    class OptimizedSingleNN {
    public:
        OptimizedSingleNN(string file_name, double __threshold = 0.8);
        pair<vector<uint>, size_t> search(vector_t &query);
        array<double, 3> verify_results(
            vector_t &query, 
            vector<uint> &src_result
        );

    // protected:
        matrix_t data_set;
        single_index_t data_index;
        size_t dimention;
        double threshold;
        pair<vector<uint>, size_t> search_pool(
            single_elem_t &pool, 
            vector_t &query, 
            uint &pool_index
        );
    };
};

GTnn::single_index_t GTnn::build_optimized_single_index(
    GTnn::matrix_t &matrix
) {
    const uint num_vectors = matrix.rows();
    const uint num_indices = (num_vectors + (1 << SINGLE_NUM_LEVELS) - 1) 
                                / (1 << SINGLE_NUM_LEVELS);
    GTnn::single_index_t data_index(num_indices);
    for (uint i = 0; i < num_indices; i++) {
        const uint offset = i * (1 << SINGLE_NUM_LEVELS);
        for (uint j = 0; j < (1 << SINGLE_NUM_LEVELS); j++) {
            const uint index = offset + j;
            if (index < num_vectors) {
                data_index[i][j] = matrix.row(index);
            } else {
                data_index[i][j] = vector_t::Zero(matrix.cols());
            }
        }
        for (uint n = 1; n <= SINGLE_NUM_LEVELS; n++) {
            const uint step = 1 << n;
            for (uint j = 0; j < (1 << SINGLE_NUM_LEVELS); j += step) {
                data_index[i][j] = data_index[i][j] + 
                                    data_index[i][j + step / 2];
            }
        }
    }
    return data_index;
}

GTnn::OptimizedSingleNN::OptimizedSingleNN(
    string file_name, 
    double __threshold
) {
    this->dimention = GTnn::extract_matrix(file_name, this->data_set);
    this->threshold = __threshold;
    this->data_index = GTnn::build_optimized_single_index(this->data_set);
}

pair<vector<uint>, size_t> GTnn::OptimizedSingleNN::search(
    GTnn::vector_t &query
) {
    vector<pair<vector<uint>, size_t>> async_results(this->data_index.size());
    array<thread, NUM_THREADS> threads;
    auto worker = [this, &async_results, &query] (uint start, uint end) {
        for (uint pool_index = start; pool_index < end; pool_index++) {
            async_results[pool_index] = this->search_pool(
                this->data_index[pool_index], query, pool_index
            );
        }
    };
    for (uint i = 0; i < NUM_THREADS; i++) {
        uint start = i * (this->data_index.size() / NUM_THREADS);
        uint end = (i + 1) * (this->data_index.size() / NUM_THREADS);
        if (i == NUM_THREADS - 1) {
            end = this->data_index.size();
        }
        threads[i] = thread(worker, start, end);
    }
    for (uint i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
    pair<vector<uint>, size_t> result = {vector<uint>(), 0};
    for (uint i = 0; i < async_results.size(); i++) {
        result.first.insert(
            result.first.end(), 
            async_results[i].first.begin(), 
            async_results[i].first.end()
        );
        result.second += async_results[i].second;
    }
    return result;
}


pair<vector<uint>, size_t> GTnn::OptimizedSingleNN::search_pool(
    GTnn::single_elem_t &pool, 
    GTnn::vector_t &query, 
    uint &pool_index
) {
    pair<vector<uint>, size_t> result = {vector<uint>(), 0};
    array<double, SINGLE_NUM_LEVELS + 1> dot_products;
    array<pair<uint, uint>, SINGLE_NUM_LEVELS + 1> ranges;
    ranges[0] = {0, (1 << SINGLE_NUM_LEVELS)};
    dot_products[0] = query.dot(pool[0]);
    uint dot_idx = 1;
    uint start = 0;
    uint end = 0;
    uint mid = 0;
    while (dot_idx) {
        if (dot_products[dot_idx - 1] < this->threshold) {
            dot_idx--;
            continue;
        }
        tie(start, end) = ranges[dot_idx - 1];
        mid = (start + end) / 2;
        if (start + 1 == end) {
            result.first.push_back(mid + pool_index * (1 << SINGLE_NUM_LEVELS));
            dot_idx--;
            continue;
        }
        result.second++;
        dot_products[dot_idx] = query.dot(pool[mid]);
        ranges[dot_idx] = {mid, end};
        dot_products[dot_idx - 1] -= dot_products[dot_idx];
        ranges[dot_idx - 1] = {start, mid};
        dot_idx++;
    }
    return result;
}
 
array<double, 3> GTnn::OptimizedSingleNN::verify_results(
    GTnn::vector_t &query, 
    vector<uint> &result
) {
    vector<uint> true_result;
    auto start = high_resolution_clock::now();
    for (long i = 0; i < data_set.rows(); i++) {
        if (query.dot(data_set.row(i)) >= this->threshold) {
            true_result.push_back(i);
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    double time = duration.count() / 1.0e+3;
    set<uint> true_set(true_result.begin(), true_result.end());
    set<uint> res_set(result.begin(), result.end());
    set<uint> match_set;
    set_intersection(true_set.begin(), true_set.end(),
                   res_set.begin(), res_set.end(),
                   inserter(match_set, match_set.begin()));
    double precision = (res_set.size() == 0) 
        ? 1.0 
        : (static_cast<double>(match_set.size()) / res_set.size());
    double recall = (true_set.size() == 0) 
        ? 1.0 
        : (static_cast<double>(match_set.size()) / true_set.size());
    return {time, precision, recall};
}

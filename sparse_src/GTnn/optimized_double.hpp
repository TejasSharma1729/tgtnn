#include "header.hpp"

#define NUM_LEVELS 7
#define INVERTED_LEVELS 5

namespace GTnn {
    using result_elem_t = array<vector<uint>, (1 << INVERTED_LEVELS)>;
    template <uint N = NUM_LEVELS> using sparse_single_index_t = array<sparse_vec_t, (1 << (N + 1))>;
    template <uint N = NUM_LEVELS> using sparse_index_t = vector<sparse_single_index_t<N>>;
    template <uint N = NUM_LEVELS> sparse_index_t<N> build_index(sparse_mat_t &matrix);

    class GroupTestingNN {
    public:
        GroupTestingNN(string file_name, double __threshold = 0.8);
        pair<vector<vector<uint>>, size_t> search(sparse_mat_t &queries);
        array<double, 3> verify_results(sparse_vec_t &query, vector<uint> &src_result);
        size_t size() {
            return this->data_set.size();
        }

    protected:
        sparse_mat_t data_set;
        sparse_index_t<NUM_LEVELS> data_index;
        size_t dimention;
        double threshold;
        pair<result_elem_t, size_t> search_pool(sparse_single_index_t<NUM_LEVELS> &pool, 
            sparse_single_index_t<INVERTED_LEVELS> &qpool, uint &pool_index);
    };
};

template <uint N>
GTnn::sparse_index_t<N> GTnn::build_index(GTnn::sparse_mat_t &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    GTnn::sparse_index_t<N> data_index(num_indices);
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
                data_index[i][index] = GTnn::add_sparse(data_index[i][2 * index], data_index[i][2 * index + 1]);
            }
        }
    }
    return data_index;
}

GTnn::GroupTestingNN::GroupTestingNN(string file_name, double __threshold) {
    tie(this->data_set, this->dimention) = GTnn::read_sparse_matrix(file_name);
    this->threshold = __threshold;
    this->data_index = GTnn::build_index<NUM_LEVELS>(this->data_set);
}

pair<vector<vector<uint>>, size_t> GTnn::GroupTestingNN::search(GTnn::sparse_mat_t &queries) {
    GTnn::sparse_index_t<INVERTED_LEVELS> query_pools = GTnn::build_index<INVERTED_LEVELS>(queries);
    vector<pair<GTnn::result_elem_t, size_t>> async_results(this->data_index.size() * query_pools.size());
    array<thread, NUM_THREADS> threads;
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
        threads[i] = thread(worker, start, end);
    }
    for (uint i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
    pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(queries.size()), 0};
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

pair<GTnn::result_elem_t, size_t> GTnn::GroupTestingNN::search_pool(
        GTnn::sparse_single_index_t<NUM_LEVELS> &pool, 
        GTnn::sparse_single_index_t<INVERTED_LEVELS> &qpool, 
        uint &pool_index) 
{
    pair<GTnn::result_elem_t, size_t> result = {GTnn::result_elem_t(), 1};
    array<double, 2 * (NUM_LEVELS + INVERTED_LEVELS) + 1> dots;
    array<uint, 2 * (NUM_LEVELS + INVERTED_LEVELS) + 1> idxs;
    array<uint, 2 * (NUM_LEVELS + INVERTED_LEVELS) + 1> qidxs;
    uint dot_idx = 1;
    dots[0] = GTnn::dot_product(qpool[1], pool[1]);
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
            dots[dot_idx - 1] = GTnn::dot_product(qpool[2 * qidx + 1], pool[idx]);
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
            dots[dot_idx - 1] = GTnn::dot_product(qpool[qidx], pool[2 * idx + 1]);
            dots[dot_idx - 2] -= dots[dot_idx - 1];
            
            idxs[dot_idx - 1] = 2 * idx + 1;
            idxs[dot_idx - 2] = 2 * idx;
            qidxs[dot_idx - 1] = qidx;
            qidxs[dot_idx - 2] = qidx;
            continue;
        }
        dot_idx += 3;
        result.second += 3;
        dots[dot_idx - 1] = GTnn::dot_product(qpool[2 * qidx + 1], pool[2 * idx + 1]);
        dots[dot_idx - 2] = GTnn::dot_product(qpool[2 * qidx + 1], pool[2 * idx]);
        dots[dot_idx - 3] = GTnn::dot_product(qpool[2 * qidx], pool[2 * idx + 1]);
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
 
array<double, 3> GTnn::GroupTestingNN::verify_results(GTnn::sparse_vec_t &query, vector<uint> &result) {
    vector<uint> true_result;
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < data_set.size(); i++) {
        if (dot_product(query, data_set[i]) >= this->threshold) {
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
    double precision = (res_set.size() == 0) ? 1.0 : (static_cast<double>(match_set.size()) / res_set.size());
    double recall = (true_set.size() == 0) ? 1.0 : (static_cast<double>(match_set.size()) / true_set.size());
    return {time, precision, recall};
}

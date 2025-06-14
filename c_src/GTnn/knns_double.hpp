#include "header.hpp"

#define NUM_LEVELS 8
#define INVERTED_LEVELS 6

namespace GTnn {
    using result_elem_t = array<vector<uint>, (1 << INVERTED_LEVELS)>;
    template <uint N = NUM_LEVELS> using single_index_t = array<vector_t, (1 << (N + 1))>;
    template <uint N = NUM_LEVELS> using index_t = vector<single_index_t<N>>;
    template <uint N = NUM_LEVELS> index_t<N> build_index(matrix_t &matrix);

    class GroupTestingNN {
    public:
        GroupTestingNN(string file_name, double __k_val = 0.8);
        pair<map<uint, vector<uint>>, size_t> search(matrix_t &queries);
        array<double, 2> verify_results(vector_t query, vector<uint> &src_result);
        size_t size() { return this->data_set.rows(); }

    protected:
        matrix_t data_set;
        index_t<NUM_LEVELS> data_index;
        size_t dimention;
        size_t k_val;
        pair<vector<vector<uint>>, size_t> search_threshold(matrix_t &queries, double threshold);
        pair<result_elem_t, size_t> search_pool(
            single_index_t<NUM_LEVELS> &pool, single_index_t<INVERTED_LEVELS> &qpool, 
            uint &pool_index, double threshold
        );
    };
};

template <uint N>
GTnn::index_t<N> GTnn::build_index(GTnn::matrix_t &matrix) {
    const uint num_vectors = matrix.rows();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    GTnn::index_t<N> data_index(num_indices);
    for (uint i = 0; i < num_indices; i++) {
        const uint offset = i * (1 << N);
        for (uint j = 0; j < (1 << N); j++) {
            const uint index = offset + j;
            if (index < num_vectors) {
                data_index[i][(1 << N) + j] = matrix.row(index);
            } else {
                data_index[i][(1 << N) + j].setZero(matrix.cols());
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

GTnn::GroupTestingNN::GroupTestingNN(string file_name, double __k_val) {
    this->dimention = GTnn::extract_matrix(file_name, this->data_set);
    if (!this->dimention) {
        cerr << "Error reading matrix from file: " << file_name << endl;
        exit(EXIT_FAILURE);
    }
    this->k_val = __k_val;
    this->data_index = GTnn::build_index<NUM_LEVELS>(this->data_set);
}

pair<map<uint, vector<uint>>, size_t> GTnn::GroupTestingNN::search(GTnn::matrix_t &query_set) {
    pair<map<uint, vector<uint>>, size_t> result;
    if (this->data_index.size() == 0 || query_set.rows() == 0) {
        return result;
    }
    vector<uint> pending_queries(query_set.rows(), 0);
    iota(pending_queries.begin(), pending_queries.end(), 0);
    GTnn::matrix_t queries;
    for (double threshold : {0.6, 0.4, 0.2, 1.0e-6}) {
        queries.resize(pending_queries.size(), this->dimention);
        for (uint i = 0; i < pending_queries.size(); i++) {
            queries.row(i) = query_set.row(pending_queries[i]);
        }
        auto [res, num_dots] = this->search_threshold(queries, threshold);
        vector<uint> still_pending_queries(0);
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
    assert(pending_queries.empty());
    for (uint i = 0; i < query_set.rows(); i++) {
        vector<pair<double, uint>> sorted_results;
        assert(result.first[i].size() >= this->k_val);
        for (uint j : result.first[i]) {
            sorted_results.push_back({query_set.row(i).dot(this->data_set.row(j)), j});
        }
        sort(sorted_results.begin(), sorted_results.end(), greater<pair<double, uint>>());
        result.first[i].clear();
        for (uint j = 0; j < this->k_val && j < sorted_results.size(); j++) {
            result.first[i].push_back(sorted_results[j].second);
        }
        sort(result.first[i].begin(), result.first[i].end());
        assert(result.first[i].size() == this->k_val);
    }
    return result;
}

pair<vector<vector<uint>>, size_t> GTnn::GroupTestingNN::search_threshold(
    GTnn::matrix_t &queries, double threshold
) {
    GTnn::index_t<INVERTED_LEVELS> query_pools = GTnn::build_index<INVERTED_LEVELS>(queries);
    vector<pair<GTnn::result_elem_t, size_t>> async_results(this->data_index.size() * query_pools.size());
    array<thread, NUM_THREADS> threads;
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
        threads[i] = thread(worker, start, end);
    }
    for (uint i = 0; i < NUM_THREADS; i++) {
        threads[i].join();
    }
    pair<vector<vector<uint>>, size_t> result = {vector<vector<uint>>(queries.rows()), 0};
    for (uint pool_index = 0; pool_index < this->data_index.size(); pool_index++) {
        for (uint qpool_index = 0; qpool_index < query_pools.size(); qpool_index++) {
            auto &async_result = async_results[pool_index * query_pools.size() + qpool_index].first;
            result.second += async_results[pool_index * query_pools.size() + qpool_index].second;
            for (uint i = 0; i < async_result.size(); i++) {
                uint qidx = i + qpool_index * (1 << INVERTED_LEVELS);
                if (qidx >= queries.rows()) {
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
        GTnn::single_index_t<NUM_LEVELS> &pool, 
        GTnn::single_index_t<INVERTED_LEVELS> &qpool, 
        uint &pool_index, double threshold) 
{
    pair<GTnn::result_elem_t, size_t> result = {GTnn::result_elem_t(), 1};
    array<double, 2 * (NUM_LEVELS + INVERTED_LEVELS) + 1> dots;
    array<uint, 2 * (NUM_LEVELS + INVERTED_LEVELS) + 1> idxs;
    array<uint, 2 * (NUM_LEVELS + INVERTED_LEVELS) + 1> qidxs;
    uint dot_idx = 1;
    dots[0] = qpool[1].dot(pool[1]);
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
                if (data_idx >= this->data_set.rows()) {
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
            dots[dot_idx - 1] = qpool[2 * qidx + 1].dot(pool[idx]);
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
            dots[dot_idx - 1] = qpool[qidx].dot(pool[2 * idx + 1]);
            dots[dot_idx - 2] -= dots[dot_idx - 1];
            
            idxs[dot_idx - 1] = 2 * idx + 1;
            idxs[dot_idx - 2] = 2 * idx;
            qidxs[dot_idx - 1] = qidx;
            qidxs[dot_idx - 2] = qidx;
            continue;
        }
        dot_idx += 3;
        result.second += 3;
        dots[dot_idx - 1] = qpool[2 * qidx + 1].dot(pool[2 * idx + 1]);
        dots[dot_idx - 2] = qpool[2 * qidx + 1].dot(pool[2 * idx]);
        dots[dot_idx - 3] = qpool[2 * qidx].dot(pool[2 * idx + 1]);
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
 
array<double, 2> GTnn::GroupTestingNN::verify_results(GTnn::vector_t query, vector<uint> &result) {
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

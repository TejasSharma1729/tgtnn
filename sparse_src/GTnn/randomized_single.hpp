#include "header.hpp"

#define INNER_NUM 8
#define NUM_LEVELS 4
#define OUTER_NUM (1 << NUM_LEVELS)
#define PAGE_SIZE (INNER_NUM * OUTER_NUM)

namespace GTnn {
    struct sparse_fast_elem_t {
        sparse_vec_t cum_sum;
        size_t base; // technically redundant, but useful for filling bytes and aligning
        array<uint8_t, INNER_NUM> offset;
    };
    typedef array<sparse_fast_elem_t, OUTER_NUM> sparse_fast_single_index_t;
    typedef vector<sparse_fast_single_index_t> sparse_fast_index_t;
    sparse_fast_index_t build_fast_index(sparse_mat_t &matrix);

    class GroupTestingNN {
    public:
        GroupTestingNN(string file_name, double __threshold = 0.8);
        pair<vector<uint>, size_t> search(sparse_vec_t &query);
        array<double, 3> verify_results(sparse_vec_t &query, vector<uint> &src_result);
    
    // protected:
        void search_thread(sparse_vec_t &query, vector<uint> &result, size_t &num_dots, uint thread_id);
        inline void individual_search(sparse_vec_t &query, vector<uint> &result, size_t &num_dots, 
            double dot, sparse_fast_elem_t &pool);
        sparse_mat_t data_set;
        sparse_fast_index_t data_index;
        uint size;
        uint dimention;
        double threshold;
    };
};

GTnn::sparse_fast_index_t GTnn::build_fast_index(GTnn::sparse_mat_t &matrix) {
    const uint num_vectors = matrix.size();
    for (uint i = num_vectors % PAGE_SIZE; i < PAGE_SIZE; i++) {
        matrix.push_back(sparse_vec_t());
    }
    const uint num_indices = (num_vectors + PAGE_SIZE - 1) / PAGE_SIZE;
    GTnn::sparse_fast_index_t data_index(num_indices);
    
    function<void(uint)> worker = [&data_index, &matrix, &num_indices](uint thread_id) {
        array<uint, PAGE_SIZE> indices;
        for (uint base = thread_id; base < num_indices; base += NUM_THREADS) {
            indices = GTnn::get_best_partition<INNER_NUM, OUTER_NUM>(matrix, base * PAGE_SIZE, 500);
            for (uint i = 0; i < OUTER_NUM; i++) {
                data_index[base][i].cum_sum = sparse_vec_t();
                data_index[base][i].base = base * PAGE_SIZE;
                for (uint j = 0; j < INNER_NUM; j++) {
                    data_index[base][i].cum_sum = GTnn::add_sparse(data_index[base][i].cum_sum,
                        matrix[indices[i * INNER_NUM + j]]);
                    data_index[base][i].offset[j] = indices[i * INNER_NUM + j] - base * PAGE_SIZE;
                }
            }
            for (uint n = 1; n <= NUM_LEVELS; n++) {
                const uint step = 1 << n;
                for (uint i = 0; i < OUTER_NUM; i += step) {
                    data_index[base][i].cum_sum = GTnn::add_sparse(data_index[base][i].cum_sum,
                        data_index[base][i + step / 2].cum_sum);
                }
            }
        }
    };

    array<thread, NUM_THREADS> threads;
    for (uint t = 0; t < NUM_THREADS; t++) {
        threads[t] = thread(worker, t);
    }
    for (uint t = 0; t < NUM_THREADS; t++) {
        threads[t].join();
    }
    return data_index;
}

GTnn::GroupTestingNN::GroupTestingNN(string file_name, double __threshold) {
    tie(this->data_set, this->dimention) = GTnn::read_sparse_matrix(file_name);
    this->threshold = __threshold;
    this->size = (this->data_set.size() + PAGE_SIZE - 1) / PAGE_SIZE;
    this->data_index = GTnn::build_fast_index(this->data_set);
}

pair<vector<uint>, size_t> GTnn::GroupTestingNN::search(GTnn::sparse_vec_t &query) {
    array<thread, NUM_THREADS - 1> threads;
    array<pair<vector<uint>, size_t>, NUM_THREADS - 1> ranges;
    pair<vector<uint>, size_t> result;
    for (uint t = 0; t < NUM_THREADS - 1; t++) {
        ranges[t].first = vector<uint>();
        ranges[t].second = 0;
        threads[t] = thread(&GTnn::GroupTestingNN::search_thread, this, ref(query), 
                ref(ranges[t].first), ref(ranges[t].second), t + 1);
    }
    this->search_thread(query, result.first, result.second, 0);
    for (uint t = 0; t < NUM_THREADS - 1; t++) {
        threads[t].join();
        result.first.insert(result.first.end(), ranges[t].first.begin(), ranges[t].first.end());
        result.second += ranges[t].second;
    }
    sort(result.first.begin(), result.first.end());
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

void GTnn::GroupTestingNN::search_thread(GTnn::sparse_vec_t &query, vector<uint> &result, 
        size_t &num_dots, uint thread_id) {
    for (uint base = thread_id; base < this->size; base += NUM_THREADS) {
        array<double, (1 + NUM_LEVELS)> dot_vals;
        array<pair<uint, uint>, (1 + NUM_LEVELS)> ranges;
        dot_vals[0] = GTnn::dot_product(query, this->data_index[base][0].cum_sum);
        ranges[0] = {0, OUTER_NUM};
        uint pos = 1;
        while (pos > 0) {
            if (dot_vals[pos - 1] < this->threshold) {
                pos--;
                continue;
            }
            uint start = ranges[pos - 1].first;
            uint end = ranges[pos - 1].second;
            uint mid = (start + end) / 2;
            if (start + 1 == end) {
                this->individual_search(query, result, num_dots, dot_vals[pos - 1], this->data_index[base][mid]);
                pos--;
                continue;
            }
            num_dots++;
            dot_vals[pos] = GTnn::dot_product(query, this->data_index[base][mid].cum_sum);
            ranges[pos] = {mid, end};
            dot_vals[pos - 1] -= dot_vals[pos];
            ranges[pos - 1] = {start, mid};
            pos++;
        }
    }
}

inline void GTnn::GroupTestingNN::individual_search(GTnn::sparse_vec_t &query, vector<uint> &result, size_t &num_dots, 
    double dot, GTnn::sparse_fast_elem_t &pool) {
    double net_dot = 0;
    for (uint i = 1; i < INNER_NUM; i++) {
        num_dots++;
        double dot_i = GTnn::dot_product(query, this->data_set[pool.base + pool.offset[i]]);
        if (dot_i >= this->threshold) {
            result.push_back(pool.base + pool.offset[i]);
        }
        net_dot += dot_i;
    }
    if (dot - net_dot >= this->threshold) {
        result.push_back(pool.base + pool.offset[0]);
    }
}
#include "header.hpp"

#define LOW_THRESHOLD 0.05

namespace GTnn {
    class dim_based_gtnn_t {
    public:
        dim_based_gtnn_t(const string &__path, const string &__dataset, double __threshold = 0.8);
        ~dim_based_gtnn_t(void) = default;
        pair<vector<uint>, size_t> search(vector_t &query);
        array<double, 3> exhaustive_search(vector_t &query, vector<uint> &result);

    protected:
        void build_index(void);
        void search_index(vector_t &query, uint dimention, vector<uint> &result, size_t &num_dot_products);
        string path;
        string dataset;
        uint dimension;
        matrix_t data;
        double threshold;
        vector<vector<pair<vector_t, uint>>> data_index;
    };
}

GTnn::dim_based_gtnn_t::dim_based_gtnn_t(const string &__path, const string &__dataset, double __threshold) {
    this->path = __path;
    this->dataset = __dataset;
    this->threshold = __threshold;
    GTnn::extract_matrix(GTnn::path_append(path, "X.txt"), this->data);
    
    this->dimension = this->data.cols();
    this->build_index();
}

pair<vector<uint>, size_t> GTnn::dim_based_gtnn_t::search(GTnn::vector_t &query) {
    pair<vector<uint>, size_t> result_and_num_dots;
    vector<thread> threads;

    vector<pair<vector<uint>, size_t>> dim_wise_results(this->dimension, {vector<uint>(), 0});
    vector<uint> dims;
    for (uint d = 0; d < this->dimension; d++) {
        if (query(d) >= LOW_THRESHOLD) {
            threads.push_back(thread(&GTnn::dim_based_gtnn_t::search_index, this, ref(query), d, 
                                    ref(dim_wise_results[d].first), ref(dim_wise_results[d].second)));
            dims.push_back(d);
        }
    }
    for (uint t = 0; t < threads.size(); t++) {
        threads[t].join();
        uint d = dims[t];
        if (dim_wise_results[d].first.size() > 0) {
            for (uint res : dim_wise_results[d].first) {
                result_and_num_dots.first.push_back(res);
            }
            result_and_num_dots.second += dim_wise_results[d].second;
        }
    }
    return result_and_num_dots;
}

array<double, 3> GTnn::dim_based_gtnn_t::exhaustive_search(GTnn::vector_t &query, vector<uint> &result) {
    array<double, 3> result_and_time = {0, 0, 0};
    
    vector<uint> true_result;
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (uint i = 0; i < this->data.rows(); i++) {
        double dot = this->data.row(i).dot(query);
        if (dot > this->threshold) {
            true_result.push_back(i);
        }
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    result_and_time[0] = duration_cast<microseconds>(end - start).count() / 1.0e+3;

    set<uint> result_set(true_result.begin(), true_result.end());
    set<uint> ground_truth_set(result.begin(), result.end());
    set<uint> intersection;
    set_intersection(result_set.begin(), result_set.end(),
                     ground_truth_set.begin(), ground_truth_set.end(),
                     inserter(intersection, intersection.begin()));
    result_and_time[1] = static_cast<double>(intersection.size()) / true_result.size();
    result_and_time[2] = static_cast<double>(intersection.size()) / result.size();
    return result_and_time;
}

void GTnn::dim_based_gtnn_t::build_index(void) {
    this->data_index.resize(this->dimension);
    for (uint d = 0; d < this->dimension; d++) {
        for (uint i = 0; i < this->data.rows(); i++) {
            if (this->data(i, d) < LOW_THRESHOLD) {
                continue;
            }
            this->data_index[d].push_back({this->data.row(i), i});
        }
        const uint log_size = ceil(log2(this->data_index[d].size()));
        this->data_index[d].resize(1 << log_size, {vector_t::Zero(this->dimension), static_cast<uint>(-1)});
        for (uint n = 1; n <= log_size; n++) {
            for (uint i = 0; i < this->data_index[d].size(); i += (1 << n)) {
                this->data_index[d][i].first += this->data_index[d][i + (1 << (n - 1))].first;
            }
        }
    }
}

void GTnn::dim_based_gtnn_t::search_index(vector_t &query, uint dimention, vector<uint> &result, 
    size_t &num_dot_products) {
    const vector<pair<GTnn::vector_t, uint>> &pool = this->data_index[dimention];
    if (pool.size() == 0) {
        return;
    }
    vector<double> dot_products;
    vector<pair<uint, uint>> ranges;
    ranges.push_back({0, pool.size()});
    dot_products.push_back(pool[0].first.dot(query));
    
    while (!ranges.empty()) {
        if (dot_products.back() < this->threshold) {
            dot_products.pop_back();
            ranges.pop_back();
            continue;
        }
        auto [start, end] = ranges.back();
        uint mid = (start + end) / 2;
        if (start + 1 == end) {
            result.push_back(pool[mid].second);
            dot_products.pop_back();
            ranges.pop_back();
            continue;
        }
        num_dot_products++;
        dot_products.push_back(pool[mid].first.dot(query));
        ranges.push_back({mid, end});
        dot_products[dot_products.size() - 2] -= dot_products.back();
        ranges[ranges.size() - 2] = {start, mid};
    }
}
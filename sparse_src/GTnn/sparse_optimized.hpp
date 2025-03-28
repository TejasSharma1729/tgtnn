#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

namespace GTnn {
    struct sparse_elem_t {
        uint index;
        float value;
    };
    typedef vector<sparse_elem_t> sparse_vec_t;
    typedef vector<sparse_vec_t> sparse_mat_t;
    pair<sparse_mat_t, size_t> read_sparse_matrix(const string &file_name);
    sparse_vec_t add_sparse(const sparse_vec_t &one, const sparse_vec_t &two);
    double dot_product(const sparse_vec_t &one, const sparse_vec_t &two);
    template <uint N> vector<array<sparse_vec_t, (1 << N)>> build_index(const sparse_mat_t &matrix);

    template <uint N>
    class GroupTestingNN {
    public:
        GroupTestingNN(string file_name, double __threshold = 0.8);
        pair<vector<uint>, size_t> search(sparse_vec_t &query);
        array<double, 3> verify_results(sparse_vec_t &query, vector<uint> &src_result);

    // protected:
        sparse_mat_t data_set;
        vector<array<sparse_vec_t, (1 << N)>> data_index;
        size_t dimention;
        double threshold;
    };
};

pair<GTnn::sparse_mat_t, size_t> GTnn::read_sparse_matrix(const string &file_name) {
    ifstream csr_file(file_name, ios::binary);
    if (!csr_file.is_open()) {
        throw runtime_error("Cannot open file " + file_name);
    }
    size_t num_vectors;
    size_t num_non_zero;
    pair<GTnn::sparse_mat_t, size_t> result_and_dimention;
    csr_file.read(reinterpret_cast<char *>(&num_vectors), sizeof(size_t));
    csr_file.read(reinterpret_cast<char *>(&result_and_dimention.second), sizeof(size_t));
    csr_file.read(reinterpret_cast<char *>(&num_non_zero), sizeof(size_t));

    vector<float> data(num_non_zero);
    vector<uint> indices(num_non_zero);
    vector<size_t> indptr(num_vectors + 1);
    csr_file.read(reinterpret_cast<char *>(indptr.data()), (num_vectors + 1) * sizeof(size_t));
    csr_file.read(reinterpret_cast<char *>(indices.data()), num_non_zero * sizeof(uint));
    csr_file.read(reinterpret_cast<char *>(data.data()), num_non_zero * sizeof(float));
    

    result_and_dimention.first.resize(num_vectors);
    for (long i = 0; i < num_vectors; i++) {
        result_and_dimention.first[i].resize(indptr[i + 1] - indptr[i]);
        double norm = 0;
        for (long j = indptr[i]; j < indptr[i + 1]; j++) {
            result_and_dimention.first[i][j - indptr[i]] = {indices[j], data[j]};
            norm += data[j] * data[j];
        }
        if (norm == 0) {
            continue;
        }
        for (size_t j = 0; j < result_and_dimention.first[i].size(); j++) {
            result_and_dimention.first[i][j].value /= sqrt(norm);
        }
    }
    return result_and_dimention;
}

GTnn::sparse_vec_t GTnn::add_sparse(const GTnn::sparse_vec_t &one, const GTnn::sparse_vec_t &two) {
    GTnn::sparse_vec_t result;
    uint i = 0, j = 0;
    while (i < one.size() && j < two.size()) {
        if (one[i].index == two[j].index) {
            result.push_back({one[i].index, one[i].value + two[j].value});
            i++;
            j++;
        } else if (one[i].index < two[j].index) {
            result.push_back(one[i]);
            i++;
        } else {
            result.push_back(two[j]);
            j++;
        }
    }
    while (i < one.size()) {
        result.push_back(one[i]);
        i++;
    }
    while (j < two.size()) {
        result.push_back(two[j]);
        j++;
    }
    return result;
}

double GTnn::dot_product(const GTnn::sparse_vec_t &one, const GTnn::sparse_vec_t &two) {
    double result = 0;
    uint i = 0, j = 0;
    while (i < one.size() && j < two.size()) {
        if (one[i].index == two[j].index) {
            result += one[i].value * two[j].value;
            i++;
            j++;
        } else if (one[i].index < two[j].index) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}

template <uint N> vector<array<GTnn::sparse_vec_t, (1 << N)>> GTnn::build_index(const GTnn::sparse_mat_t &matrix) {
    const uint num_vectors = matrix.size();
    const uint num_indices = (num_vectors + (1 << N) - 1) / (1 << N);
    vector<array<GTnn::sparse_vec_t, (1 << N)>> data_index(num_indices);
    for (uint i = 0; i < num_indices; i++) {
        const uint offset = i * (1 << N);
        for (uint j = 0; j < (1 << N); j++) {
            const uint index = offset + j;
            if (index < num_vectors) {
                data_index[i][j] = matrix[index];
            }
        }
        for (uint n = 1; n <= N; n++) {
            const uint step = 1 << n;
            for (uint j = 0; j < (1 << N); j += step) {
                data_index[i][j] = GTnn::add_sparse(data_index[i][j], data_index[i][j + step / 2]);
            }
        }
    }
    return data_index;
}

template <uint N> GTnn::GroupTestingNN<N>::GroupTestingNN(string file_name, double __threshold) {
    tie(this->data_set, this->dimention) = GTnn::read_sparse_matrix(file_name);
    this->threshold = __threshold;
    this->data_index = GTnn::build_index<N>(this->data_set);
}

template <uint N>  pair<vector<uint>, size_t> GTnn::GroupTestingNN<N>::search(GTnn::sparse_vec_t &query) {
    array<double, N + 1> dot_products;
    array<pair<uint, uint>, N + 1> ranges;
    pair<vector<uint>, size_t> result;
    for (array<GTnn::sparse_vec_t, (1 << N)> &pool : this->data_index) {
        ranges[0] = {0, (1 << N)};
        dot_products[0] = GTnn::dot_product(query, pool[0]);
        uint dot_idx = 1;
        uint start = 0;
        uint end = 0;
        uint mid = 0;
        result.second++;
        while (dot_idx) {
            if (dot_products[dot_idx - 1] < this->threshold) {
                dot_idx--;
                continue;
            }
            tie(start, end) = ranges[dot_idx - 1];
            if (start + 1 == end) {
                result.first.push_back(mid);
                dot_idx--;
                continue;
            }
            mid = (start + end) / 2;
            result.second++;
            dot_products[dot_idx] = GTnn::dot_product(query, pool[mid]);
            ranges[dot_idx] = {mid, end};
            dot_products[dot_idx - 1] -= dot_products[dot_idx];
            ranges[dot_idx - 1] = {start, mid};
            dot_idx++;
        }
    }
    return result;
}

template <uint N>  
array<double, 3> GTnn::GroupTestingNN<N>::verify_results(GTnn::sparse_vec_t &query, vector<uint> &result) {
    vector<uint> true_result;
    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < data_set.size(); i++) {
        if (dot_product(query, data_set[i]) >= threshold) {
            true_result.push_back(i);
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    double time = duration.count() / 1.0e+3;
    unordered_set<uint> true_set(true_result.begin(), true_result.end());
    unordered_set<uint> res_set(result.begin(), result.end());
    double precision = (res_set.size() == 0) ? 1.0 : (static_cast<double>(true_set.size()) / res_set.size());
    double recall = (true_set.size() == 0) ? 1.0 : (static_cast<double>(true_set.size()) / true_set.size());
    return {time, precision, recall};
}
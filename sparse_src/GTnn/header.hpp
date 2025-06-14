#include <bits/stdc++.h>
using namespace std;
using namespace std::chrono;

#define NUM_THREADS 16

namespace GTnn {
    struct sparse_elem_t {
        uint index;
        float value;
    };
    typedef vector<sparse_elem_t> sparse_vec_t;
    typedef vector<sparse_vec_t> sparse_mat_t;
    template <uint M, uint N> array<uint, M*N> get_best_partition(sparse_mat_t &data, uint start, uint t);

    bool check_file(ofstream &file);
    bool check_file(ifstream &file);
    bool check_file(FILE *file);
    bool check_file(const string &file_name);
    bool check_file(const char *file_name);

    pair<sparse_mat_t, size_t> read_sparse_matrix(const string &file_name);
    sparse_vec_t add_sparse(sparse_vec_t &one, sparse_vec_t &two);
    double dot_product(sparse_vec_t &one, sparse_vec_t &two);
    bool compare_sparse(sparse_vec_t one, sparse_vec_t two);
    bool compare_float(float one, float two) {
        return fabs(one - two) < 1.0e-5;
    }
}

bool GTnn::check_file(ofstream &file) {
    return file.is_open();
}
bool GTnn::check_file(ifstream &file) {
    return file.is_open();
}
bool GTnn::check_file(FILE *file) {
    return file != nullptr;
}

bool GTnn::check_file(const string &file_name) {
    ifstream file(file_name);
    bool result = file.is_open();
    file.close();
    return result;
}

bool GTnn::check_file(const char *file_name) {
    ifstream file(file_name);
    bool result = file.is_open();
    file.close();
    return result;
}

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
    for (size_t i = 0; i < num_vectors; i++) {
        result_and_dimention.first[i].resize(indptr[i + 1] - indptr[i]);
        double norm = 0;
        for (size_t j = indptr[i]; j < indptr[i + 1]; j++) {
            result_and_dimention.first[i][j - indptr[i]] = {indices[j], data[j]};
            norm += data[j] * data[j];
            if (data[j] < 0) {
                throw runtime_error("Negative value in sparse matrix");
            }
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

GTnn::sparse_vec_t GTnn::add_sparse(GTnn::sparse_vec_t &one, GTnn::sparse_vec_t &two) {
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

double GTnn::dot_product(GTnn::sparse_vec_t &one, GTnn::sparse_vec_t &two) {
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

bool GTnn::compare_sparse(GTnn::sparse_vec_t one, GTnn::sparse_vec_t two) {
    if (one.size() != two.size()) {
        return false;
    }
    for (uint i = 0; i < one.size(); i++) {
        if (one[i].index != two[i].index || !compare_float(one[i].value, two[i].value)) {
            return false;
        }
    }
    return true;
}

template <uint M, uint N> array<uint, M*N> GTnn::get_best_partition(GTnn::sparse_mat_t &data, uint start, uint t) {
    constexpr uint MN = M * N;
    array<uint, M*N> best_perm;
    double max_sum = -1.0;
    vector<uint> base_indices(MN);
    iota(base_indices.begin(), base_indices.end(), start);
    random_device rd;
    mt19937 rng(rd());
    
    for (uint iter = 0; iter < t; ++iter) {
        vector<uint> current = base_indices;
        shuffle(current.begin(), current.end(), rng);
        double total = 0.0;
        // Process each of the N groups
        for (uint group = 0; group < N; ++group) {
            unordered_map<uint, float> group_sum;
            // Aggregate sparse vectors in current group
            for (uint i = group*M; i < (group+1)*M; ++i) {
                const auto& vec = data[current[i]];
                for (const auto& elem : vec) {
                    group_sum[elem.index] += elem.value;
                }
            }
            // Calculate squared L2 norm
            double group_sq = 0;
            for (const auto& [idx, val] : group_sum) {
                group_sq += val * val;
            }
            total += group_sq;
        }
        // Update best permutation
        if (total > max_sum) {
            max_sum = total;
            copy(current.begin(), current.end(), best_perm.begin());
        }
    }
    return best_perm;
}
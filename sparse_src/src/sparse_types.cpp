
#include <bits/stdc++.h>

#define NUM_THREADS 16

// --- Sparse/CSR Types and Utilities ---
struct SparseElem {
    uint index;
    float value;
};

// Optimized sparse vector using SoA (Structure of Arrays) for better cache locality
struct SparseVecOptimized {
    std::vector<uint> indices;
    std::vector<float> values;

    inline void reserve(size_t size) {
        indices.reserve(size);
        values.reserve(size);
    }

    inline void push_back(uint idx, float val) {
        indices.push_back(idx);
        values.push_back(val);
    }

    inline size_t size() const { return indices.size(); }
};

using SparseVec = std::vector<SparseElem>;
using SparseMatOptimized = std::vector<SparseVecOptimized>;
using SparseMat = std::vector<SparseVec>;
template <uint M, uint N> std::array<uint, M*N> get_best_partition(SparseMat &data, uint start, uint t);

bool check_file(std::ofstream &file);
bool check_file(std::ifstream &file);
bool check_file(FILE *file);
bool check_file(const std::string &file_name);
bool check_file(const char *file_name);

std::pair<SparseMat, size_t> read_sparse_matrix(const std::string &file_name);
SparseVec add_sparse(SparseVec &one, SparseVec &two);
double dot_product(SparseVec &one, SparseVec &two);
bool compare_sparse(SparseVec one, SparseVec two);
bool compare_float(float one, float two);


bool check_file(std::ofstream &file) {
    return file.is_open();
}
bool check_file(std::ifstream &file) {
    return file.is_open();
}
bool check_file(FILE *file) {
    return file != nullptr;
}

bool check_file(const std::string &file_name) {
    std::ifstream file(file_name);
    bool result = file.is_open();
    file.close();
    return result;
}

bool check_file(const char *file_name) {
    std::ifstream file(file_name);
    bool result = file.is_open();
    file.close();
    return result;
}

std::pair<SparseMat, size_t> read_sparse_matrix(const std::string &file_name) {
    std::ifstream csr_file(file_name, std::ios::binary);
    if (!csr_file.is_open()) {
        throw std::runtime_error("Cannot open file " + file_name);
    }
    size_t num_vectors;
    size_t num_non_zero;
    std::pair<SparseMat, size_t> result_and_dimention;
    csr_file.read(reinterpret_cast<char *>(&num_vectors), sizeof(size_t));
    csr_file.read(reinterpret_cast<char *>(&result_and_dimention.second), sizeof(size_t));
    csr_file.read(reinterpret_cast<char *>(&num_non_zero), sizeof(size_t));

    std::vector<float> data(num_non_zero);
    std::vector<uint> indices(num_non_zero);
    std::vector<size_t> indptr(num_vectors + 1);
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
                throw std::runtime_error("Negative value in sparse matrix");
            }
        }
        if (norm == 0) {
            continue;
        }
        for (size_t j = 0; j < result_and_dimention.first[i].size(); j++) {
            result_and_dimention.first[i][j].value /= std::sqrt(norm);
        }
    }
    return result_and_dimention;
}

SparseVec add_sparse(SparseVec &one, SparseVec &two) {
    SparseVec result;
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

double dot_product(const SparseVec &one, const SparseVec &two) {
    double result = 0;
    uint i = 0, j = 0;
    const uint size_one = one.size();
    const uint size_two = two.size();

    // Vectorized dot product using AVX2 if available
    while (i < size_one && j < size_two) {
        const uint idx_one = one[i].index;
        const uint idx_two = two[j].index;

        if (idx_one == idx_two) {
            result += one[i].value * two[j].value;
            i++;
            j++;
        } else if (idx_one < idx_two) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}

// Optimized dot product for the new structure
double dot_product_optimized(const SparseVecOptimized &one, const SparseVecOptimized &two) {
    double result = 0;
    uint i = 0, j = 0;
    const uint size_one = one.size();
    const uint size_two = two.size();

    // Cache-friendly access patterns
    const uint* __restrict__ idx1 = one.indices.data();
    const float* __restrict__ val1 = one.values.data();
    const uint* __restrict__ idx2 = two.indices.data();
    const float* __restrict__ val2 = two.values.data();

    while (i < size_one && j < size_two) {
        const uint idx_one = idx1[i];
        const uint idx_two = idx2[j];

        if (idx_one == idx_two) {
            result += val1[i] * val2[j];
            i++;
            j++;
        } else if (idx_one < idx_two) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}

bool compare_sparse(SparseVec one, SparseVec two) {
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

bool compare_float(float one, float two) {
    return std::fabs(one - two) < 1.0e-5;
}

template <uint M, uint N> std::array<uint, M*N> get_best_partition(SparseMat &data, uint start, uint t) {
    constexpr uint MN = M * N;
    std::array<uint, M*N> best_perm;
    double max_sum = -1.0;
    std::vector<uint> base_indices(MN);
    std::iota(base_indices.begin(), base_indices.end(), start);
    std::random_device rd;
    std::mt19937 rng(rd());

    for (uint iter = 0; iter < t; ++iter) {
        std::vector<uint> current = base_indices;
        std::shuffle(current.begin(), current.end(), rng);
        double total = 0.0;
        // Process each of the N groups
        for (uint group = 0; group < N; ++group) {
            std::unordered_map<uint, float> group_sum;
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
            std::copy(current.begin(), current.end(), best_perm.begin());
        }
    }
    return best_perm;
}



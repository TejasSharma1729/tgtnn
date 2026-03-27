#ifndef D06ED106_555D_4C4D_8C5F_06CF6C90B5F0
#define D06ED106_555D_4C4D_8C5F_06CF6C90B5F0

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

typedef unsigned int uint;
typedef __uint128_t uint128_t;
typedef __int128_t int128_t;

const uint NUM_POOLS_COEFF = 400; 
const uint POOLS_PER_ITEM = 3; 
const uint SIGNATURE_COEFF = 6; 

using std::vector, std::array, std::pair, std::shared_ptr;
using std::set, std::map, std::queue, std::unordered_map, std::unordered_set;
using std::cin, std::cout, std::cerr, std::endl;
using std::optional, std::nullopt;

/**
 * @brief Sparse dataset structure using CSR format.
 */
struct SparseDataset {
    vector<float> data;      // float32
    vector<uint32_t> indices; // uint32
    vector<uint64_t> indptr;  // uint64
    uint32_t num_rows;
    uint32_t num_cols;

    inline uint32_t nnz(uint32_t row) const {
        return indptr[row+1] - indptr[row];
    }

    inline const float* row_data(uint32_t row) const {
        return data.data() + indptr[row];
    }

    inline const uint32_t* row_indices(uint32_t row) const {
        return indices.data() + indptr[row];
    }
};

/**
 * @brief Compute dot product between a dense Eigen vector and a sparse row.
 */
inline float dot_sparse_dense(
    const float* s_data, 
    const uint32_t* s_indices, 
    uint32_t s_nnz, 
    const Eigen::VectorXf& dense_query
) {
    float res = 0;
    for (uint32_t i = 0; i < s_nnz; ++i) {
        res += s_data[i] * dense_query(s_indices[i]);
    }
    return res;
}

/**
 * @brief Efficiently scores a query against a set of identified items using Sparse data.
 */
inline vector<uint> getTopKSparse(
    const Eigen::VectorXf &query,
    const SparseDataset &dataset,
    const std::set<uint> &identified,
    uint k
) {
    if (identified.empty()) return {};
    vector<pair<float, uint>> scores;
    scores.reserve(identified.size());
    
    vector<uint> indices_to_check(identified.begin(), identified.end());
    
    #pragma omp parallel
    {
        vector<pair<float, uint>> local_scores;
        #pragma omp for nowait
        for (size_t i = 0; i < indices_to_check.size(); ++i) {
            uint idx = indices_to_check[i];
            float dp = dot_sparse_dense(
                dataset.row_data(idx), 
                dataset.row_indices(idx), 
                dataset.nnz(idx), 
                query
            );
            local_scores.push_back({dp, idx});
        }
        #pragma omp critical
        scores.insert(scores.end(), local_scores.begin(), local_scores.end());
    }

    auto cmp = [](const pair<float, uint> &a, const pair<float, uint> &b) {
        return a.first > b.first; 
    };
    
    if (scores.size() > k) {
        std::nth_element(scores.begin(), scores.begin() + k, scores.end(), cmp);
        scores.resize(k);
    }
    std::sort(scores.begin(), scores.end(), cmp);

    vector<uint> topK;
    for (const auto &p : scores) topK.push_back(p.second);
    // std::sort(topK.begin(), topK.end()); // Saffron seems to expect sorted by score or sorted by index? 
    // Original Saffron sorted topK indices at the end.
    std::sort(topK.begin(), topK.end());
    return topK;
}

struct HashBucket {
    uint hash_val; 
    uint start_idx; 
    uint num_items; 
};

struct Candidate {
    uint pool_idx; 
    uint item_idx; 
};

/**
 * @brief Efficiently scores a query against a set of identified items using Dense data.
 */
inline vector<uint> getTopKEigen(
    const Eigen::VectorXf &query,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &dataset,
    const std::set<uint> &identified,
    uint k
) {
    if (identified.empty()) return {};
    vector<pair<float, uint>> scores;
    scores.reserve(identified.size());
    
    for (uint idx : identified) {
        float dp = query.dot(dataset.row(idx));
        scores.push_back({dp, idx});
    }

    auto cmp = [](const pair<float, uint> &a, const pair<float, uint> &b) {
        return a.first > b.first; 
    };
    
    if (scores.size() > k) {
        std::nth_element(scores.begin(), scores.begin() + k, scores.end(), cmp);
        scores.resize(k);
    }
    std::sort(scores.begin(), scores.end(), cmp);

    vector<uint> topK;
    for (const auto &p : scores) topK.push_back(p.second);
    std::sort(topK.begin(), topK.end());
    return topK;
}

/**
 * @brief Normalizes a 2D numpy array of data points.
 */
inline vector<vector<float>> normalizeDataset(pybind11::array_t<float> data_points_arr, bool normalize) {
    auto r = data_points_arr.unchecked<2>();
    uint num_items = data_points_arr.shape(0);
    uint dimension = data_points_arr.shape(1);
    vector<vector<float>> data_points(num_items, vector<float>(dimension));
    
    #pragma omp parallel for
    for (int i = 0; i < (int)num_items; ++i) {
        float norm_sq = 0;
        for (uint j = 0; j < dimension; ++j) {
            float val = r(i, j);
            data_points[i][j] = val;
            norm_sq += val * val;
        }
        if (normalize) {
            float norm = std::sqrt(norm_sq);
            if (norm > 1e-9) {
                for (uint j = 0; j < dimension; ++j) data_points[i][j] /= norm;
            }
        }
    }
    return data_points;
}

/**
 * @brief Normalizes a 1D numpy array data point.
 */
inline vector<float> normalizeDataPoint(pybind11::array_t<float> query_arr, bool normalize) {
    auto r = query_arr.unchecked<1>();
    uint dimension = query_arr.shape(0);
    vector<float> query(dimension);
    float norm_sq = 0;
    for (uint i = 0; i < dimension; ++i) {
        query[i] = r(i);
        norm_sq += query[i] * query[i];
    }
    if (normalize) {
        float norm = std::sqrt(norm_sq);
        if (norm > 1e-9) {
            for (uint i = 0; i < dimension; ++i) query[i] /= norm;
        }
    }
    return query;
}

/**
 * @brief PRNG for on-the-fly weight generation (SplitMix64).
 */
inline uint64_t splitmix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

/**
 * @brief Utilities for combining hash values.
 */
inline uint64_t combine_hashes(uint64_t item1, uint64_t item2) {
    return item1 * 0xC4DD05BF + item2 * 0x6C8702C9;
}

#endif /* D06ED106_555D_4C4D_8C5F_06CF6C90B5F0 */

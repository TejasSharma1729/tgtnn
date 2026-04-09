#ifndef D06ED106_555D_4C4D_8C5F_06CF6C90B5F0
#define D06ED106_555D_4C4D_8C5F_06CF6C90B5F0

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

typedef unsigned int uint;
typedef unsigned short ushort;
typedef __uint128_t uint128_t;
typedef __int128_t int128_t;
typedef __float128 quadruple;

/** @brief Coefficient for computing the default number of pools: num_pools = k * NUM_POOLS_COEFF. */
const uint NUM_POOLS_COEFF = 400;
/** @brief Default number of pools each item is independently assigned to. */
const uint POOLS_PER_ITEM = 3;
/** @brief Coefficient used in signature-length calculation (signature_length = 3 * (2*L + 1)). */
const uint SIGNATURE_COEFF = 6;

using std::vector, std::array, std::pair, std::shared_ptr;
using std::set, std::map, std::queue, std::unordered_map, std::unordered_set;
using std::cin, std::cout, std::cerr, std::endl;
using std::optional, std::nullopt;

/**
 * @brief Sparse dataset in Compressed Sparse Row (CSR) format.
 *
 * Mirrors the layout of a scipy.sparse.csr_matrix so that numpy arrays
 * can be zero-copy mapped into this structure.
 */
struct SparseDataset {
    /** @brief Non-zero float32 values in row-major order (CSR data array). */
    vector<float> data;
    /** @brief Column indices of the non-zero values (CSR indices array). */
    vector<uint32_t> indices;
    /** @brief Row pointer array; row i spans columns [indptr[i], indptr[i+1]) (CSR indptr). */
    vector<uint64_t> indptr;
    /** @brief Number of rows (items/documents) in the dataset. */
    uint32_t num_rows;
    /** @brief Number of columns (features/dimensions) in the dataset. */
    uint32_t num_cols;

    /**
     * @brief Returns the number of non-zero elements in a given row.
     * @param row Zero-based row index.
     * @return uint32_t Number of non-zeros in row `row`.
     */
    inline uint32_t nnz(uint32_t row) const {
        return indptr[row+1] - indptr[row];
    }

    /**
     * @brief Returns a pointer to the first non-zero value of a given row.
     * @param row Zero-based row index.
     * @return const float* Pointer into the data array at the start of row `row`.
     */
    inline const float* row_data(uint32_t row) const {
        return data.data() + indptr[row];
    }

    /**
     * @brief Returns a pointer to the first column index of a given row.
     * @param row Zero-based row index.
     * @return const uint32_t* Pointer into the indices array at the start of row `row`.
     */
    inline const uint32_t* row_indices(uint32_t row) const {
        return indices.data() + indptr[row];
    }
};

/**
 * @brief Compute the dot product between a sparse row and a dense Eigen vector.
 *
 * @param s_data        Pointer to the non-zero float values of the sparse row.
 * @param s_indices     Pointer to the column indices of the non-zero values.
 * @param s_nnz         Number of non-zero elements in the sparse row.
 * @param dense_query   Dense Eigen vector to dot against.
 * @return float        The resulting dot product.
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
 * @brief Score a query against a candidate set and return the top-k item indices.
 *
 * Computes dot products in parallel via OpenMP, partially sorts by score, and
 * returns the top-k indices sorted by their global index (not by score).
 *
 * @param query      Dense Eigen query vector.
 * @param dataset    Sparse dataset in CSR format containing the indexed items.
 * @param identified Vector of candidate global item indices to score.
 * @param k          Maximum number of results to return.
 * @return vector<uint> Up to k item indices sorted by global index.
 */
inline vector<uint> getTopKSparse(
    const Eigen::VectorXf &query,
    const SparseDataset &dataset,
    const std::vector<uint> &identified,
    uint k
) {
    if (identified.empty()) return {};
    vector<pair<float, uint>> scores;
    scores.reserve(identified.size());
        
    #pragma omp parallel
    {
        vector<pair<float, uint>> local_scores;
        #pragma omp for nowait
        for (size_t i = 0; i < identified.size(); ++i) {
            uint idx = identified[i];
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

/**
 * @brief Score a query against a candidate set and return the top-k item indices.
 *
 * Computes dot products in parallel via OpenMP, partially sorts by score, and
 * returns the top-k indices sorted by their global index (not by score).
 *
 * @param query      Dense Eigen query vector.
 * @param dataset    Sparse dataset in CSR format containing the indexed items.
 * @param identified Set of candidate global item indices to score.
 * @param k          Maximum number of results to return.
 * @return vector<uint> Up to k item indices sorted by global index.
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

/**
 * @brief Represents a contiguous run of items that share the same hash value.
 *
 * Used inside PoolInvertedIndex to map a hash value to a slice of the
 * sorted_item_indices_ array.
 */
struct HashBucket {
    /** @brief Hash value shared by all items in this bucket. */
    uint hash_val;
    /** @brief Index into sorted_item_indices_ where this bucket's items begin. */
    uint start_idx;
    /** @brief Number of items in this bucket. */
    uint num_items;
};

/**
 * @brief Identifies a single item candidate produced during the peeling algorithm.
 */
struct Candidate {
    /** @brief Index of the pool in which the item was identified. */
    uint pool_idx;
    /** @brief Global index of the identified item. */
    uint item_idx;
};

/**
 * @brief Score a query against a candidate set in a dense dataset and return the top-k indices.
 *
 * Uses Eigen dot products; does not use OpenMP (intended for dense/small matrices).
 * Returns up to k item indices sorted by global index.
 *
 * @param query      Dense Eigen query vector.
 * @param dataset    Row-major Eigen matrix where each row is a dataset item.
 * @param identified Set of candidate global item indices to score.
 * @param k          Maximum number of results to return.
 * @return vector<uint> Up to k item indices sorted by global index.
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
 * @brief Convert a 2-D numpy float32 array to a nested vector, optionally L2-normalizing each row.
 *
 * @param data_points_arr 2-D numpy array of shape (num_items, dimension).
 * @param normalize       If true, each row is divided by its L2 norm (rows with norm < 1e-9 are left unchanged).
 * @return vector<vector<float>> Row-major copy of the data, possibly normalized.
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
 * @brief Convert a 1-D numpy float32 array to a vector, optionally L2-normalizing it.
 *
 * @param query_arr 1-D numpy array representing a single data point.
 * @param normalize If true, the vector is divided by its L2 norm (vectors with norm < 1e-9 are left unchanged).
 * @return vector<float> Copy of the query, possibly normalized.
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
 * @brief SplitMix64 PRNG: maps a 64-bit integer to a pseudo-random 64-bit integer.
 *
 * Used for on-the-fly random weight generation in hashers to avoid storing
 * full projection matrices.
 *
 * @param x Input value (typically a seed or counter).
 * @return uint64_t Pseudo-random 64-bit output.
 */
inline uint64_t splitmix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return x;
}

/**
 * @brief Combine two 64-bit hash values into a single hash value.
 *
 * Used in MinHasher to merge multiple per-table hash values into one compound hash.
 *
 * @param item1 First hash value.
 * @param item2 Second hash value.
 * @return uint64_t Combined hash value.
 */
inline uint64_t combine_hashes(uint64_t item1, uint64_t item2) {
    return item1 * 0xC4DD05BF + item2 * 0x6C8702C9;
}

#endif /* D06ED106_555D_4C4D_8C5F_06CF6C90B5F0 */

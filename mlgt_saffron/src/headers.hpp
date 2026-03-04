#ifndef D06ED106_555D_4C4D_8C5F_06CF6C90B5F0
#define D06ED106_555D_4C4D_8C5F_06CF6C90B5F0

#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

typedef unsigned int uint;

const uint NUM_POOLS_COEFF = 400; // Increased to 400 for 1M+ points to ensure better isolation.
const uint POOLS_PER_ITEM = 3; // c_d; each item appears in c_d pools
const uint SIGNATURE_COEFF = 6; // c_L; num_signature_bits =  ceil(log2(n)) * c_L

using std::vector, std::array, std::pair;
using std::set, std::map, std::queue, std::unordered_map, std::unordered_set;
using std::cout, std::endl;
using std::optional, std::nullopt;

/**
 * @brief Efficiently scores a query against a set of identified items using Eigen.
 * 
 * Performs parallel inner product computations for all candidate items and uses 
 * an efficient selection algorithm (nth_element) to find the top K.
 * 
 * @param query The query vector (Eigen).
 * @param data The global data matrix containing the point vectors.
 * @param identified A set of item indices recovered by the peeling algorithm.
 * @param k The number of neighbors to return.
 * @return vector<uint> The k nearest neighbor indices among the identified set.
 */
inline vector<uint> getTopKEigen(
    const Eigen::VectorXf &query,
    const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> &data,
    const std::set<uint> &identified,
    uint k
) {
    if (identified.empty()) return {};
    vector<pair<float, uint>> scores;
    scores.reserve(identified.size());
    
    // Convert identifying set to a vector for parallelization
    vector<uint> indices(identified.begin(), identified.end());
    
    #pragma omp parallel
    {
        vector<pair<float, uint>> local_scores;
        #pragma omp for nowait
        for (size_t i = 0; i < indices.size(); ++i) {
            uint idx = indices[i];
            float dp = data.row(idx).dot(query);
            local_scores.push_back({dp, idx});
        }
        #pragma omp critical
        scores.insert(scores.end(), local_scores.begin(), local_scores.end());
    }

    auto cmp = [](const pair<float, uint> &a, const pair<float, uint> &b) {
        return a.first > b.first; // Max heap for descending order
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
 * @brief Normalizes a single data point to have unit L2 norm, if requested.
 * 
 * @param data_point_arr 1D numpy array of floats.
 * @param normalize If true, the output vector is normalized.
 * @return vector<float> The (possibly normalized) data point.
 */
vector<float> normalizeDataPoint(
    const pybind11::array_t<float> &data_point_arr, 
    bool normalize
) {
    auto r = data_point_arr.unchecked<1>();
    vector<float> data_point(r.shape(0), 0.0f);
    float norm_sq = 0;
    for (uint i = 0; i < r.shape(0); ++i) {
        norm_sq += r(i) * r(i);
    }
    float norm = sqrt(norm_sq);
    if (normalize && norm > 1e-9) {
        for (uint i = 0; i < r.shape(0); ++i) {
            data_point[i] = r(i) / norm;
        }
    } else {
        for (uint i = 0; i < r.shape(0); ++i) {
            data_point[i] = r(i);
        }
    }
    return data_point;
}

/**
 * @brief Normalizes an entire dataset to have unit L2 norm for each point, if requested.
 * 
 * @param data_points_arr 2D numpy array of floats.
 * @param normalize If true, each output vector is normalized.
 * @return vector<vector<float>> The (possibly normalized) dataset as a list of vectors.
 */
vector<vector<float>> normalizeDataset(
    const pybind11::array_t<float> &data_points_arr,
    bool normalize
) {
    auto r = data_points_arr.unchecked<2>();
    vector<vector<float>> data_points(
        r.shape(0), 
        vector<float>(r.shape(1), 0.0f)
    );
    for (uint i = 0; i < r.shape(0); ++i) {
        float norm_sq = 0;
        for (uint j = 0; j < r.shape(1); ++j) {
            norm_sq += r(i, j) * r(i, j);
        }
        float norm = sqrt(norm_sq);
        if (normalize && norm > 1e-9) {
            for (uint j = 0; j < r.shape(1); ++j) {
                data_points[i][j] = r(i, j) / norm;
            }
        } else {
            for (uint j = 0; j < r.shape(1); ++j) {
                data_points[i][j] = r(i, j);
            }
        }
    }
    return data_points;
}


/**
 * @brief A simple struct to represent hash buckets for debugging and analysis.
 */
struct HashBucket {
    uint hash_val; // The hash value corresponding to this bucket
    uint start_idx; // Starting index in the global doc_index_ for items in this bucket
    uint num_items; // Number of items that hash to this bucket
};


/**
 * @brief Represents a candidate item found in a pool.
 */
struct Candidate {
    uint pool_idx; // Index of the pool where the candidate was found
    uint item_idx; // Index of the candidate item within the pool
};



#endif /* D06ED106_555D_4C4D_8C5F_06CF6C90B5F0 */

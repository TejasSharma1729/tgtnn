#ifndef B6E5EB81_CEAF_4632_819C_6A9B8B992E8C
#define B6E5EB81_CEAF_4632_819C_6A9B8B992E8C

#include "headers.hpp"
#include "Saffron.hpp"
#include "BloomHashIndex.hpp"


/**
 * @brief An experimental SAFFRON variant that uses independent Bloom filters for every test.
 * 
 * BloomGroupTestingSaffron allocates a dedicated BloomHashIndex for every (pool, bit) pair.
 * Unlike the standard MLGT variant which shares a projection matrix, this variant 
 * ensures that noise is independent across every test bit, theoretically improving 
 * the resolution of doubletons at the cost of significantly higher indexing 
 * and search complexity.
 */
class BloomGroupTestingSaffron : public Saffron {
protected:
    vector<vector<BloomHashIndex>> pool_test_indices_; // [pool][bit]
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_eigen_;
    uint num_hashes_;
    uint hash_bits_;
    uint threshold_;
    uint dimension_;
    bool normalize_; 

public:
    /**
     * @brief Initializes the per-test Bloom Group Testing index.
     * 
     * @param data_points_arr Numpy array of data points [N, D].
     * @param num_neighbors Target sparsity (k) for SAFFRON.
     * @param num_hashes Number of hashes for each individual Bloom filter.
     * @param hash_bits bits per hash.
     * @param threshold Match threshold.
     * @param debug Debug level.
     * @param normalize Whether to L2-normalize vectors.
     */
    BloomGroupTestingSaffron(
        pybind11::array_t<float> data_points_arr,
        uint num_neighbors = 100,
        uint num_hashes = BLOOM_NUM_HASHES,
        uint hash_bits = BLOOM_HASH_BITS,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0,
        bool normalize = true
    ) : Saffron(data_points_arr.shape(0), num_neighbors, debug),
        num_hashes_(num_hashes),
        hash_bits_(hash_bits),
        threshold_(threshold),
        dimension_(data_points_arr.shape(1)),
        normalize_(normalize)
    {
        // 1. Data Prep
        auto r = data_points_arr.unchecked<2>();
        data_eigen_.resize(num_features_, dimension_);
        #pragma omp parallel for
        for (int i = 0; i < (int)num_features_; ++i) {
            float norm_sq = 0;
            for (int j = 0; j < (int)dimension_; ++j) {
                float val = r(i, j);
                data_eigen_(i, j) = val;
                norm_sq += val * val;
            }
            if (normalize_) {
                float norm = std::sqrt(norm_sq);
                if (norm > 1e-9) data_eigen_.row(i) /= norm;
            }
        }

        // 2. Build pool-test indices
        pool_test_indices_.resize(num_pools_);
        cout << "Building " << num_pools_ * signature_length_ << " specialized Bloom indices..." << endl;
        #pragma omp parallel for
        for (int p = 0; p < (int)num_pools_; ++p) {
            pool_test_indices_[p].resize(signature_length_);
            for (int b = 0; b < (int)signature_length_; ++b) {
                vector<uint> items;
                for (uint item_idx : pools_.pools_to_items[p]) {
                    if (getSignature(item_idx, signature_length_)[b]) {
                        items.push_back(item_idx);
                    }
                }
                // Each BloomHashIndex now derives from its OWN BloomHashFunction.
                pool_test_indices_[p][b] = BloomHashIndex(dimension_, data_eigen_, items, num_hashes, hash_bits, threshold);
            }
        }
    }

protected:
    /**
     * @brief Computes test residuals by querying the independent Bloom filters.
     * 
     * In this variant, every single test bit (pool, signature_bit) uses its own
     * independent Bloom filter to decide if a query vector should be 'active'
     * in that parity sum.
     * 
     * @param query_vec The normalized query vector.
     * @return vector<vector<bool>> The num_pools x signature_bits residual matrix.
     */
    inline vector<vector<bool>> getResiduals(const Eigen::VectorXf& query_vec) const {
        // No shared_hasher_ query_hashes calculation here.
        vector<vector<bool>> residuals(num_pools_, vector<bool>(signature_length_, false));
        
        #pragma omp parallel for collapse(2)
        for (int p = 0; p < (int)num_pools_; ++p) {
            for (int b = 0; b < (int)signature_length_; ++b) {
                // Each test checks if the query matches using its OWN private projection matrix.
                // This ensures independent noise per test bit.
                if (pool_test_indices_[p][b](query_vec)) {
                    residuals[p][b] = true; 
                }
            }
        }
        return residuals;
    }

public:
    /**
     * @brief Searches for nearest neighbors using the group testing peeling algorithm.
     * 
     * @param query_arr 1D numpy array representing the query point.
     * @return vector<uint> Indices of the top-k nearest neighbors found.
     */
    inline vector<uint> search(pybind11::array_t<float> query_arr) {
        Eigen::Map<const Eigen::VectorXf> q_raw(query_arr.data(), dimension_);
        Eigen::VectorXf query = q_raw;
        if (normalize_) {
            float norm = query.norm();
            if (norm > 1e-9) query /= norm;
        }

        vector<vector<bool>> residuals = getResiduals(query);
        set<uint> identified = peelingAlgorithm(residuals);
        return getTopKEigen(query, data_eigen_, identified, sparsity_);
    }

    /**
     * @brief Performs a search using the callable interface.
     * @param query_arr 1D numpy array.
     * @return vector<uint> Nearest neighbor indices.
     */
    inline vector<uint> operator()(pybind11::array_t<float> query_arr) {
        return search(query_arr);
    }
};




#endif /* B6E5EB81_CEAF_4632_819C_6A9B8B992E8C */

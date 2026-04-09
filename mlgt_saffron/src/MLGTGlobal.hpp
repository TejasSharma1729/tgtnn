#ifndef D38E46F8_MLGT_GLOBAL_SAFFRON_HPP
#define D38E46F8_MLGT_GLOBAL_SAFFRON_HPP

#include "headers.hpp"
#include "BloomHashFunction.hpp"
#include "Saffron.hpp"
#include "GlobalInvertedIndex.hpp"

/**
 * @brief Multi-Label Group Testing (MLGT) Global implementation for Dense data.
 * 
 * Uses a single Global Inverted Index to find matches across all items,
 * and recovers identified items efficiently.
 */
class MLGTGlobal : public Saffron {
protected:
    BloomHashFunction shared_hasher_; 
    GlobalInvertedIndex global_index_;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_eigen_;
    uint num_hashes_;
    uint hash_bits_;
    uint threshold_;
    uint dimension_;
    bool normalize_; 

public:
    /**
     * @brief Initializes the MLGTGlobal index.
     */
    MLGTGlobal(
        pybind11::array_t<float> data_points_arr,
        uint num_neighbors = 100,
        uint num_hashes = BLOOM_NUM_HASHES,
        uint hash_bits = BLOOM_HASH_BITS,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0,
        bool normalize = true
    ) : Saffron(data_points_arr.shape(0), num_neighbors, debug),
        shared_hasher_(data_points_arr.shape(1), num_hashes, hash_bits, threshold, debug),
        num_hashes_(num_hashes),
        hash_bits_(hash_bits),
        threshold_(threshold),
        dimension_(data_points_arr.shape(1)),
        normalize_(normalize)
    {
        // Convert to Eigen Matrix
        auto r = data_points_arr.unchecked<2>();
        data_eigen_.resize(r.shape(0), r.shape(1));
        #pragma omp parallel for
        for (int i = 0; i < (int)r.shape(0); ++i) {
            float norm_sq = 0;
            for (int j = 0; j < (int)r.shape(1); ++j) {
                float val = r(i, j);
                data_eigen_(i, j) = val;
                norm_sq += val * val;
            }
            if (normalize_) {
                float norm = std::sqrt(norm_sq);
                if (norm > 1e-9) data_eigen_.row(i) /= norm;
            }
        }

        // Pre-calculate hashes for all items
        vector<vector<uint>> all_hashes(num_features_);
        #pragma omp parallel for
        for (int item_idx = 0; item_idx < (int)num_features_; ++item_idx) {
            all_hashes[item_idx] = shared_hasher_(data_eigen_.row(item_idx));
        }

        // Build one single global index
        global_index_ = GlobalInvertedIndex(num_hashes_, threshold_);
        vector<uint> all_item_indices(num_features_);
        std::iota(all_item_indices.begin(), all_item_indices.end(), 0);
        global_index_.build(all_hashes, all_item_indices);
        
        if (debug_ > 0) {
            cout << "[MLGTGlobal] Built 1 global index." << endl;
        }
    }

    ~MLGTGlobal() = default;

public:
    /**
     * @brief Performs a nearest neighbor search.
     * 
     * @param query_arr The query vector (numpy array).
     * @return vector<uint> Top K item indices.
     */
    inline vector<uint> search(pybind11::array_t<float> query_arr) {
        Eigen::Map<const Eigen::VectorXf> q_raw(query_arr.data(), dimension_);
        Eigen::VectorXf query = q_raw;
        if (normalize_) {
            float norm = query.norm();
            if (norm > 1e-9) query /= norm;
        }

        vector<uint> query_hashes = shared_hasher_(query);
        vector<uint> matched_items = global_index_.get_matches(query_hashes);
        
        std::set<uint> identified(matched_items.begin(), matched_items.end());
        return getTopKEigen(query, data_eigen_, identified, sparsity_);
    }

    /**
     * @brief Performs a search using a callable interface.
     * @param query_arr The query vector.
     * @return vector<uint> Top K item indices.
     */
    inline vector<uint> operator()(pybind11::array_t<float> query_arr) {
        return search(query_arr);
    }
};

#endif /* D38E46F8_MLGT_GLOBAL_SAFFRON_HPP */

#ifndef B2CD5C8C_40DC_4EB3_A0FF_E8E6C37E465D
#define B2CD5C8C_40DC_4EB3_A0FF_E8E6C37E465D

#include "headers.hpp"
#include "BloomHashFunction.hpp"
#include "MinHasher.hpp"
#include "WeightedMinHasher.hpp"
#include "Saffron.hpp"
#include "GlobalInvertedIndex.hpp"


/**
 * @brief Generic Sparse Multi-Label Group Testing (MLGT) Saffron implementation.
 * 
 * This class implements a sub-linear time nearest neighbor search algorithm based 
 * on Group Testing and the SAFFRON recovery scheme. It uses a templated Hasher 
 * to project vectors into a discrete space, and a GlobalInvertedIndex to 
 * perform thresholded lookups across multiple pools.
 * 
 * @tparam Hasher A type that satisfies the HasherType concept (inherits from BaseHasher).
 */
template <HasherType Hasher>
class MLGTSaffron : public Saffron {
public:
    /** @brief Alias for the template hasher type. */
    using HasherAlias = Hasher;
protected:
    /** @brief The hashing engine used for indexing and queries. */
    Hasher shared_hasher_; 
    /** @brief The single global inverted index. */
    GlobalInvertedIndex global_index_;
    /** @brief The indexed sparse dataset in CSR format. */
    SparseDataset dataset_;
    /** @brief XOR-sum signatures used by SAFFRON for recovery. */
    vector<vector<bool>> item_signatures_;
    /** @brief Cached number of hashes from the hasher. */
    uint num_hashes_;
    /** @brief Minimum match threshold for candidate identification. */
    uint threshold_;
    /** @brief Dimensionality of the data. */
    uint dimension_;
    /** @brief Whether to L2-normalize vectors during search. */
    bool normalize_; 

public:
    /**
     * @brief Construct a new MLGTSaffron index and populate internal structures.
     * 
     * @param data_arr 1D numpy array of float32 values (CSR data).
     * @param indices_arr 1D numpy array of uint32 indices (CSR indices).
     * @param indptr_arr 1D numpy array of uint64 pointers (CSR indptr).
     * @param num_cols Total number of columns (features).
     * @param hasher An instance of the chosen Hasher.
     * @param num_neighbors The number of nearest neighbors to recover (target sparsity).
     * @param threshold Identification match threshold.
     * @param debug Debug verbosity level.
     * @param normalize If true, L2-normalizes vectors before processing.
     */
    MLGTSaffron(
        pybind11::array_t<float> data_arr,
        pybind11::array_t<uint32_t> indices_arr,
        pybind11::array_t<uint64_t> indptr_arr,
        uint32_t num_cols,
        Hasher hasher,
        uint num_neighbors = 100,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0,
        bool normalize = true
    ) : Saffron(indptr_arr.shape(0) - 1, num_neighbors, debug),
        shared_hasher_(hasher),
        num_hashes_(hasher.num_hashes),
        threshold_(threshold),
        dimension_(num_cols),
        normalize_(normalize)
    {
        // Populate SparseDataset
        auto r_data = data_arr.unchecked<1>();
        auto r_indices = indices_arr.unchecked<1>();
        auto r_indptr = indptr_arr.unchecked<1>();

        dataset_.data.assign(r_data.data(0), r_data.data(0) + r_data.shape(0));
        dataset_.indices.assign(r_indices.data(0), r_indices.data(0) + r_indices.shape(0));
        dataset_.indptr.assign(r_indptr.data(0), r_indptr.data(0) + r_indptr.shape(0));
        dataset_.num_rows = num_features_;
        dataset_.num_cols = num_cols;

        if (normalize_) {
            #pragma omp parallel for
            for (int i = 0; i < (int)num_features_; ++i) {
                float norm_sq = 0;
                for (uint64_t j = dataset_.indptr[i]; j < dataset_.indptr[i+1]; ++j) {
                    norm_sq += dataset_.data[j] * dataset_.data[j];
                }
                float norm = std::sqrt(norm_sq);
                if (norm > 1e-9) {
                    for (uint64_t j = dataset_.indptr[i]; j < dataset_.indptr[i+1]; ++j) {
                        dataset_.data[j] /= norm;
                    }
                }
            }
        }

        // Pre-calculate signatures
        item_signatures_.resize(num_features_);
        #pragma omp parallel for
        for (int i = 0; i < (int)num_features_; ++i) {
            item_signatures_[i] = getSignature(i, signature_length_);
        }

        // Pre-calculate hashes
        vector<vector<uint>> all_hashes(num_features_);
        #pragma omp parallel for
        for (int i = 0; i < (int)num_features_; ++i) {
            all_hashes[i] = shared_hasher_(
                dataset_.row_data(i), 
                dataset_.row_indices(i), 
                dataset_.nnz(i)
            );
        }

        // Build one single global index
        global_index_ = GlobalInvertedIndex(num_hashes_, threshold_);
        vector<uint> all_item_indices(num_features_);
        std::iota(all_item_indices.begin(), all_item_indices.end(), 0);
        global_index_.build(all_hashes, all_item_indices);
        
        if (debug_ > 0) {
            cout << "[MLGTSaffron] Built 1 global index." << endl;
        }
    }

    ~MLGTSaffron() = default;

protected:
    /**
     * @brief Computes residuals for a query by querying the pool-specific inverted indices.
     * 
     * @param query_vec The dense query vector.
     * @return vector<vector<bool>> Residual matrix (num_pools x signature_bits).
     */
    inline vector<vector<bool>> getResiduals(const Eigen::VectorXf& query_vec) const {
        vector<uint> query_hashes = shared_hasher_(query_vec);
        vector<vector<bool>> residuals(num_pools_, vector<bool>(signature_length_, false));
        
        vector<uint> matched_items = global_index_.get_matches(query_hashes);
        
        for (uint global_item_idx : matched_items) {
            const vector<bool>& sig = item_signatures_[global_item_idx];
            for (uint p : pools_.items_to_pools[global_item_idx]) {
                for (uint b = 0; b < signature_length_; ++b) {
                    if (sig[b]) {
                        residuals[p][b] = !residuals[p][b]; 
                    }
                }
            }
        }
        return residuals;
    }

public:
    /**
     * @brief Searches for the top K nearest neighbors.
     * 
     * @param query_arr 1D numpy array representing the query point.
     * @return vector<uint> Indices of the top neighbors found.
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
        return getTopKSparse(query, dataset_, identified, sparsity_);
    }

    /**
     * @brief Callable interface for search.
     * 
     * @param query_arr 1D numpy array.
     * @return vector<uint> Indices of the top neighbors.
     */
    inline vector<uint> operator()(pybind11::array_t<float> query_arr) {
        return search(query_arr);
    }
};

using MLGTSaffronBloom = MLGTSaffron<BloomHashFunction>;
using MLGTSaffronMinHash = MLGTSaffron<MinHasher>;
using MLGTSaffronWeightedMinHash = MLGTSaffron<WeightedMinHasher>;
using MLGTSaffronSparseSRP = MLGTSaffron<SparseSRPHasher>;
using MLGTSaffronDenseSRP = MLGTSaffron<DenseSRPHasher>;


#endif /* B2CD5C8C_40DC_4EB3_A0FF_E8E6C37E465D */

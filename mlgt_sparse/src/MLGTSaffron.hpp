#ifndef B2CD5C8C_40DC_4EB3_A0FF_E8E6C37E465D
#define B2CD5C8C_40DC_4EB3_A0FF_E8E6C37E465D

#include "headers.hpp"
#include "BloomHashFunction.hpp"
#include "Saffron.hpp"
#include "GlobalInvertedIndex.hpp"


/**
 * @brief Generic Sparse Multi-Label Group Testing (MLGT) Saffron implementation.
 */
template <HasherType Hasher>
class MLGTSaffron : public Saffron {
public:
    using HasherAlias = Hasher;
protected:
    Hasher shared_hasher_; 
    vector<GlobalInvertedIndex> pool_indices_;
    SparseDataset dataset_;
    vector<vector<bool>> item_signatures_;
    uint num_hashes_;
    uint threshold_;
    uint dimension_;
    bool normalize_; 

public:
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

        // Build one index PER POOL
        pool_indices_.resize(num_pools_);
        #pragma omp parallel for
        for (int p = 0; p < (int)num_pools_; ++p) {
            pool_indices_[p] = GlobalInvertedIndex(num_hashes_, threshold_);
            
            vector<vector<uint>> pool_hashes;
            pool_hashes.reserve(pools_.pools_to_items[p].size());
            for (uint global_idx : pools_.pools_to_items[p]) {
                pool_hashes.push_back(all_hashes[global_idx]);
            }
            pool_indices_[p].build(pool_hashes, pools_.pools_to_items[p]);
        }
        
        if (debug_ > 0) {
            cout << "[MLGTSaffron] Built " << num_pools_ << " pool indices." << endl;
        }
    }

    ~MLGTSaffron() = default;

protected:
    inline vector<vector<bool>> getResiduals(const Eigen::VectorXf& query_vec) const {
        vector<uint> query_hashes = shared_hasher_(query_vec);
        vector<vector<bool>> residuals(num_pools_, vector<bool>(signature_length_, false));
        
        #pragma omp parallel for
        for (int p = 0; p < (int)num_pools_; ++p) {
            vector<uint> matched_items = pool_indices_[p].get_matches(query_hashes);
            for (uint global_item_idx : matched_items) {
                const vector<bool>& sig = item_signatures_[global_item_idx];
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

    inline vector<uint> operator()(pybind11::array_t<float> query_arr) {
        return search(query_arr);
    }
};

using MLGTSaffronBloom = MLGTSaffron<BloomHashFunction>;
using MLGTSaffronMinHash = MLGTSaffron<MinHasher>;
using MLGTSaffronSparseSRP = MLGTSaffron<SparseSRPHasher>;
using MLGTSaffronDenseSRP = MLGTSaffron<DenseSRPHasher>;


#endif /* B2CD5C8C_40DC_4EB3_A0FF_E8E6C37E465D */

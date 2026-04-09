#ifndef JE9B1C8B_7F3A_4C9B_9F2D_5E6C8A37E123
#define JE9B1C8B_7F3A_4C9B_9F2D_5E6C8A37E123

#include "headers.hpp"
#include "BloomHashFunction.hpp"
#include "MinHasher.hpp"
#include "WeightedMinHasher.hpp"
#include "SparseSRPHasher.hpp"
#include "DenseSRPHasher.hpp"
#include "GlobalInvertedIndex.hpp"


/**
 * @brief Abstract base class for all MLGT variants.
 *
 * Adds a pure-virtual  search interface so that Python (and C++) code can hold 
 * a polymorphic pointer to any concrete variant (MinHash, Bloom, SRP, …) 
 * through this type.
 */
class MLGTGlobalBase {
protected:
    /** @brief Number of items in the dataset (rows). */
    uint num_rows_;
    /** @brief Number of nearest neighbors to return per query. */
    uint num_neighbors_;
    /** @brief Number of SAFFRON measurement pools. */
    uint num_pools_;
    /** @brief Number of pools each item is assigned to. */
    uint pools_per_item_;
    /** @brief Debug verbosity level. */
    int debug_;
public:
    /** @brief Construct a new MLGTGlobalBase object with common parameters. */
    MLGTGlobalBase(uint32_t num_rows, uint32_t num_neighbors, uint32_t num_pools, uint32_t pools_per_item, int debug)
        : num_rows_(num_rows), num_neighbors_(num_neighbors), num_pools_(num_pools), pools_per_item_(pools_per_item), debug_(debug) {}
    virtual ~MLGTGlobalBase() = default;

    /**
     * @brief Search for approximate nearest neighbours of a query vector.
     * @param query_arr 1-D float32 numpy array of length num_cols.
     * @return vector<uint> Up to sparsity_ item indices (sorted by global index).
     */
    virtual vector<uint> search(pybind11::array_t<float> query_arr) = 0;

    /** @brief Callable alias for search(). */
    virtual vector<uint> operator()(pybind11::array_t<float> query_arr) = 0;
};


/**
 * @brief Generic Sparse Multi-Label Group Testing (MLGT) implementation.
 * 
 * This class implements a sub-linear time nearest neighbor search algorithm based 
 * on Group Testing and the SAFFRON recovery scheme. It uses a templated Hasher 
 * to project vectors into a discrete space, and a GlobalInvertedIndex to 
 * perform thresholded lookups across multiple pools.
 * 
 * @tparam Hasher A type that satisfies the HasherType concept (inherits from BaseHasher).
 */
template <HasherType Hasher>
class MLGTGlobal : public MLGTGlobalBase {
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
     * @brief Construct a new MLGT index and populate internal structures.
     * 
     * @param data_arr 1D numpy array of float32 values (CSR data).
     * @param indices_arr 1D numpy array of uint32 indices (CSR indices).
     * @param indptr_arr 1D numpy array of uint64 pointers (CSR indptr).
     * @param num_cols Total number of columns (features).
     * @param hasher An instance of the chosen Hasher.
     * @param num_neighbors The number of nearest neighbors to recover (target sparsity).
     * @param num_pools The number of  pools.
     * @param threshold Identification match threshold.
     * @param debug Debug verbosity level.
     * @param normalize If true, L2-normalizes vectors before processing.
     */
    MLGTGlobal(
        pybind11::array_t<float> data_arr,
        pybind11::array_t<uint32_t> indices_arr,
        pybind11::array_t<uint64_t> indptr_arr,
        uint32_t num_cols,
        Hasher hasher,
        uint num_neighbors = 100,
        uint num_pools = 0,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0,
        bool normalize = true
    ) : MLGTGlobalBase(indptr_arr.shape(0) - 1, num_neighbors, num_pools, POOLS_PER_ITEM, debug),
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
        dataset_.num_rows = num_rows_;
        dataset_.num_cols = num_cols;

        if (normalize_) {
            #pragma omp parallel for
            for (int i = 0; i < (int)num_rows_; ++i) {
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

        // Pre-calculate hashes
        vector<vector<uint>> all_hashes(num_rows_);
        #pragma omp parallel for
        for (int i = 0; i < (int)num_rows_; ++i) {
            all_hashes[i] = shared_hasher_(
                dataset_.row_data(i), 
                dataset_.row_indices(i), 
                dataset_.nnz(i)
            );
        }

        // Build one single global index
        global_index_ = GlobalInvertedIndex(num_hashes_, threshold_);
        vector<uint> all_item_indices(num_rows_);
        std::iota(all_item_indices.begin(), all_item_indices.end(), 0);
        global_index_.build(all_hashes, all_item_indices);
        
        if (debug_ > 0) {
            cout << "[MLGTGlobal] Built 1 global index." << endl;
        }
    }

    ~MLGTGlobal() = default;

public:
    /**
     * @brief Searches for the top K nearest neighbors.
     * 
     * @param query_arr 1D numpy array representing the query point.
     * @return vector<uint> Indices of the top neighbors found.
     */
    inline vector<uint> search(pybind11::array_t<float> query_arr) override {
        auto r_query = query_arr.unchecked<1>();
        Eigen::Map<const Eigen::VectorXf> query(r_query.data(0), r_query.shape(0));
        vector<uint> query_hashes = shared_hasher_(query);
        vector<uint> matched_items = global_index_.get_matches(query_hashes);
        return getTopKSparse(query, dataset_, matched_items, num_neighbors_);
    }

    /**
     * @brief Callable interface for search.
     * 
     * @param query_arr 1D numpy array.
     * @return vector<uint> Indices of the top neighbors.
     */
    inline vector<uint> operator()(pybind11::array_t<float> query_arr) override {
        return search(query_arr);
    }
};

using MLGTGlobalBloom = MLGTGlobal<BloomHashFunction>;
using MLGTGlobalMinHash = MLGTGlobal<MinHasher>;
using MLGTGlobalWeightedMinHash = MLGTGlobal<WeightedMinHasher>;
using MLGTGlobalSparseSRP = MLGTGlobal<SparseSRPHasher>;
using MLGTGlobalDenseSRP = MLGTGlobal<DenseSRPHasher>;


#endif /* JE9B1C8B_7F3A_4C9B_9F2D_5E6C8A37E123 */

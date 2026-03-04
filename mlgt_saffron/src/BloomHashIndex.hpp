#ifndef B73AB2D1_1964_40F9_9FC6_C0C2EB143F0F
#define B73AB2D1_1964_40F9_9FC6_C0C2EB143F0F

#include "headers.hpp"
#include "BloomHashFunction.hpp"
#include "GlobalInvertedIndex.hpp"


/**
 * @brief A high-performance Bloom Filter index for high-dimensional vector search.
 * 
 * BloomHashIndex combines Locality Sensitive Hashing (LSH) with a memory-efficient 
 * inverted index (GlobalInvertedIndex). It allows for rapid candidate identification 
 * by projecting vectors into a discrete hash space and performing binary-search 
 * lookups on sorted buckets.
 */
class BloomHashIndex : public BloomHashFunction {
public:
    GlobalInvertedIndex index_;
    uint num_items_;

    /**
     * @brief Empty constructor for BloomHashIndex.
     */
    BloomHashIndex() : BloomHashFunction(0, 0, 0, 0, 0), num_items_(0) {}

    /**
     * @brief Specialized constructor for building a Bloom filter from a full dataset.
     * 
     * @param data_points_arr Numpy array of data points [N, D].
     * @param num_hashes Number of independent compound hash functions.
     * @param num_bits Number of bits packed into each hash value.
     * @param threshold Number of hash matches required for inclusion in candidates.
     * @param debug Debug verbosity level.
     */
    BloomHashIndex(
        pybind11::array_t<float> data_points_arr,
        uint num_hashes = BLOOM_NUM_HASHES,
        uint num_bits = BLOOM_HASH_BITS,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0
    ) : BloomHashFunction(data_points_arr.shape(1), num_hashes, num_bits, threshold, debug),
        index_(num_hashes, threshold) {
        
        num_items_ = data_points_arr.shape(0);
        
        auto r = data_points_arr.unchecked<2>();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> data_eigen(num_items_, dimension_);
        #pragma omp parallel for
        for (int i = 0; i < (int)num_items_; ++i) {
            for (int j = 0; j < (int)dimension_; ++j) {
                data_eigen(i, j) = r(i, j);
            }
        }

        vector<vector<uint>> all_hashes(num_items_);
        #pragma omp parallel for
        for (int i = 0; i < (int)num_items_; ++i) {
            all_hashes[i] = BloomHashFunction::operator()(data_eigen.row(i));
        }
        
        vector<uint> item_indices(num_items_);
        std::iota(item_indices.begin(), item_indices.end(), 0);
        index_.build(all_hashes, item_indices);
    }

    /**
     * @brief Specialized constructor for indexing a specific subset of items.
     * 
     * @param dimension Input dimensionality.
     * @param data_eigen The global data matrix containing the point vectors.
     * @param item_indices Sub-indices of the items to include in this specific index.
     * @param num_hashes Hash count.
     * @param num_bits Bits per hash.
     * @param threshold Recovery threshold.
     * @param debug Debug level.
     */
    BloomHashIndex(
        uint dimension,
        const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& data_eigen,
        const vector<uint>& item_indices,
        uint num_hashes = BLOOM_NUM_HASHES,
        uint num_bits = BLOOM_HASH_BITS,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0
    ) : BloomHashFunction(dimension, num_hashes, num_bits, threshold, debug),
        index_(num_hashes, threshold) {
        
        num_items_ = item_indices.size();
        if (num_items_ == 0) return;

        vector<vector<uint>> subset_hashes(num_items_);
        #pragma omp parallel for
        for (int i = 0; i < (int)num_items_; ++i) {
            subset_hashes[i] = BloomHashFunction::operator()(data_eigen.row(item_indices[i]));
        }
        index_.build(subset_hashes, item_indices);
    }

    /**
     * @brief Returns global item indices that match the pre-computed query hashes.
     * @param query_hashes Pre-computed Bloom signatures for the query vector.
     * @return vector<uint> Candidate item indices.
     */
    inline vector<uint> get_matches(const vector<uint> &query_hashes) const {
        return index_.get_matches(query_hashes);
    }

    /**
     * @brief Boolean check for membership (true if at least one candidate matches).
     * @param query_hashes Pre-computed Bloom signatures for the query vector.
     * @return true if matches found.
     */
    inline bool matches(const vector<uint>& query_hashes) const {
        return !index_.get_matches(query_hashes).empty();
    }

    /**
     * @brief Performs a membership query for a single Eigen vector.
     * @param query The query vector (Eigen).
     * @return true if the vector is likely in the indexed set.
     */
    inline bool operator() (const Eigen::VectorXf &query) const {
        return matches(BloomHashFunction::operator()(query));
    }
};



#endif /* B73AB2D1_1964_40F9_9FC6_C0C2EB143F0F */

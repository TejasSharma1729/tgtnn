#ifndef AC616250_4435_429E_9849_574892510586
#define AC616250_4435_429E_9849_574892510586

#include "headers.hpp"


/**
 * @brief Pool-local inverted index used by MLGTSaffron for thresholded hash lookups.
 *
 * Stores a compact sorted inverted index for one pool's subset of items.
 * At query time it counts how many of the query's hash values match each item
 * and XOR-accumulates the SAFFRON signatures of items that meet the match
 * threshold.  The resulting binary residual vector is consumed by the peeling
 * algorithm in Saffron::peelingAlgorithm().
 */
class PoolInvertedIndex {
private:
    /** @brief Contiguous storage of item indices sorted by (hash-function, hash-value). */
    vector<uint16_t> sorted_item_indices_;
    /** @brief Flat list of hash buckets across all hash functions. */
    vector<HashBucket> buckets_;
    /** @brief Exclusive prefix-sum offsets into `buckets_` per hash function; length = num_hashes_ + 1. */
    vector<uint> bucket_offsets_;
    /** @brief Global indices of the items assigned to this pool. */
    vector<uint> pool_item_indices_;
    /** @brief Per-pool SAFFRON signatures; element [p] is the signature of the p-th in-pool item. */
    vector<vector<bool>> item_signatures_;
    /** @brief Number of items assigned to this pool. */
    uint pool_size_;
    /** @brief Number of independent hash functions indexed. */
    uint num_hashes_;
    /** @brief Minimum number of hash matches required to include an item in the residual XOR. */
    uint threshold_;
    /** @brief Length (in bits) of each SAFFRON signature stored in item_signatures_. */
    uint signature_length_;

public:
    /**
     * @brief Default constructor; creates an empty, uninitialized index.
     *        Call build() before issuing any queries.
     */
    PoolInvertedIndex() : num_hashes_(0), threshold_(0) {}

    /**
     * @brief Construct a PoolInvertedIndex with the given parameters.
     *        Call build() to populate the index before querying.
     *
     * @param num_hashes Number of independent hash functions used during indexing.
     * @param threshold  Minimum hash-match count for an item to contribute to the residual.
     */
    PoolInvertedIndex(uint num_hashes, uint threshold)
        : num_hashes_(num_hashes), threshold_(threshold) {}

    /**
     * @brief Populate the inverted index from pre-computed hash values and signatures.
     *
     * Iterates over the items assigned to this pool, groups them by their hash value
     * for each hash function, and records the sorted buckets.  Also caches the SAFFRON
     * signature for each in-pool item.
     *
     * @param item_hashes       2-D vector [num_items][num_hashes_] of precomputed hash values
     *                          for the entire dataset (indexed by global item index).
     * @param pool_item_indices Global indices of the items that belong to this pool.
     * @param item_signatures   SAFFRON signatures for the entire dataset; indexed by global item index.
     */
    void build(
        const vector<vector<uint>>& item_hashes,
        const vector<uint>& pool_item_indices,
        const vector<vector<bool>>& item_signatures
    ) {
        if (item_hashes.empty() || pool_item_indices.empty() || item_signatures.empty()) {
            return;
        }
        uint N = item_hashes.size();
        pool_size_ = pool_item_indices.size();
        pool_item_indices_ = pool_item_indices;
        item_signatures_.resize(pool_size_);
        signature_length_ = item_signatures[0].size();
        
        for (uint pIdx = 0; pIdx < pool_size_; pIdx++) {
            item_signatures_[pIdx] = item_signatures[pool_item_indices[pIdx]];
        }

        sorted_item_indices_.resize(pool_size_ * num_hashes_);
        bucket_offsets_.resize(1, 0);

        for (uint hIdx = 0; hIdx < num_hashes_; ++hIdx) {
            uint baseOffset = hIdx * pool_size_;
            vector<pair<uint, uint>> hash_and_indices;
            for (uint pIdx = 0; pIdx < pool_item_indices.size(); pIdx++) {
                uint fIdx = pool_item_indices[pIdx];
                hash_and_indices.push_back({item_hashes[fIdx][hIdx], pIdx});
            }
            std::sort(hash_and_indices.begin(), hash_and_indices.end());
            
            uint current_hash = hash_and_indices[0].first;
            uint startIdx = 0;
            uint i = 0;

            while (i < hash_and_indices.size()) {
                uint hash_val = hash_and_indices[i].first;
                uint item_idx = hash_and_indices[i].second;
                if (hash_val != current_hash) {
                    buckets_.push_back({current_hash, baseOffset + startIdx, i - startIdx});
                    current_hash = hash_val;
                    startIdx = i;
                }
                sorted_item_indices_[baseOffset + i] = item_idx;
                i++;
            }

            buckets_.push_back({current_hash, baseOffset + startIdx, i - startIdx});
            bucket_offsets_.push_back(buckets_.size());
        }
    }

    /**
     * @brief Compute the XOR-sum residual for this pool given a query's hash values.
     *
     * For each hash function, binary-searches the bucket array for matching hash values and
     * increments per-item match counters.  Once an item's counter reaches `threshold_`, its
     * SAFFRON signature is XOR-accumulated into the running residual.  Returns early after
     * two items have been identified (doubleton-variant optimisation).
     *
     * @param query_hashes Vector of length `num_hashes_` containing the query's hash values
     *                     as produced by the corresponding Hasher.
     * @return vector<bool> XOR-accumulated binary residual of length `signature_length_`.
     */
    vector<bool> get_residual(const vector<uint> &query_hashes) const {
        if (pool_size_ == 0 || signature_length_ == 0) return {};
        
        vector<uint16_t> matches(pool_size_, 0);
        vector<bool> residual(signature_length_, false);
        uint num_matches = 0;
        // Stop at 2 matches (above threshold_) for doubletons variant.

        for (uint hIdx = 0; hIdx < num_hashes_; hIdx++) {
            const uint hash_val = query_hashes[hIdx];
            int l = bucket_offsets_[hIdx];
            int r = bucket_offsets_[hIdx + 1] - 1;
            int bucket = -1;
            
            while (r >= l) {
                int m = (l + r) / 2;
                if (buckets_[m].hash_val == hash_val) {
                    bucket = m;
                    break;
                } else if (buckets_[m].hash_val > hash_val) {
                    r = m - 1;
                } else {
                    l = m + 1;
                }
            }
            if (bucket == -1) {
                continue;
            }

            for (uint pIdx = 0; pIdx < buckets_[bucket].num_items; pIdx++) {
                uint in_pool_idx = sorted_item_indices_[buckets_[bucket].start_idx + pIdx];
                matches[in_pool_idx]++;
                if (matches[in_pool_idx] != threshold_) {
                    continue;
                }
                for (uint b = 0; b < item_signatures_[in_pool_idx].size(); b++) {
                    residual[b] = residual[b] ^ item_signatures_[in_pool_idx][b];
                }
                if (++num_matches == 2) {
                    return residual;
                }
            }
        }
        return residual;
    }

    /**
     * @brief Callable interface; delegates to get_residual().
     *
     * @param query_hashes Vector of length `num_hashes_` containing the query's hash values.
     * @return vector<bool> XOR-accumulated binary residual of length `signature_length_`.
     */
    vector<bool> operator()(const vector<uint> &query_hashes) const {
        return get_residual(query_hashes);
    }

    /** @brief Returns the number of hash functions this index was built with. */
    inline uint num_hashes() const { return num_hashes_; }

    /** @brief Returns the number of items assigned to this pool. */
    inline uint pool_size() const { return pool_size_; }

    /** @brief Returns the minimum hash-match threshold for residual accumulation. */
    inline uint threshold() const { return threshold_; }
};


#endif // AC616250_4435_429E_9849_574892510586
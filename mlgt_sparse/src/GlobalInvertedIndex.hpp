#ifndef B827CE93_864C_4295_A4D3_0F4800283341
#define B827CE93_864C_4295_A4D3_0F4800283341

#include "headers.hpp"

/**
 * @brief Memory-efficient inverted index using sorted buckets.
 * 
 * Provides an efficient way to index items by their hash values and retrieve 
 * items that match at least `threshold` of a query's hashes. The index uses 
 * a contiguous array of sorted item indices to maximize cache locality.
 */
class GlobalInvertedIndex {
private:
    /** @brief CONTIGUOUS storage of item indices, sorted by their hash values. */
    vector<uint> sorted_item_indices_;
    /** @brief Maps each unique hash value to its bucket in sorted_item_indices_. */
    vector<HashBucket> buckets_;
    /** @brief Number of hash values expected per item. */
    uint num_hashes_;
    /** @brief Minimum match threshold. */
    uint threshold_;

public:
    /**
     * @brief Empty constructor.
     */
    GlobalInvertedIndex() : num_hashes_(0), threshold_(0) {}

    /**
     * @brief Construct a new Global Inverted Index.
     * @param num_hashes Number of compound hashes per item.
     * @param threshold Minimum number of matches for a candidate.
     */
    GlobalInvertedIndex(uint num_hashes, uint threshold) 
        : num_hashes_(num_hashes), threshold_(threshold) {}

    /**
     * @brief Builds the inverted index from a set of hashes and global item indices.
     * 
     * @param item_hashes 2D vector [num_items, num_hashes] of hash values.
     * @param global_item_indices The global indices of the items being indexed.
     */
    void build(const vector<vector<uint>>& item_hashes, const vector<uint>& global_item_indices) {
        if (item_hashes.empty()) return;
        uint n = item_hashes.size();
        
        // 1. Flatten all hashes into a list of (hash, global_item_idx)
        vector<pair<uint, uint>> flattened;
        flattened.reserve(n * num_hashes_);
        for (uint i = 0; i < n; ++i) {
            for (uint h = 0; h < num_hashes_; ++h) {
                flattened.push_back({item_hashes[i][h], global_item_indices[i]});
            }
        }

        // 2. Sort by hash value
        std::sort(flattened.begin(), flattened.end());

        // 3. Populate sorted_item_indices_ and buckets_
        sorted_item_indices_.resize(flattened.size());
        buckets_.clear();

        if (flattened.empty()) return;

        uint current_hash = flattened[0].first;
        uint start_idx = 0;

        for (uint i = 0; i < flattened.size(); ++i) {
            sorted_item_indices_[i] = flattened[i].second;
            if (flattened[i].first != current_hash) {
                buckets_.push_back({current_hash, start_idx, i - start_idx});
                current_hash = flattened[i].first;
                start_idx = i;
            }
        }
        buckets_.push_back({current_hash, start_idx, (uint)flattened.size() - start_idx});
    }

    /**
     * @brief Retrieves all items that match at least `threshold_` of the query hashes.
     * 
     * @param query_hashes A vector of length `num_hashes_` containing query hashes.
     * @return vector<uint> A list of candidate item indices.
     */
    vector<uint> get_matches(const vector<uint>& query_hashes) const {
        if (buckets_.empty()) return {};
        
        unordered_map<uint, uint> counts;
        for (uint q_h : query_hashes) {
            // Binary search for the hash bucket
            auto it = std::lower_bound(buckets_.begin(), buckets_.end(), q_h, 
                [](const HashBucket& b, uint val) { return b.hash_val < val; });
            
            if (it != buckets_.end() && it->hash_val == q_h) {
                for (uint i = 0; i < it->num_items; ++i) {
                    counts[sorted_item_indices_[it->start_idx + i]]++;
                }
            }
        }

        vector<uint> matches;
        for (auto const& [item_idx, count] : counts) {
            if (count >= threshold_) {
                matches.push_back(item_idx);
            }
        }
        return matches;
    }

    /**
     * @brief Returns the number of hashes the index expects.
     * @return uint Hash count.
     */
    inline uint num_hashes() const { return num_hashes_; }
};

#endif /* B827CE93_864C_4295_A4D3_0F4800283341 */

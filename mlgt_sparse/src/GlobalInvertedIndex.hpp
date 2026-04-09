#ifndef B827CE93_864C_4295_A4D3_0F4800283341
#define B827CE93_864C_4295_A4D3_0F4800283341

#include "headers.hpp"
#include <unordered_map>
#include <cstdint>

/**
 * @brief Memory-efficient inverted index using an array-of-vectors or hash table.
 * 
 * Provides an efficient way to index items by their hash values and retrieve 
 * items that match at least `threshold` of a query's hashes.
 */
class GlobalInvertedIndex {
private:
    /** @brief CONTIGUOUS storage of item indices. */
    vector<uint> sorted_item_indices_;
    
    /** 
     * @brief Flat array mapping [hash_index][hash_value] to the start index in sorted_item_indices_. 
     * size = (num_hashes_ + 1) * hash_range
     */
    vector<uint> bucket_offsets_;
    
    uint num_hashes_;
    uint threshold_;
    uint max_item_id_;
    uint hash_range_; // Typically 2^b (e.g. 2^20)

public:
    GlobalInvertedIndex() : num_hashes_(0), threshold_(0), max_item_id_(0), hash_range_(0) {}

    GlobalInvertedIndex(uint num_hashes, uint threshold, uint hash_range = 1048576) 
        : num_hashes_(num_hashes), threshold_(threshold), max_item_id_(0), hash_range_(hash_range) {}

    void build(const vector<vector<uint>>& item_hashes, const vector<uint>& global_item_indices) {
        if (item_hashes.empty()) return;
        uint n = item_hashes.size();
        
        // Find max hash value to size the flat array accurately
        uint max_hash = 0;
        max_item_id_ = 0;
        for (uint i = 0; i < n; ++i) {
            if (global_item_indices[i] > max_item_id_) max_item_id_ = global_item_indices[i];
            for (uint h = 0; h < num_hashes_; ++h) {
                if (item_hashes[i][h] > max_hash) max_hash = item_hashes[i][h];
            }
        }
        
        // Ensure our range can cover the max hash
        hash_range_ = max_hash + 1;
        
        // bucket_offsets_ shape is basically (num_hashes_ * hash_range_ + 1)
        // It stores the prefix sum of sizes (like CSR indptr)
        bucket_offsets_.assign(num_hashes_ * hash_range_ + 1, 0);
        
        // 1. Count frequencies of each (hash_index, hash_value)
        for (uint i = 0; i < n; ++i) {
            for (uint h = 0; h < num_hashes_; ++h) {
                uint flat_idx = h * hash_range_ + item_hashes[i][h];
                bucket_offsets_[flat_idx]++;
            }
        }
        
        // 2. Prefix sum to get offsets
        uint cumsum = 0;
        for (size_t i = 0; i < bucket_offsets_.size(); ++i) {
            uint count = bucket_offsets_[i];
            bucket_offsets_[i] = cumsum;
            cumsum += count;
        }
        
        // 3. Scatter item indices into the contiguous array using bucket_offsets_
        sorted_item_indices_.resize(cumsum);
        vector<uint> current_offsets = bucket_offsets_; // Copy for tracking inserts
        
        for (uint i = 0; i < n; ++i) {
            for (uint h = 0; h < num_hashes_; ++h) {
                uint flat_idx = h * hash_range_ + item_hashes[i][h];
                uint pos = current_offsets[flat_idx]++;
                sorted_item_indices_[pos] = global_item_indices[i];
            }
        }
    }

    vector<uint> get_matches(const vector<uint>& query_hashes) const {
        if (bucket_offsets_.empty()) return {};
        
        vector<uint16_t> counts(max_item_id_ + 1, 0);
        vector<uint> nz_indices;
        // Avoid reallocation overhead
        nz_indices.reserve(1000);
        
        for (uint h = 0; h < num_hashes_; ++h) {
            uint q_h = query_hashes[h];
            if (q_h >= hash_range_) continue;
            
            uint flat_idx = h * hash_range_ + q_h;
            uint start = bucket_offsets_[flat_idx];
            uint end = bucket_offsets_[flat_idx + 1];
            
            for (uint i = start; i < end; ++i) {
                uint item_idx = sorted_item_indices_[i];
                if (counts[item_idx] == 0) {
                    nz_indices.push_back(item_idx);
                }
                counts[item_idx]++;
            }
        }

        vector<uint> matches;
        for (uint item_idx : nz_indices) {
            if (counts[item_idx] >= threshold_) {
                matches.push_back(item_idx);
            }
        }
        return matches;
    }

    inline uint num_hashes() const { return num_hashes_; }
};

#endif /* B827CE93_864C_4295_A4D3_0F4800283341 */

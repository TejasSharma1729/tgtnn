#ifndef B827CE93_864C_4295_A4D3_0F4800283341
#define B827CE93_864C_4295_A4D3_0F4800283341

#include "headers.hpp"

/**
 * @brief Memory-efficient inverted index using sorted buckets.
 * 
 * Provides an efficient way to index many items and query for matches based on multiple hash values.
 * Uses a flat indexing structure to maximize cache efficiency and minimize memory overhead.
 */
class GlobalInvertedIndex {
public:
    vector<vector<HashBucket>> hash_buckets_; 
    vector<uint> doc_index_; 
    uint num_hashes_;
    uint threshold_;

    /**
     * @brief Empty constructor for GlobalInvertedIndex.
     */
    GlobalInvertedIndex() : num_hashes_(0), threshold_(0) {}
    
    /**
     * @brief Construct a new Global Inverted Index with specified parameters.
     * @param num_hashes The number of hashes (LSH functions) per item.
     * @param threshold The number of matching hashes required for a query match.
     */
    GlobalInvertedIndex(uint num_hashes, uint threshold) 
        : num_hashes_(num_hashes), threshold_(threshold) {
        hash_buckets_.resize(num_hashes);
    }

    /**
     * @brief Builds the inverted index from given item hashes.
     * 
     * This function clears any existing data, sorts the hash-item pairs for each hash
     * function, and packs them into a cache-efficient flat storage layout.
     * 
     * @param all_hashes A 2D vector where all_hashes[i] contains the hashes for item i.
     * @param item_indices The global indices of the items being indexed.
     */
    void build(const vector<vector<uint>>& all_hashes, const vector<uint>& item_indices) {
        if (all_hashes.empty() || item_indices.empty()) return;
        
        num_hashes_ = all_hashes[0].size();
        hash_buckets_.assign(num_hashes_, vector<HashBucket>());
        doc_index_.clear();
        
        uint N = item_indices.size();
        doc_index_.reserve(N * num_hashes_);

        for (uint h = 0; h < num_hashes_; ++h) {
            vector<pair<uint, uint>> hash_item_pairs;
            hash_item_pairs.reserve(N);
            for (uint i = 0; i < N; ++i) {
                uint global_idx = item_indices[i];
                hash_item_pairs.emplace_back(all_hashes[i][h], global_idx);
            }
            std::sort(hash_item_pairs.begin(), hash_item_pairs.end());

            for (uint i = 0; i < N; ) {
                uint h_val = hash_item_pairs[i].first;
                uint start_idx = i;
                while (i + 1 < N && hash_item_pairs[i + 1].first == h_val) i++;
                
                uint count = i - start_idx + 1;
                hash_buckets_[h].push_back({h_val, (uint)doc_index_.size(), count});
                for (uint j = start_idx; j <= i; ++j) {
                    doc_index_.push_back(hash_item_pairs[j].second);
                }
                i++;
            }
        }
    }

    /**
     * @brief Retrieves global item indices that match at least `threshold_` query hashes.
     * 
     * @param query_hashes The pre-computed hashes of the query vector.
     * @return vector<uint> A list of item indices that are candidates for similarity.
     */
    inline vector<uint> get_matches(const vector<uint> &query_hashes) const {
        if (num_hashes_ == 0 || doc_index_.empty()) return {};
        
        unordered_map<uint, uint> counts;
        for (uint h = 0; h < num_hashes_; ++h) {
            uint q_h = query_hashes[h];
            const auto& buckets = hash_buckets_[h];
            auto it = std::lower_bound(buckets.begin(), buckets.end(), q_h, 
                [](const HashBucket& b, uint val) { return b.hash_val < val; });
            
            if (it != buckets.end() && it->hash_val == q_h) {
                for (uint i = 0; i < it->num_items; ++i) {
                    counts[doc_index_[it->start_idx + i]]++;
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
     * @brief Returns the number of hash functions the index expects.
     * @return uint Hash count.
     */
    inline uint num_hashes() const { return num_hashes_; }
};



#endif /* B827CE93_864C_4295_A4D3_0F4800283341 */

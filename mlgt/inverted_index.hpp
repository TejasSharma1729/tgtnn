#ifndef AA26A202_7BD3_4190_A336_376AA5438FFA
#define AA26A202_7BD3_4190_A336_376AA5438FFA

#include "data_structures.hpp"

/**
 * Highly optimized inverted index for fast hash lookups during decoding.
 * 
 * Uses an unordered_map-based structure for scalability:
 * - Maps hash values to lists of point IDs efficiently
 * - Supports O(1) average lookup time
 * - Minimal memory overhead compared to CSR
 * - Per-pool and per-table organization
 * 
 * Design:
 * - Per-pool and per-table organization for hierarchical lookups
 * - Uses unordered_map for sparse hash distributions
 * - No need for pre-sorting or binary search
 */
class InvertedIndex {
public:
    using HashValue = int32_t;
    using PointID = uint; // Global point ID
    using InPoolIndex = uint16_t; // Index within a pool (0 to points_per_pool-1)
    using Frequency = int32_t;
    
    // Hash -> Points mapping structure. Stores InPoolIndex!
    using HashTable = std::unordered_map<HashValue, std::vector<InPoolIndex>>;
    using PoolIndex = std::vector<HashTable>;
    using GlobalIndex = std::vector<PoolIndex>;
    
protected:
    GlobalIndex index_;
    uint num_pools_;
    uint num_tables_;
    uint num_features_;
    uint points_per_pool_; // Store this for validation/optimization
    int debug_ = 0;
    
    // Store global point IDs mapping to reconstruct results if needed
    // Or we rely on the caller maintaining the 'pools' matrix.
    // For standalone completeness, we could store it, but to save memory we rely on caller.
    // Actually, query returns InPoolIndex to be fast.
    
public:
    InvertedIndex(
        const std::vector<std::vector<int32_t>>& hash_features,
        const std::vector<std::vector<uint>>& pools,
        int debug = 0
    ) {
        num_pools_ = pools.size();
        num_tables_ = hash_features[0].size();
        num_features_ = hash_features.size();
        points_per_pool_ = pools.empty() ? 0 : pools[0].size(); 
        debug_ = debug;
        
        index_.resize(num_pools_);
        
        if (debug_ > 0) {
            std::cout << "[SAFFRON] InvertedIndex build: num_pools=" << num_pools_
                      << ", num_tables=" << num_tables_
                      << ", num_features=" << num_features_ 
                      << ", points_per_pool approx=" << points_per_pool_ << std::endl;
        }

        for (uint pool_idx = 0; pool_idx < num_pools_; ++pool_idx) {
            index_[pool_idx].resize(num_tables_);
            
            const auto& pool_points = pools[pool_idx];
            
            for (uint table_idx = 0; table_idx < num_tables_; ++table_idx) {
                // Build hash->InPoolIndex mapping for this pool and table
                for (size_t i = 0; i < pool_points.size(); ++i) {
                    uint point_id = pool_points[i];
                    if (point_id < num_features_) {
                        HashValue hash_val = hash_features[point_id][table_idx];
                        // Store the index within the pool (i), not the global point_id
                        index_[pool_idx][table_idx][hash_val].push_back(static_cast<InPoolIndex>(i));
                    }
                }
            }
        }

        if (debug_ > 0) {
            auto mem = memory_usage();
            std::cout << "[SAFFRON] InvertedIndex built. Approx memory=" << mem << " bytes" << std::endl;
        }
    }
    
    /**
     * Query the index for points matching hashes across all tables.
     * Returns INDICES WITHIN THE POOL (0..points_per_pool-1)
     */
    inline std::vector<InPoolIndex> query(
        uint pool_id, 
        const std::vector<int32_t>& query_hashes, 
        uint min_tables = 1
    ) const {
        std::vector<InPoolIndex> results;
        if (pool_id >= num_pools_) return results;
        
        std::vector<uint8_t> counts(points_per_pool_, 0);
        
        for (uint t = 0; t < num_tables_; ++t) {
            if (t >= query_hashes.size()) break;
            
            const auto& table = index_[pool_id][t];
            auto it = table.find(query_hashes[t]);
            
            if (it != table.end()) {
                for (InPoolIndex idx : it->second) {
                    if (idx < points_per_pool_) {
                        counts[idx]++;
                    }
                }
            }
        }
        
        for (uint i = 0; i < points_per_pool_; ++i) {
            if (counts[i] >= min_tables) {
                results.push_back(static_cast<InPoolIndex>(i));
            }
        }
        
        return results;
    }

    /**
     * Identify positive pools using the optimized alternating sign algorithm.
     * 
     * @param query_hashes Query hash values for each table
     * @param threshold Match threshold
     * @param points_per_pool Max points per pool (size of temp buffer)
     * @return Boolean vector indicating positive pools
     */
    inline std::vector<bool> identify_positive_pools(
        const std::vector<int32_t>& query_hashes, 
        uint threshold,
        uint points_per_pool
    ) const {
        std::vector<bool> positive_pools(num_pools_, false);
        
        // "re-uses ONE points_per_pool sized temporary vector"
        // Using int16_t to allow for sign flipping logic avoiding overflow with typical table counts (e.g. 100)
        std::vector<int16_t> counts(points_per_pool, 0); 
        
        int match_thresh_int = static_cast<int>(threshold);

        for (uint pool_id = 0; pool_id < num_pools_; ++pool_id) {
            bool positive_mode = (pool_id % 2 == 0);
            bool is_positive = false;

            // Iterate tables
            for (uint table_idx = 0; table_idx < num_tables_; ++table_idx) {
                HashValue q_hash = query_hashes[table_idx];
                
                // Direct access for speed
                const auto& table = index_[pool_id][table_idx];
                auto it = table.find(q_hash);
                
                if (it != table.end()) {
                    // Iterate matching points in this pool
                    for (InPoolIndex idx : it->second) {
                        if (idx >= points_per_pool) continue; // Safety check

                        int16_t val = counts[idx];
                        int16_t next_val;

                        if (positive_mode) {
                            // "set the value ... to max(1, val + 1)"
                            // This resets any negative value (garbage from prev pool) to 1
                            next_val = val + 1;
                            if (next_val < 1) next_val = 1;
                            
                            if (next_val >= match_thresh_int) {
                                is_positive = true;
                            }
                        } else {
                            // "set the value ... to min(-1, val - 1)"
                            // This resets any positive value (garbage from prev pool) to -1
                            next_val = val - 1;
                            if (next_val > -1) next_val = -1;
                            
                            if (next_val <= -match_thresh_int) {
                                is_positive = true;
                            }
                        }
                        
                        counts[idx] = next_val;
                    }
                }
            }
            
            if (is_positive) {
                positive_pools[pool_id] = true;
            }
        }
        
        return positive_pools;
    }
    
    // Legacy support for multi-table point retrieval (SLOW compared to identify_positive_pools)
    // Note: This now returns InPoolIndex, not Global PointID
    inline Vec query_multi_table(
        uint pool_id,
        const std::vector<int32_t>& query_hashes,
        uint threshold,
        int debug = 0
    ) const {
        std::vector<InPoolIndex> pool_indices = query(pool_id, query_hashes, threshold);
        Vec result;
        result.reserve(pool_indices.size());
        for (auto idx : pool_indices) {
            result.push_back(static_cast<uint>(idx));
        }
        return result;
    }
    
    inline size_t memory_usage() const {
        size_t total = 0;
        for (const auto& pool : index_) {
            for (const auto& table : pool) {
                total += table.bucket_count() * sizeof(void*);
                for (const auto& [hash, points] : table) {
                    total += sizeof(HashValue);
                    total += points.capacity() * sizeof(InPoolIndex);
                }
            }
        }
        return total;
    }
    
    inline std::unordered_map<std::string, size_t> statistics() const {
        std::unordered_map<std::string, size_t> stats;
        size_t total_hashes = 0;
        size_t total_points = 0;
        size_t total_tables = 0;
        
        for (const auto& pool : index_) {
            for (const auto& table : pool) {
                if (!table.empty()) {
                    total_tables++;
                    total_hashes += table.size();
                    for (const auto& [hash, points] : table) {
                        total_points += points.size();
                    }
                }
            }
        }
        
        stats["num_pools"] = num_pools_;
        stats["num_tables"] = num_tables_;
        stats["num_features"] = num_features_;
        stats["total_tables"] = total_tables;
        stats["total_hashes"] = total_hashes;
        stats["total_points"] = total_points;
        stats["avg_hashes_per_table"] = total_tables > 0 ? total_hashes / total_tables : 0;
        
        return stats;
    }
};


#endif /* AA26A202_7BD3_4190_A336_376AA5438FFA */

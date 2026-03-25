#ifndef EB0C1D6C_5A3F_4C2C_8C1D_3F7E2D6B8A9E
#define EB0C1D6C_5A3F_4C2C_8C1D_3F7E2D6B8A9E

#include "data_structures.hpp"
#include "inverted_index.hpp"
#include "saffron.hpp"
#include "simhash.hpp"


/**
 * MLGT with SAFFRON Index Wrapper, has two indices
 * 
 * Has an index for T-matrix pools --> tables --> hash value --> point IDs
 * Has an index for U-matrix pools --> tables --> hash value --> point IDs
 */
class MLGTSaffronIndex {
public:
    uint num_buckets;
    uint num_tests;
    uint num_tables;
    uint num_features;
    int debug_;

protected:
    std::unique_ptr<InvertedIndex> T_index_;
    std::unique_ptr<InvertedIndex> U_index_;
    VecVec T_pools_;
    VecVec U_pools_;
    
public:
    /**
     * Initialize MLGT SAFFRON Index
     * @param hash_features Hash features matrix
     * @param saffron Saffron decoder instance
     */
    MLGTSaffronIndex(
        const std::vector<std::vector<int32_t>>& hash_features,
        Saffron& saffron,
        int debug = 0
    ) : num_buckets(saffron.num_buckets),
        num_tests(saffron.num_tests),
        num_tables(hash_features[0].size()),
        num_features(hash_features.size()),
        debug_(debug)
    {
        // Generate T-matrix pools
        T_pools_.reserve(saffron.num_buckets);
        for (uint i = 0; i < saffron.num_buckets; ++i) {
            Vec pool;
            for (uint j = 0; j < num_features; ++j) {
                if (saffron.T_matrix[i][j]) {
                    pool.push_back(j);
                }
            }
            T_pools_.push_back(pool);
        }
        
        T_index_ = std::make_unique<InvertedIndex>(hash_features, T_pools_, debug_);
        if (debug_ > 0) {
            std::cout << "[SAFFRON] MLGTSaffronIndex: Created T_index_" << std::endl;
        }
        
        // Generate U-matrix pools
        U_pools_.reserve(saffron.num_tests);
        for (uint i = 0; i < saffron.num_tests; ++i) {
            Vec pool;
            for (uint j = 0; j < num_features; ++j) {
                if (saffron.U_matrix[i][j]) {
                    pool.push_back(j);
                }
            }
            U_pools_.push_back(pool);
        }
        
        U_index_ = std::make_unique<InvertedIndex>(hash_features, U_pools_, debug_);
        if (debug_ > 0) {
            std::cout << "[SAFFRON] MLGTSaffronIndex: Created U_index_" << std::endl;
        }
    }

    /**
     * Query the index using a combined pool_id (bucket * num_tests + test) across all tables.
     */
    inline Vec query(
        uint pool_id,
        const std::vector<int32_t>& query_hashes,
        uint min_tables = 1
    ) const {
        uint bucket_id = pool_id / num_tests;
        uint test_id = pool_id % num_tests;
        
        if (bucket_id >= num_buckets || test_id >= num_tests) return {};

        // Query T (all tables)
        std::vector<InvertedIndex::InPoolIndex> t_indices = T_index_->query(bucket_id, query_hashes, min_tables);
        
        // Query U (all tables)
        std::vector<InvertedIndex::InPoolIndex> u_indices = U_index_->query(test_id, query_hashes, min_tables);
        
        // Convert to Global IDs
        Vec t_global, u_global;
        t_global.reserve(t_indices.size());
        u_global.reserve(u_indices.size());
        
        const auto& t_pool = T_pools_[bucket_id];
        for (auto idx : t_indices) {
            if (idx < t_pool.size()) t_global.push_back(t_pool[idx]);
        }
        
        const auto& u_pool = U_pools_[test_id];
        for (auto idx : u_indices) {
            if (idx < u_pool.size()) u_global.push_back(u_pool[idx]);
        }
        
        Vec result;
        std::set_intersection(
            t_global.begin(), t_global.end(),
            u_global.begin(), u_global.end(),
            std::back_inserter(result)
        );
        
        return result;
    }

    inline size_t memory_usage() const {
        return T_index_->memory_usage() + U_index_->memory_usage();
    }

    inline std::unordered_map<std::string, size_t> statistics() const {
        std::unordered_map<std::string, size_t> stats;
        for (const auto& [key, value] : T_index_->statistics()) {
            stats["T_" + key] = value;
        }
        for (const auto& [key, value] : U_index_->statistics()) {
            stats["U_" + key] = value;
        }
        return stats;
    }
};


/**
 * MLGT with SAFFRON
 * 
 * Combines SimHash locality-sensitive hashing with SAFFRON group testing.
 */
class MLGTsaffron {
public:
    uint num_tables;
    uint num_bits;
    uint dimension;
    uint num_neighbors;
    // threshold removed
    
    std::vector<std::vector<float32_t>> features;  // Normalized features
    uint num_features;
    
    Saffron saffron;
    VecVec pool_list;
    std::vector<std::vector<int32_t>> hash_features_;
    std::unique_ptr<MLGTSaffronIndex> index_;
    std::unique_ptr<SimHash> simhash_;
    int debug_ = 0;
    
public:
    /**
     * Initialize MLGT with SAFFRON.
     * 
     * @param num_tables Number of hash tables
     * @param num_bits Bits per hash value
     * @param dimension Feature dimension
     * @param num_neighbors Expected number of neighbors (sparsity for SAFFRON)
     * @param features Feature matrix (will be L2 normalized)
     */
    MLGTsaffron(
        uint num_tables,
        uint num_bits,
        uint dimension,
        uint num_neighbors,
        const std::vector<std::vector<float32_t>>& features_in,
        int debug = 0
    ) : num_tables(num_tables),
        num_bits(num_bits),
        dimension(dimension),
        num_neighbors(num_neighbors),
        features(features_in),
        num_features(features_in.size()),
        saffron(num_features, num_neighbors, debug),
        index_(nullptr)
    {
        debug_ = debug;
        // L2 normalize features
        for (auto& feature_vec : features) {
            float32_t norm = 0.0f;
            for (float32_t val : feature_vec) {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            if (norm > 1e-6f) {
                for (float32_t& val : feature_vec) {
                    val /= norm;
                }
            }
        }
        if (debug_ > 0) {
            std::cout << "[SAFFRON] MLGTsaffron: normalized features, num_features=" << num_features
                      << ", dim=" << dimension << std::endl;
        }
        
        // Generate pools
        pool_list = saffron.pools(debug_);
        
        // Initialize SimHash and generate hash features
        simhash_ = std::make_unique<SimHash>(num_tables, num_bits, dimension, debug_);
        hash_features_ = simhash_->hash_features(features, debug_);
        if (debug_ > 0) {
            std::cout << "[SAFFRON] MLGTsaffron: generated hash features." << std::endl;
        }
        
        // Create index
        std::vector<std::vector<int32_t>> pools_int;
        for (const auto& pool : pool_list) {
            std::vector<int32_t> pool_int;
            for (uint item : pool) {
                pool_int.push_back(static_cast<int32_t>(item));
            }
            pools_int.push_back(pool_int);
        }
        
        index_ = std::make_unique<MLGTSaffronIndex>(hash_features_, const_cast<Saffron&>(saffron), debug_);
        if (debug_ > 0) {
            std::cout << "[SAFFRON] MLGTsaffron: InvertedIndex created." << std::endl;
        }
    }
    
    /**
     * Query for nearest neighbors using SAFFRON-enhanced MLGT.
     * 
     * @param query_feature Query feature vector
     * @param k Number of neighbors to return
     * @return Vector of neighbor indices
     */
    inline Vec query(
        const std::vector<float32_t>& query_feature,
        uint k,
        int debug = 0
    ) const {
        Vec results;
        
        if (!index_) {
            return results;
        }
        
        // 1. Generate query hashes
        std::vector<int32_t> query_hashes = simhash_->hash_vector(query_feature, debug);
        
        // 2. Query each pool
        std::unordered_map<uint, uint> candidate_scores;
        
        for (uint pool_idx = 0; pool_idx < saffron.num_pools; ++pool_idx) {
            // Use index to query this pool
            // (Implementation details depend on pool structure)
            Vec matches = index_->query(pool_idx, query_hashes, 1);
            for (auto id : matches) candidate_scores[id]++;
        }
        
        // 3. Sort by score and return top k
        std::vector<std::pair<uint, uint>> scored_items;
        for (const auto& [item_id, score] : candidate_scores) {
            scored_items.push_back({item_id, score});
        }
        
        std::sort(scored_items.begin(), scored_items.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (uint i = 0; i < std::min(k, (uint)scored_items.size()); ++i) {
            results.push_back(scored_items[i].first);
        }
        if (std::max(debug_, debug) > 0) {
            std::cout << "[SAFFRON] MLGTsaffron.query(): candidates=" << candidate_scores.size()
                      << ", returned=" << results.size() << std::endl;
        }
        
        return results;
    }
    
    /**
     * Get top K hash neighbors for a query using direct hash comparison.
     * 
     * Returns candidates that match query hashes directly.
     * Candidates are ranked by number of matching hash tables.
     * This method bypasses the MLGT pool structure and scans all features.
     * 
     * @param query_feature Query feature vector
     * @param k Number of neighbors to return
     * @param min_matching_tables Minimum number of hash tables that must match (default: 1)
     * @return Vector of up to k neighbor indices sorted by match count (descending)
     */
    inline Vec get_top_k_hash_neighbors(
        const std::vector<float32_t>& query_feature,
        uint k,
        uint min_matching_tables = 1,
        int debug = 0
    ) const {
        Vec results;
        
        // Generate query hashes
        std::vector<int32_t> query_hashes = simhash_->hash_vector(query_feature, debug);
        // Direct comparison of hash_features_ vs query_hashes (no MLGT/index queries)
        // For each feature, count table-wise exact matches against query_hashes
        std::unordered_map<PointID, uint> candidate_scores;
        candidate_scores.reserve(num_features);

        for (uint fid = 0; fid < num_features; ++fid) {
            uint matches = 0;
            const auto &hf = hash_features_[fid];
            // assume hf.size() == num_tables
            for (uint t = 0; t < num_tables; ++t) {
                if (hf[t] == query_hashes[t]) ++matches;
            }
            if (matches >= min_matching_tables) {
                candidate_scores[fid] = matches;
            }
        }
        
        // Sort by score and return top k
        std::vector<std::pair<PointID, uint>> scored_items;
        scored_items.reserve(candidate_scores.size());
        
        for (const auto& [point_id, score] : candidate_scores) {
            scored_items.push_back({point_id, score});
        }
        
        std::sort(scored_items.begin(), scored_items.end(),
                  [](const auto& a, const auto& b) { 
                      return a.second > b.second;  // Sort by score descending
                  });
        
        for (uint i = 0; i < std::min(k, (uint)scored_items.size()); ++i) {
            results.push_back(scored_items[i].first);
        }
        if (std::max(debug_, debug) > 0) {
            std::cout << "[SAFFRON] get_top_k_hash_neighbors(): candidates=" << candidate_scores.size()
                      << ", returned=" << results.size() << std::endl;
        }
        
        return results;
    }
    
    /**
     * Get hash neighbors with score information using direct hash comparison.
     * 
     * Returns candidates matched by hashes with their match scores.
     * Score indicates the number of matching hash tables between query and candidate.
     * This method bypasses the MLGT pool structure and scans all features.
     * 
     * @param query_feature Query feature vector
     * @param k Number of neighbors to return
     * @param min_matching_tables Minimum number of hash tables that must match (default: 1)
     * @return Vector of pairs (neighbor_id, match_score), sorted by score descending
     */
    inline std::vector<std::pair<PointID, uint>> get_hash_neighbors_with_scores(
        const std::vector<float32_t>& query_feature,
        uint k,
        uint min_matching_tables = 1,
        int debug = 0
    ) const {
        std::vector<std::pair<PointID, uint>> results;
        
        // Generate query hashes
        std::vector<int32_t> query_hashes = simhash_->hash_vector(query_feature, debug);
        
        // Direct comparison of hash_features_ vs query_hashes
        std::unordered_map<PointID, uint> candidate_scores;
        candidate_scores.reserve(num_features);

        for (uint fid = 0; fid < num_features; ++fid) {
            uint matches = 0;
            const auto &hf = hash_features_[fid];
            for (uint t = 0; t < num_tables; ++t) {
                if (hf[t] == query_hashes[t]) ++matches;
            }
            if (matches >= min_matching_tables) {
                candidate_scores[fid] = matches;
            }
        }
        
        // Convert to vector and sort
        std::vector<std::pair<PointID, uint>> scored_items;
        scored_items.reserve(candidate_scores.size());
        
        for (const auto& [point_id, score] : candidate_scores) {
            scored_items.push_back({point_id, score});
        }
        
        std::sort(scored_items.begin(), scored_items.end(),
                  [](const auto& a, const auto& b) { 
                      return a.second > b.second;  // Sort by score descending
                  });
        
        // Return top k
        uint limit = std::min(k, (uint)scored_items.size());
        results.insert(results.end(), scored_items.begin(), scored_items.begin() + limit);
        
        if (std::max(debug_, debug) > 0) {
            std::cout << "[SAFFRON] get_hash_neighbors_with_scores(): candidates=" << candidate_scores.size()
                      << ", returned=" << results.size() << std::endl;
        }
        return results;
    }
    
    /**
     * Get hash neighbors from a specific pool.
     * 
     * Queries only a single pool for hash neighbors matching the query.
     * Useful for analyzing pool-specific matches.
     * 
     * @param query_feature Query feature vector
     * @param pool_id Pool index to query
     * @param k Number of neighbors to return
     * @param min_matching_tables Minimum number of hash tables that must match (default: 1)
     * @return Vector of up to k neighbor indices in the specified pool
     */
    inline Vec get_pool_hash_neighbors(
        const std::vector<float32_t>& query_feature,
        uint pool_id,
        uint k,
        uint min_matching_tables = 1,
        int debug = 0
    ) const {
        Vec results;
        
        if (!index_ || pool_id >= saffron.num_pools) {
            return results;
        }
        
        // Generate query hashes
        std::vector<int32_t> query_hashes = simhash_->hash_vector(query_feature, debug);
        
        // Query this specific pool
        Vec matching_points = index_->query(
            pool_id, 
            query_hashes, 
            min_matching_tables
        );
        
        // Return top k
        uint limit = std::min(k, (uint)matching_points.size());
        results.insert(results.end(), matching_points.begin(), matching_points.begin() + limit);
        if (std::max(debug_, debug) > 0) {
            std::cout << "[SAFFRON] get_pool_hash_neighbors(pool=" << pool_id
                      << "): returned=" << results.size() << std::endl;
        }
        
        return results;
    }
};

#endif // EB0C1D6C_5A3F_4C2C_8C1D_3F7E2D6B8A9E
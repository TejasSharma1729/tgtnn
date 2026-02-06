#ifndef B00FB789_2F58_408D_A242_F706286620C3
#define B00FB789_2F58_408D_A242_F706286620C3

#include "data_structures.hpp"
#include "inverted_index.hpp"
#include "simhash.hpp"


std::pair<Matrix, BitMatrix> obtain_pools(
    uint num_features,
    uint num_pools,
    uint pools_per_point,
    uint points_per_pool,
    int debug = 0
) {
    if (points_per_pool * num_pools < pools_per_point * num_features) {
        throw std::invalid_argument("obtain_pools: insufficient capacity to assign points to pools");
    }

    Matrix pools(num_pools, Vec(points_per_pool, (uint)-1));
    std::vector<uint> pool_fill_count(num_pools, 0);
    BitMatrix pooling_matrix(num_pools, std::vector<bool>(num_features, false));
    
    std::vector<uint> point_assignments;
    point_assignments.reserve(num_features * pools_per_point);
    for (uint i = 0; i < num_features; ++i) {
        for (uint j = 0; j < pools_per_point; ++j) {
            point_assignments.push_back(i);
        }
    }
    
    if (debug > 0) std::cout << "Creating random pool assignment..." << std::endl;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(point_assignments.begin(), point_assignments.end(), g);
    
    std::uniform_int_distribution<uint> dist(0, num_pools - 1);
    
    for (size_t idx = 0; idx < point_assignments.size(); ++idx) {
        uint point_idx = point_assignments[idx];
        int selected_pool = -1;
        
        for (int k = 0; k < 100; ++k) {
            uint p = dist(g);
            if (pool_fill_count[p] < points_per_pool && !pooling_matrix[p][point_idx]) {
                selected_pool = (int)p;
                break;
            }
        }
        
        if (selected_pool == -1) {
            std::vector<uint> candidates;
            for (uint p = 0; p < num_pools; ++p) {
                if (pool_fill_count[p] < points_per_pool && !pooling_matrix[p][point_idx]) {
                    candidates.push_back(p);
                }
            }
            if (candidates.empty()) {
                throw std::runtime_error("Cannot assign point to pool (capacity exceeded)");
            }
            std::uniform_int_distribution<size_t> cand_dist(0, candidates.size() - 1);
            selected_pool = candidates[cand_dist(g)];
        }
        
        pools[selected_pool][pool_fill_count[selected_pool]] = point_idx;
        pooling_matrix[selected_pool][point_idx] = true;
        pool_fill_count[selected_pool]++;
        
        if (debug > 0 && idx % 100000 == 0) {
            std::cout << "Assigned " << idx << " points" << std::endl;
        }
    }
    
    return {pools, pooling_matrix};
}

std::vector<uint> my_matching_algo(
    const BitMatrix& pooling_matrix,
    std::vector<uint> positive_pools
) {
    uint num_pools = pooling_matrix.size();
    if (num_pools == 0) return {};
    uint num_features = pooling_matrix[0].size();
    
    std::vector<uint8_t> possible(num_features, 1);
    std::vector<uint8_t> candidates(num_features, 0);
    
    for (uint i = 0; i < num_pools; ++i) {
        if (positive_pools[i] == 0) {
            for (uint j = 0; j < num_features; ++j) {
                if (pooling_matrix[i][j]) {
                    possible[j] = 0;
                }
            }
        }
    }
    
    while (true) {
        std::vector<uint8_t> newly_identified(num_features, 0);
        
        for (uint i = 0; i < num_pools; ++i) {
            if (positive_pools[i] == 0) continue;
            
            int count = 0;
            uint last_j = 0;
            for (uint j = 0; j < num_features; ++j) {
                if (pooling_matrix[i][j] && possible[j]) {
                    count++;
                    last_j = j;
                    if (count > 1) break;
                }
            }
            
            if (count == 1) {
                newly_identified[last_j] = 1;
            }
        }
        
        uint sum_newly = 0;
        for (auto v : newly_identified) sum_newly += v;
        if (sum_newly == 0) break;
        
        for (uint j = 0; j < num_features; ++j) {
            if (newly_identified[j]) {
                candidates[j] = 1;
                possible[j] = 0;
            }
        }
        
        std::vector<uint8_t> identified_pools(num_pools, 0);
        for (uint i = 0; i < num_pools; ++i) {
             for (uint j = 0; j < num_features; ++j) {
                 if (pooling_matrix[i][j] && newly_identified[j]) {
                     identified_pools[i] = 1;
                     break;
                 }
             }
        }
        
        for (uint i = 0; i < num_pools; ++i) {
            if (identified_pools[i]) {
                positive_pools[i] = 0;
            }
        }
    }
    
    std::vector<uint> result_indices;
    for (uint j = 0; j < num_features; ++j) {
        if (candidates[j]) {
            result_indices.push_back(j);
        }
    }
    return result_indices;
}


class MLGT {
public:
    uint num_tables;
    uint num_bits;
    uint dimension;
    uint num_pools;
    uint pools_per_point;
    uint points_per_pool;
    // threshold removed
    uint match_threshold; // min number of table matches
    uint num_neighbors;
    
    std::vector<std::vector<float32_t>> features;  // Normalized features
    uint num_features;
 
protected:
    std::vector<std::vector<int32_t>> hash_features_;
    std::unique_ptr<InvertedIndex> index_;
    std::unique_ptr<SimHash> simhash_;
    BitMatrix pooling_matrix_;
    int debug_ = 0;

public:
    /**
     * @brief Construct a new MLGT object
     * 
     * @param num_tables Number of hash tables
     * @param num_bits Number of bits per hash
     * @param dimension Dimension of input vectors
     * @param num_pools Number of pools
     * @param pools_per_point Number of pools each point is assigned to
     * @param points_per_pool Number of points in each pool
     * @param match_threshold Minimum number of table matches for a point to be considered
     * @param num_neighbors Number of neighbors to return (top_k)
     * @param features_in Input features matrix
     * @param debug Debug level
     */
    MLGT(
        uint num_tables,
        uint num_bits,
        uint dimension,
        uint num_pools,
        uint pools_per_point,
        uint points_per_pool,
        uint match_threshold,
        uint num_neighbors,
        const std::vector<std::vector<float32_t>>& features_in,
        int debug = 0
    ) : num_tables(num_tables),
        num_bits(num_bits),
        dimension(dimension),
        num_pools(num_pools),
        pools_per_point(pools_per_point),
        points_per_pool(points_per_pool),
        match_threshold(match_threshold),
        num_neighbors(num_neighbors),
        features(features_in),
        num_features(features_in.size()),
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
            std::cout << "[MLGT] MLGT: normalized features, num_features=" << num_features
                      << ", dim=" << dimension << std::endl;
        }
        
        // Initialize SimHash and generate hash features
        simhash_ = std::make_unique<SimHash>(num_tables, num_bits, dimension, debug_);
        hash_features_ = simhash_->hash_features(features, debug_);
        if (debug_ > 0) {
            std::cout << "[MLGT] MLGT: generated hash features." << std::endl;
        }
        
        // Generate pools
        auto pools_pair = obtain_pools(num_features, num_pools, pools_per_point, points_per_pool, debug_);
        Matrix pools = pools_pair.first;
        pooling_matrix_ = pools_pair.second;
        
        index_ = std::make_unique<InvertedIndex>(hash_features_, pools, debug_);
        if (debug_ > 0) {
            std::cout << "[MLGT] MLGT: InvertedIndex created." << std::endl;
        }
    }
    
    /**
     * @brief Query the MLGT index for nearest neighbors
     * 
     * @param query_feature Query vector
     * @return std::vector<uint> List of nearest neighbor indices
     */
    inline Vec query(
        const std::vector<float32_t>& query_feature
    ) {
        Vec results;
        
        if (!index_) {
            return results;
        }

        // Generate query hashes
        std::vector<int32_t> query_hashes = simhash_->hash_vector(query_feature, debug_);
        
        // Identify positive pools using optimized multi-pool algorithm
        std::vector<bool> positive_pools_bool = index_->identify_positive_pools(
            query_hashes, 
            match_threshold,
            points_per_pool
        );
        
        // Convert bool vector to uint vector for compatibility with matching algo
        std::vector<uint> positive_pools(num_pools);
        for (uint i = 0; i < num_pools; ++i) {
            positive_pools[i] = positive_pools_bool[i] ? 1 : 0;
        }
        
        // Recover candidates using group testing algorithm
        Vec candidates = my_matching_algo(pooling_matrix_, positive_pools);
        
        if (candidates.empty()) {
             // Fallback logic here if desired
             return results; 
        }

        // Rank candidates by hash match score
        std::vector<std::pair<int, uint>> scores;
        scores.reserve(candidates.size());
        
        for (uint cand_idx : candidates) {
            if (cand_idx >= num_features) continue;
            
            const auto& cand_hashes = hash_features_[cand_idx];
            int score = 0;
            for (size_t t = 0; t < num_tables; ++t) {
                if (cand_hashes[t] == query_hashes[t]) {
                    score++;
                }
            }
            scores.push_back({score, cand_idx});
        }
        
        // Sort descending by score
        std::sort(scores.begin(), scores.end(), [](const std::pair<int, uint>& a, const std::pair<int, uint>& b) {
            return a.first > b.first;
        });
        
        // Collect top K
        for (size_t i = 0; i < scores.size() && i < num_neighbors; ++i) {
            results.push_back(scores[i].second);
        }
        
        return results;
    }
};


#endif /* B00FB789_2F58_408D_A242_F706286620C3 */

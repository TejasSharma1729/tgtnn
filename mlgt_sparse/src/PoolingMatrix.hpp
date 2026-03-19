#ifndef D9513A33_85A9_4731_A0C8_774439B3ECCE
#define D9513A33_85A9_4731_A0C8_774439B3ECCE

#include "headers.hpp"

/**
 * @brief Represents the mapping between items and pools for the SAFFRON algorithm.
 */
struct PoolingMatrix {
    vector<vector<uint>> pools_to_items; ///< List of vectors, where each vector contains the item indices in that pool.
    vector<vector<uint>> items_to_pools; ///< List of vectors, where each vector contains the pool indices the item belongs to.
    uint num_features; ///< Total number of features/items (n).
    uint num_pools; ///< Total number of pools (m).
};

/**
 * @brief Randomized pooling matrix generation for SAFFRON.
 * 
 * Maps each of the n items into multiple random pools. This ensures that even
 * in the presence of noise, each item appears in a different set of pools
 * with high probability, facilitating sparse recovery.
 * 
 * @param num_features Total number of features (n).
 * @param sparsity Expected sparsity level (k).
 * @param debug Debug verbosity level.
 * @return PoolingMatrix The mapping between items and pools.
 */
inline PoolingMatrix computePools(uint num_features, uint sparsity, int debug = 0) {
    uint num_pools = sparsity * NUM_POOLS_COEFF;
    uint pools_per_item = POOLS_PER_ITEM;
    
    PoolingMatrix pooling_matrix;
    pooling_matrix.num_features = num_features;
    pooling_matrix.num_pools = num_pools;
    pooling_matrix.pools_to_items.resize(num_pools);
    pooling_matrix.items_to_pools.resize(num_features);

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<uint> dis(0, num_pools - 1);
    
    for (uint item_idx = 0; item_idx < num_features; ++item_idx) {
        unordered_set<uint> chosen_pools;
        while (chosen_pools.size() < pools_per_item) {
            chosen_pools.insert(dis(gen));
        }
        for (uint pool_idx : chosen_pools) {
            pooling_matrix.pools_to_items[pool_idx].push_back(item_idx);
            pooling_matrix.items_to_pools[item_idx].push_back(pool_idx);
        }
    }
    
    if (debug > 0) {
        cout << "[Compute Pools] num_features=" << num_features
             << ", sparsity=" << sparsity
             << ", num_pools=" << num_pools
             << ", pools_per_item=" << pools_per_item << endl;
    }
    
    return pooling_matrix;
}


#endif /* D9513A33_85A9_4731_A0C8_774439B3ECCE */

#ifndef F938C541_109E_4EC0_AE90_E0FED079CBC9
#define F938C541_109E_4EC0_AE90_E0FED079CBC9

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
 * @brief Computes the pooling matrix for SAFFRON.
 * 
 * @param num_features Total number of features (n).
 * @param sparsity Expected sparsity level (k).
 * @param debug Debug level.
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

    mt19937 gen(random_device{}());
    uniform_int_distribution<uint> dis(0, num_pools - 1);
    
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


/**
 * @brief Generates a robust 6L SAFFRON signature for singleton and doubleton recovery.
 * The signature consists of 3 blocks, each containing a parity bit, U1, and ~U1.
 * Block 1: Always active (h=1).
 * Block 2: Active based on a pseudo-random hash bit H1(i).
 * Block 3: Active based on a pseudo-random hash bit H2(i).
 * 
 * @param j The item index.
 * @param signature_length Total length of the 6L signature (3 * (2*L + 1)).
 * @return vector<bool> The boolean signature.
 */
inline vector<bool> getSignature(uint j, uint signature_length) {
    uint block_len = signature_length / 3;
    uint num_bits = (block_len - 1) / 2;
    vector<bool> signature(signature_length, false);
    
    // Block 1: Always 1
    // Block 2 & 3: Random masks to separate items in doubletons
    bool h[3] = { true, (bool)((j >> 7) & 1), (bool)((j >> 13) & 1) }; // Simple deterministic bits

    for (uint b = 0; b < 3; ++b) {
        if (!h[b]) continue;
        uint offset = b * block_len;
        signature[offset] = true; // Parity bit
        for (uint bit_idx = 0; bit_idx < num_bits; ++bit_idx) {
            bool bit = (j & (1 << bit_idx)) != 0;
            signature[offset + 1 + bit_idx] = bit;
            signature[offset + 1 + num_bits + bit_idx] = !bit;
        }
    }
    return signature;
}


/**
 * @brief Decodes a single block of a signature.
 * 
 * @param measurement A slice of the measurement vector for one block.
 * @return optional<uint> The decoded item index, or nullopt if 0 or multi-pool.
 */
inline optional<uint> decodeBlock(const vector<bool>& measurement) {
    if (!measurement[0]) return nullopt; // Parity 0 -> even/zero items
    uint num_bits = (measurement.size() - 1) / 2;
    uint index = 0;
    for (uint bit_idx = 0; bit_idx < num_bits; ++bit_idx) {
        bool b1 = measurement[1 + bit_idx];
        bool b2 = measurement[1 + num_bits + bit_idx];
        if (b1 == b2) return nullopt; // Collision or empty
        if (b1) index |= (1 << bit_idx);
    }
    return index;
}


/**
 * @brief Decodes a signature from a measurement vector, supporting doubleton resolution.
 * 
 * @param measurement A boolean vector of measurement bits (6L).
 * @param signature_length Length of the signature.
 * @return vector<uint> List of identified item indices (0, 1, or 2 items).
 */
inline vector<uint> decodeSignature(
    const vector<bool>& measurement,
    uint signature_length
) {
    uint block_len = signature_length / 3;
    uint num_bits = (block_len - 1) / 2;
    
    // 1. Try singleton decoding from Block 1
    vector<bool> block1(measurement.begin(), measurement.begin() + block_len);
    optional<uint> s = decodeBlock(block1);
    if (s.has_value()) return { s.value() };
    
    // 2. Try doubleton recovery if block 1 parity is 0
    if (!block1[0]) {
        uint S = 0;
        bool non_zero = false;
        for (uint i = 0; i < num_bits; ++i) {
            if (block1[1 + i]) { S |= (1 << i); non_zero = true; }
        }
        if (!non_zero) return {}; // Truly empty

        // Look for a singleton in other blocks
        for (uint b = 1; b < 3; ++b) {
            vector<bool> block_m(measurement.begin() + b * block_len, measurement.begin() + (b + 1) * block_len);
            optional<uint> i1 = decodeBlock(block_m);
            if (i1.has_value()) {
                uint val1 = i1.value();
                uint val2 = S ^ val1;
                // Doubleton recovered: {val1, val2}
                return { val1, val2 };
            }
        }
    }
    
    return {};
}


/**
 * @brief Represents a candidate item found in a pool.
 */
struct Candidate {
    uint pool_idx;
    uint item_idx;
};

/**
 * @brief The core Saffron algorithm implementation for sparse recovery.
 */
class Saffron {
protected:
    PoolingMatrix pools_; // The pooling matrix defining the item-to-pool and pool-to-item mappings.
    uint num_features_; // Total number of features/items (n).
    uint sparsity_; // Expected sparsity level (k).
    uint num_pools_; // Total number of pools (m).
    uint signature_length_; // Length of the signatures.
    int debug_ = 0; // Debug level (0 for none, higher values for more verbose output)

public:
    /**
     * @brief Initializes the Saffron algorithm setup.
     * 
     * @param num_features Total number of features (n).
     * @param sparsity Expected sparsity level (k).
     * @param debug Debug level.
     */
    Saffron(uint num_features, uint sparsity, int debug = 0) :
        num_features_(num_features),
        sparsity_(sparsity),
        debug_(debug)
    {
        uint L = ceil(log2(num_features));
        signature_length_ = 3 * (2 * L + 1);
        pools_ = computePools(num_features, sparsity, debug);
        num_pools_ = pools_.num_pools;
    }

    /**
     * @brief Returns the number of features.
     * @return uint 
     */
    inline uint num_features() const { return num_features_; }

    /**
     * @brief Returns the sparsity level.
     * @return uint 
     */
    inline uint sparsity() const { return sparsity_; }

    /**
     * @brief Returns the number of pools.
     * @return uint 
     */
    inline uint num_pools() const { return num_pools_; }

    /**
     * @brief Returns the length of the signatures.
     * @return uint 
     */
    inline uint signature_length() const { return signature_length_; }

    ~Saffron() = default;

    /**
     * @brief Executes the peeling algorithm to recover identified items from residuals.
     * 
     * @param residuals A 2D vector of booleans (num_pools x signature_bits).
     * @param debug Debug level.
     * @return set<uint> A set of identified item indices.
     */
    inline set<uint> peelingAlgorithm(vector<vector<bool>> residuals, int debug = 0) {
        set<uint> identified;
        queue<Candidate> candidates;
        assert(residuals.size() == pools_.num_pools && "Residuals size mismatch");

        auto check_pool = [&](uint p_idx) {
            vector<uint> decoded = decodeSignature(residuals[p_idx], signature_length_);
            for (uint it : decoded) {
                if (it < num_features_) {
                    candidates.push({p_idx, it});
                }
            }
        };

        for (uint pool_idx = 0; pool_idx < pools_.num_pools; ++pool_idx) {
            check_pool(pool_idx);
        }

        if (debug_ > 0) {
            cout << "Initialized peeling with " << candidates.size() 
                << " candidates." << endl;
        }

        while (!candidates.empty()) {
            Candidate cand = candidates.front();
            candidates.pop();

            if (identified.count(cand.item_idx)) continue;
            
            identified.insert(cand.item_idx);
            
            vector<bool> sig = getSignature(cand.item_idx, signature_length_);
            for (uint p_idx : pools_.items_to_pools[cand.item_idx]) {
                for (uint b = 0; b < signature_length_; ++b) {
                    if (sig[b]) residuals[p_idx][b].flip();
                }
                check_pool(p_idx);
            }
        }
        return identified;
    }
};


#endif /* F938C541_109E_4EC0_AE90_E0FED079CBC9 */

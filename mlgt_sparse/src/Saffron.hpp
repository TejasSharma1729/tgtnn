#ifndef F938C541_109E_4EC0_AE90_E0FED079CBC9
#define F938C541_109E_4EC0_AE90_E0FED079CBC9

#include "headers.hpp"
#include "PoolingMatrix.hpp"


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
 * @brief Decodes a single block of a signature to extract an index.
 * 
 * @param measurement A slice of the measurement vector for one block.
 * @return optional<uint> The decoded item index, or nullopt if 0 or multi-pool collision.
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
 * @brief Decodes a full signature from a measurement vector, supporting doubleton resolution.
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
    if (s.has_value()) {
        uint val = s.value();
        vector<bool> expected = getSignature(val, signature_length);
        bool match = true;
        for (uint i = 0; i < signature_length; ++i) {
            if (expected[i] != measurement[i]) {
                match = false;
                break;
            }
        }
        if (match) return { val };
    }
    
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
                
                // Verify doubleton
                vector<bool> s1 = getSignature(val1, signature_length);
                vector<bool> s2 = getSignature(val2, signature_length);
                bool match = true;
                for (uint i = 0; i < signature_length; ++i) {
                    if ((s1[i] ^ s2[i]) != measurement[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) return { val1, val2 };
            }
        }
    }
    
    return {};
}


/**
 * @brief Base class for the Sparse All-Fast Fourier Transform (SAFFRON) recovery algorithm.
 * 
 * Provides core mechanisms for pooling, signature generation, and the iterative peeling 
 * algorithm used to recover sparse components (singletons and doubletons) from 
 * XOR-sum binary residuals across multiple pools.
 */
class Saffron {
protected:
    /** @brief The pooling matrix defining the item-to-pool and pool-to-item mappings. */
    PoolingMatrix pools_; 
    /** @brief Total number of features/items (n). */
    uint num_features_; 
    /** @brief Expected sparsity level (k). */
    uint sparsity_; 
    /** @brief Total number of pools (m). */
    uint num_pools_; 
    /** @brief Length of the SAFFRON signatures. */
    uint signature_length_; 
    /** @brief Debug verbosity level. */
    int debug_ = 0; 

public:
    /**
     * @brief Initialises the SAFFRON algorithm: builds the pooling matrix and computes signature parameters.
     *
     * @param num_features  Total number of items/features in the dataset (n).
     * @param sparsity      Expected number of nearest neighbours to recover (k); controls pool count when num_pools == 0.
     * @param num_pools     Number of measurement pools (m); if 0, defaults to sparsity * NUM_POOLS_COEFF.
     * @param pools_per_item Number of pools each item is randomly assigned to (default POOLS_PER_ITEM).
     * @param debug         Debug verbosity level (0 = silent).
     */
    Saffron(
        uint num_features,
        uint sparsity,
        uint num_pools = 0,
        uint pools_per_item = POOLS_PER_ITEM,
        int debug = 0
    ) :
        num_features_(num_features),
        sparsity_(sparsity),
        num_pools_(num_pools),
        debug_(debug)
    {
        uint L = ceil(log2(num_features));
        signature_length_ = 3 * (2 * L + 1);
        if (num_pools_ == 0) {
            num_pools_ = sparsity * NUM_POOLS_COEFF;
        }
        pools_ = computePools(num_features_, num_pools_, pools_per_item, debug_);
    }

    /**
     * @brief Returns the total number of features (n).
     * @return uint Number of features.
     */
    inline uint num_features() const { return num_features_; }

    /**
     * @brief Returns the total number of features (n). Alias for num_features().
     * @return uint Number of features.
     */
    inline uint size() const { return num_features_; }

    /**
     * @brief Returns the expected sparsity level (k).
     * @return uint Sparsity level.
     */
    inline uint sparsity() const { return sparsity_; }

    /**
     * @brief Returns the total number of pools (m).
     * @return uint Pool count.
     */
    inline uint num_pools() const { return num_pools_; }

    /**
     * @brief Returns the length of the signatures.
     * @return uint Signature length.
     */
    inline uint signature_length() const { return signature_length_; }

    ~Saffron() = default;

    /**
     * @brief Executes the peeling algorithm to recover identified items from residuals.
     * 
     * The peeling algorithm iteratively identifies singletons and doubletons from 
     * pool residuals, peels them off by XORing their signatures, and repeats until 
     * no more items can be identified.
     * 
     * @param residuals A 2D vector of booleans (num_pools x signature_bits).
     * @param debug Local debug level.
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

        // Initialize queue with all identifiable items from initial residuals
        for (uint pool_idx = 0; pool_idx < pools_.num_pools; ++pool_idx) {
            check_pool(pool_idx);
        }

        if (debug_ > 0 || debug > 0) {
            cout << "Initialized peeling with " << candidates.size() 
                << " candidates." << endl;
        }

        while (!candidates.empty()) {
            Candidate cand = candidates.front();
            candidates.pop();

            if (identified.count(cand.item_idx)) continue;
            
            identified.insert(cand.item_idx);
            
            // Peel off the identified item from all pools it belongs to
            vector<bool> sig = getSignature(cand.item_idx, signature_length_);
            for (uint p_idx : pools_.items_to_pools[cand.item_idx]) {
                for (uint b = 0; b < signature_length_; ++b) {
                    residuals[p_idx][b] = residuals[p_idx][b] ^ sig[b];
                }
                // Re-check the modified pool for new identifiable items
                check_pool(p_idx);
            }
        }
        return identified;
    }
};


#endif /* F938C541_109E_4EC0_AE90_E0FED079CBC9 */

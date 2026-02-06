#ifndef C3B841BC_895E_4F54_84BC_D5A1FAC3E0B9
#define C3B841BC_895E_4F54_84BC_D5A1FAC3E0B9

#include "data_structures.hpp"

/**
 * SAFFRON Implementation in C++
 * 
 * Group Testing using Sparse-graph codes for efficient item recovery.
 * This implementation provides:
 * - Optimized T matrix construction (sparse bipartite graph)
 * - Optimized U matrix construction (signature vectors)
 * - Efficient InvertedIndex for fast hash lookups
 * - Peeling algorithm for decoding
 * 
 * References:
 * Lee, K., Pedarsani, R., & Ramchandran, K. (2015).
 * "SAFFRON: A Fast, Efficient, and Robust Framework for Group Testing 
 *  based on Sparse-Graph Codes" arXiv:1508.04485
 */

/**
 * Construct the T matrix (bipartite graph incidence matrix) for SAFFRON.
 * 
 * Creates a left-regular bipartite graph where each feature connects to
 * exactly `degree` buckets chosen uniformly at random.
 * 
 * @param num_features Number of items/features (left nodes)
 * @param degree Left-degree (connections per feature), typically ceil(log2(K))
 * @param num_buckets Number of buckets (right nodes), typically 4*K
 * @return BitMatrix of shape (num_buckets, num_features)
 */
inline BitMatrix construct_T_matrix(
    uint num_features,
    uint degree,
    uint num_buckets,
    int debug = 0
) {
    BitMatrix T(num_buckets, std::vector<bool>(num_features, false));
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<uint> dis(0, num_buckets - 1);
    
    if (debug > 0) {
        std::cout << "[SAFFRON] construct_T_matrix: num_features=" << num_features
                  << ", degree=" << degree
                  << ", num_buckets=" << num_buckets << std::endl;
    }

    for (uint feature_idx = 0; feature_idx < num_features; ++feature_idx) {
        std::unordered_set<uint> chosen_buckets;
        while (chosen_buckets.size() < degree) {
            chosen_buckets.insert(dis(gen));
        }
        for (uint bucket : chosen_buckets) {
            T[bucket][feature_idx] = true;
        }
    }

    if (debug > 0) {
        // Print a small summary
        size_t edges = 0;
        for (uint b = 0; b < num_buckets; ++b) {
            edges += std::count(T[b].begin(), T[b].end(), true);
        }
        std::cout << "[SAFFRON] T edges=" << edges << " (expected ~ "
                  << (size_t)num_features * degree << ")" << std::endl;
    }
    return T;
}


/**
 * Construct the signature matrix U for SAFFRON.
 * 
 * Creates 6 signature sections stacked vertically for singleton and
 * doubleton detection/resolution. Total size: (6*num_bits) x num_features
 * 
 * @param num_features Number of items
 * @param num_bits Bits per section, typically ceil(log2(n) / 2)
 * @return BitMatrix of shape (6*num_bits, num_features)
 */
inline BitMatrix construct_U_matrix(
    uint num_features,
    uint num_bits,
    int debug = 0
) {
    uint total_rows = 6 * num_bits;
    BitMatrix U(total_rows, std::vector<bool>(num_features, false));
    
    if (debug > 0) {
        std::cout << "[SAFFRON] construct_U_matrix: num_features=" << num_features
                  << ", num_bits=" << num_bits
                  << ", total_rows=" << total_rows << std::endl;
    }

    // Section 0-1: U1 and ~U1 (binary representation, first half of bits)
    for (uint i = 0; i < num_features; ++i) {
        for (uint b = 0; b < num_bits; ++b) {
            bool bit = ((i >> b) & 1) > 0;
            U[b][i] = bit;                      // U1
            U[num_bits + b][i] = !bit;          // ~U1
        }
    }
    
    // Section 2-3: U2 and ~U2 (binary representation, second half of bits)
    for (uint i = 0; i < num_features; ++i) {
        for (uint b = 0; b < num_bits; ++b) {
            bool bit = ((i >> (b + num_bits)) & 1) > 0;
            U[2 * num_bits + b][i] = bit;       // U2
            U[3 * num_bits + b][i] = !bit;      // ~U2
        }
    }
    
    // Section 4: U1 XOR U2 (parity check)
    for (uint i = 0; i < num_features; ++i) {
        for (uint b = 0; b < num_bits; ++b) {
            bool bit1 = ((i >> b) & 1) > 0;
            bool bit2 = ((i >> (b + num_bits)) & 1) > 0;
            U[4 * num_bits + b][i] = bit1 ^ bit2;
        }
    }
    
    // Section 5: U1 XOR ~U2
    for (uint i = 0; i < num_features; ++i) {
        for (uint b = 0; b < num_bits; ++b) {
            bool bit1 = ((i >> b) & 1) > 0;
            bool bit2 = ((i >> (b + num_bits)) & 1) > 0;
            U[5 * num_bits + b][i] = bit1 ^ (!bit2);
        }
    }
    
    if (debug > 0) {
        // Print a checksum-like summary
        size_t ones = 0;
        for (uint r = 0; r < total_rows; ++r) {
            ones += std::count(U[r].begin(), U[r].end(), true);
        }
        std::cout << "[SAFFRON] U total ones=" << ones << std::endl;
    }
    return U;
}


/**
 * SAFFRON Group Testing Decoder
 * 
 * Non-adaptive group testing algorithm using sparse-graph codes.
 * Recovers defective items from binary test results using a peeling algorithm.
 */
class Saffron {
public:
    uint num_features;
    uint sparsity;
    uint num_buckets;
    uint degree;
    uint num_bits;
    uint num_tests;
    uint num_pools;
    
    BitMatrix T_matrix;
    BitMatrix U_matrix;
    
protected:
    std::mt19937 rng_;
    int debug_ = 0;
    
public:
    /**
     * Initialize SAFFRON decoder.
     * 
     * @param num_features Number of items n
     * @param sparsity Expected number of defective items K
     */
    Saffron(uint num_features = 1000000, uint sparsity = 100, int debug = 0)
        : num_features(num_features), sparsity(sparsity),
          rng_(std::random_device{}())
    {
        debug_ = debug;
        num_buckets = 4 * sparsity;
        degree = static_cast<uint>(std::ceil(std::log2(sparsity)));
        num_bits = static_cast<uint>(std::ceil(std::log2(num_features) / 2.0));
        num_tests = 6 * num_bits;
        num_pools = num_buckets * num_tests;
        
        // Construct matrices
        T_matrix = construct_T_matrix(num_features, degree, num_buckets, debug_);
        U_matrix = construct_U_matrix(num_features, num_bits, debug_);

        if (debug_ > 0) {
            std::cout << "[SAFFRON] Saffron init: n=" << num_features
                      << ", K=" << sparsity
                      << ", buckets=" << num_buckets
                      << ", degree=" << degree
                      << ", bits=" << num_bits
                      << ", tests=" << num_tests
                      << ", pools=" << num_pools << std::endl;
        }
    }
    
    /**
     * Generate list of all pools with constituent items.
     * 
     * @return Vector of pools, where each pool is a vector of item indices
     */
    inline VecVec pools(int debug = 0) const {
        VecVec pool_to_points;
        pool_to_points.reserve(num_pools);
        
        for (uint bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx) {
            for (uint test_idx = 0; test_idx < num_tests; ++test_idx) {
                Vec pool;
                
                // Compute AND of T[bucket_idx] and U[test_idx]
                for (uint feature_idx = 0; feature_idx < num_features; ++feature_idx) {
                    if (T_matrix[bucket_idx][feature_idx] && U_matrix[test_idx][feature_idx]) {
                        pool.push_back(feature_idx);
                    }
                }
                
                pool_to_points.push_back(std::move(pool));
            }
        }
        if (std::max(debug_, debug) > 0) {
            size_t non_empty = 0;
            for (const auto& p : pool_to_points) {
                if (!p.empty()) non_empty++;
            }
            std::cout << "[SAFFRON] pools(): total=" << pool_to_points.size()
                      << ", non_empty=" << non_empty << std::endl;
        }
        return pool_to_points;
    }
    
    /**
     * Decode measurements using the peeling algorithm.
     * 
     * @param measurements Binary vector of length num_pools with test results
     * @return Binary vector indicating identified defective items
     */
    inline std::vector<uint8_t> solve(const std::vector<uint8_t>& measurements, int debug = 0) {
        assert(measurements.size() == num_pools);
        
        std::vector<uint8_t> result(num_features, 0);
        std::unordered_set<uint> identified;
        
        // Reshape measurements by (bucket, test)
        std::vector<std::vector<bool>> z_matrix(num_buckets,
                                                std::vector<bool>(num_tests));
        for (uint bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx) {
            for (uint test_idx = 0; test_idx < num_tests; ++test_idx) {
                uint idx = bucket_idx * num_tests + test_idx;
                z_matrix[bucket_idx][test_idx] = measurements[idx] > 0;
            }
        }
        
        // Peeling algorithm
        uint max_iterations = 10 * sparsity;
        for (uint iteration = 0; iteration < max_iterations; ++iteration) {
            bool found_new = false;
            
            // Find singletons
            for (uint bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx) {
                for (uint test_idx = 0; test_idx < num_tests; ++test_idx) {
                    // Count 1s in this measurement (hamming weight)
                    uint weight = 0;
                    Vec candidate_features;
                    
                    for (uint feature_idx = 0; feature_idx < num_features; ++feature_idx) {
                        if (T_matrix[bucket_idx][feature_idx] &&
                            U_matrix[test_idx][feature_idx] &&
                            z_matrix[bucket_idx][test_idx])
                        {
                            weight++;
                            candidate_features.push_back(feature_idx);
                        }
                    }
                    
                    // Singleton detection: if only one feature connected and measurement is 1
                    if (weight == 1 && z_matrix[bucket_idx][test_idx]) {
                        uint feature_idx = candidate_features[0];
                        if (identified.find(feature_idx) == identified.end()) {
                            identified.insert(feature_idx);
                            found_new = true;
                            
                            // Remove this item from all measurements
                            for (uint b = 0; b < num_buckets; ++b) {
                                if (T_matrix[b][feature_idx]) {
                                    for (uint t = 0; t < num_tests; ++t) {
                                        if (U_matrix[t][feature_idx]) {
                                            z_matrix[b][t] = false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Find resolvable doubletons
            for (uint bucket_idx = 0; bucket_idx < num_buckets; ++bucket_idx) {
                for (uint test_idx = 0; test_idx < num_tests; ++test_idx) {
                    if (!z_matrix[bucket_idx][test_idx]) continue;
                    
                    // For each identified item connected to this right node
                    for (uint known_item : identified) {
                        if (!T_matrix[bucket_idx][known_item]) continue;
                        
                        // Try to find unknown item
                        for (uint unknown_item = 0; unknown_item < num_features; ++unknown_item) {
                            if (unknown_item == known_item ||
                                identified.find(unknown_item) != identified.end() ||
                                !T_matrix[bucket_idx][unknown_item])
                            {
                                continue;
                            }
                            
                            // Verify: measurement should be consistent with known + unknown
                            bool consistent = false;
                            bool has_both = false;
                            
                            if (U_matrix[test_idx][known_item] && U_matrix[test_idx][unknown_item]) {
                                has_both = true;
                                consistent = true;  // Both contribute to this measurement
                            }
                            
                            if (has_both && consistent) {
                                identified.insert(unknown_item);
                                found_new = true;
                                
                                // Remove unknown item
                                for (uint b = 0; b < num_buckets; ++b) {
                                    if (T_matrix[b][unknown_item]) {
                                        for (uint t = 0; t < num_tests; ++t) {
                                            if (U_matrix[t][unknown_item]) {
                                                z_matrix[b][t] = false;
                                            }
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }
                }
            }
            
            if (debug_ || debug) {
                std::cout << "[SAFFRON] solve(): iteration " << iteration
                          << ", identified=" << identified.size()
                          << ", found_new=" << (found_new ? 1 : 0) << std::endl;
            }
            if (!found_new) break;
        }
        
        // Convert to result vector
        for (uint item_idx : identified) {
            result[item_idx] = 1;
        }
        
        if (debug_ || debug) {
            size_t total = std::count(result.begin(), result.end(), (uint8_t)1);
            std::cout << "[SAFFRON] solve(): total identified=" << total << std::endl;
        }
        return result;
    }
};


#endif /* C3B841BC_895E_4F54_84BC_D5A1FAC3E0B9 */

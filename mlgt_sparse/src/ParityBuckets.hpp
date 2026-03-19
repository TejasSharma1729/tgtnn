#ifndef D8058364_68BC_40DA_A13B_138D365E4EAC
#define D8058364_68BC_40DA_A13B_138D365E4EAC

#include "headers.hpp"
#include "HashFunction.hpp"

/**
 * @brief A memory-efficient parity bucket for Saffron.
 * 
 * Each bit in the `buckets` vector represents the parity (XOR sum) of items 
 * mapping to that bucket.
 */
struct ParityBuckets {
    uint num_buckets = (1 << NUM_HASH_BITS); // 2^num_hash_bits buckets
    vector<bool> buckets; ///< Bitset representing bucket parities.

    /**
     * @brief Construct a new Parity Buckets object.
     * @param num_buckets Number of buckets for the parity buckets (default: 2^NUM_HASH_BITS).
     */
    ParityBuckets(
        uint num_buckets = (1 << NUM_HASH_BITS)
    ) : num_buckets(num_buckets), buckets(num_buckets, false) {}

    /**
     * @brief Flips the parity bit at the given hash value's bucket (insertion/deletion).
     * 
     * In the XOR-sum space, insertion and deletion are identical (XORing with 1).
     * 
     * @param hash_value The hash value determining which bucket to flip.
     */
    inline void insert(uint hash_value) {
        buckets[hash_value] = !buckets[hash_value];
    }

    /**
     * @brief Gets the parity bit for a given hash value.
     * 
     * Used during search to determine if an odd number of similar items mapped 
     * to this LSH bucket.
     * 
     * @param hash_value The hash value to check.
     * @return true If parity is odd.
     * @return false If parity is even.
     */
    inline bool get_parity(uint hash_value) const {
        return buckets[hash_value];
    }
};

#endif /* D8058364_68BC_40DA_A13B_138D365E4EAC */

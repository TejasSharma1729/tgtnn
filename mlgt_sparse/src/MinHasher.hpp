#ifndef E74FF110_A8B3_4DF0_8E7F_4A5094D0BD66
#define E74FF110_A8B3_4DF0_8E7F_4A5094D0BD66

#include "headers.hpp"
#include "BaseHasher.hpp"

/**
 * @brief MinHash implementation for Jaccard similarity.
 * 
 * MinHash estimates the Jaccard similarity between sets by computing the minimum 
 * value of a random permutation of the set elements. In this implementation, 
 * it operates on the indices of non-zero features in a sparse vector.
 */
class MinHasher : public BaseHasher {
public:
    /** @brief Number of independent min-hash values to concatenate for each 'table'. */
    uint32_t hashes_per_table;
    /** @brief Power of 2 representing the range of the final hash values (e.g., 16 for 2^16). */
    uint32_t hash_range_pow;

    /**
     * @brief Construct a new Min Hasher.
     * @param nh Number of independent hash tables (results in `nh` final hash values).
     * @param hpt Number of min-hashes to combine per table (higher = lower false positive rate).
     * @param hrp Range power for the output hashes.
     * @param s Random seed.
     */
    MinHasher(uint32_t nh = 1, uint32_t hpt = 1, uint32_t hrp = 16, uint32_t s = 42)
        : BaseHasher(nh, s), hashes_per_table(hpt), hash_range_pow(hrp) {}

    /**
     * @brief Core hashing logic that handles the permutation and densification.
     * 
     * @tparam T The integer type used for feature indices.
     * @param result Pointer to the output array where hash values will be stored.
     * @param indices Pointer to the input feature indices.
     * @param len Number of features in the input vector.
     */
    template<typename T>
    void hash_internal(uint32_t *result, const T *indices, uint32_t len) const {
        uint32_t num_hashes_to_generate = num_hashes * hashes_per_table;
        if (num_hashes_to_generate == 0) {
            return;
        }

        std::vector<uint64_t> prelim_result(num_hashes_to_generate, UINT64_MAX);

        for (uint32_t i = 0; i < len; i++) {
            uint64_t val = (uint64_t)indices[i] ^ seed;
            val = splitmix64(val);
            // Map 64-bit hash to [0, num_hashes_to_generate) without division or overflow
            uint32_t binid = static_cast<uint32_t>((static_cast<uint128_t>(val) * static_cast<uint128_t>(num_hashes_to_generate)) >> 64);
            if (prelim_result[binid] > val) {
                prelim_result[binid] = val;
            }
        }

        // Densification: Fill empty bins by borrowing from neighbors
        for (uint32_t i = 0; i < num_hashes_to_generate; i++) {
            if (prelim_result[i] != UINT64_MAX) continue;
            uint64_t count = 0;
            uint64_t next = UINT64_MAX;
            while (next == UINT64_MAX && count < 100) {
                count++;
                uint32_t index = static_cast<uint32_t>(combine_hashes(i, count) % num_hashes_to_generate);
                next = prelim_result[index];
            }
            prelim_result[i] = (next == UINT64_MAX) ? 0 : next;
        }

        // Combine: Concatenate internal hashes into final table hashes
        for (uint32_t table = 0; table < num_hashes; table++) {
            uint64_t combined = prelim_result[table * hashes_per_table];
            for (uint32_t h = 1; h < hashes_per_table; h++) {
                combined = combine_hashes(combined, prelim_result[table * hashes_per_table + h]);
            }
            result[table] = static_cast<uint32_t>(combined >> (64 - hash_range_pow));
        }
    }

    /**
     * @brief Implementation of sparse hashing interface.
     * 
     * @param data Ignored for MinHash (only indices matter).
     * @param indices Pointer to CSR column indices.
     * @param nnz Number of non-zero elements.
     * @return vector<uint32_t> The computed set of hashes.
     */
    virtual vector<uint32_t> operator()(const float* data, const uint32_t* indices, uint32_t nnz) const override {
        vector<uint32_t> res(num_hashes);
        hash_internal(res.data(), indices, nnz);
        return res;
    }

    /**
     * @brief Implementation of dense hashing interface.
     * 
     * @param q Dense query vector. Elements with magnitude > 1e-9 are treated as present in the set.
     * @return vector<uint32_t> The computed set of hashes.
     */
    virtual vector<uint32_t> operator()(const Eigen::VectorXf& q) const override {
        vector<uint32_t> indices;
        for (int i = 0; i < q.size(); ++i) {
            if (std::abs(q(i)) > 1e-9) indices.push_back(i);
        }
        vector<uint32_t> res(num_hashes);
        hash_internal(res.data(), indices.data(), (uint32_t)indices.size());
        return res;
    }
    
    /**
     * @brief Legacy hash method for backward compatibility.
     * 
     * @param result Pointer to output array (uint64_t).
     * @param indices Pointer to feature indices (uint64_t).
     * @param len Number of features.
     */
    void hash(uint64_t *result, const uint64_t *indices, uint64_t len) const {
        vector<uint32_t> res(num_hashes);
        hash_internal(res.data(), indices, (uint32_t)len);
        for(uint32_t i=0; i<num_hashes; ++i) result[i] = res[i];
    }
};

#endif /* E74FF110_A8B3_4DF0_8E7F_4A5094D0BD66 */

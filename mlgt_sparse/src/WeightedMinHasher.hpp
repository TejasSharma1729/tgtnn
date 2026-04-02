#ifndef C0A80101_WEIGHTED_MINHASHER_HPP
#define C0A80101_WEIGHTED_MINHASHER_HPP

#include "headers.hpp"
#include "BaseHasher.hpp"

/**
 * @brief Weighted MinHash implementation using a BagMinHash-inspired approach.
 *
 * Estimates the Weighted Jaccard similarity between weighted sparse vectors.
 * Each non-zero feature is treated as a weighted element; the minimum of
 * a random exponential variable scaled by the inverse weight is used as the
 * hash value.  Multiple independent hash functions are combined into compound
 * table hashes to reduce the false-positive rate.
 */
class WeightedMinHasher : public BaseHasher {
public:
    /** @brief Number of independent min-hash values to concatenate per table. */
    uint32_t hashes_per_table;
    /** @brief Power of 2 representing the range of the final hash values (e.g., 16 → [0, 2^16)). */
    uint32_t hash_range_pow;

    /**
     * @brief Construct a new WeightedMinHasher.
     *
     * @param nh  Number of independent hash tables (number of final hash values returned).
     * @param hpt Number of internal min-hash values combined per table (higher = fewer collisions).
     * @param hrp Range power for output hash values; output is in [0, 2^hrp).
     * @param s   Random seed used for hash function generation.
     */
    WeightedMinHasher(uint32_t nh = 1, uint32_t hpt = 1, uint32_t hrp = 16, uint32_t s = 42)
        : BaseHasher(nh, s), hashes_per_table(hpt), hash_range_pow(hrp) {}

    /**
     * @brief Core hashing logic that computes weighted min-hash values.
     *
     * For each of the `num_hashes * hashes_per_table` hash functions, selects the
     * feature with the smallest −log(u)/weight (exponential racing) and records its
     * random tag.  The per-table tags are then combined with combine_hashes() and
     * truncated to `hash_range_pow` bits.
     *
     * @param result  Output array of length `num_hashes`; filled with the final hash values.
     * @param data    Pointer to the non-zero float weights of the sparse vector.
     * @param indices Pointer to the column indices of the non-zero elements.
     * @param len     Number of non-zero elements.
     */
    void hash_internal(uint32_t *result, const float *data, const uint32_t *indices, uint32_t len) const {
        if (len == 0) {
            std::fill(result, result + num_hashes, 0);
            return;
        }

        uint32_t total_hashes = num_hashes * hashes_per_table;
        std::vector<uint64_t> prelim_result(total_hashes);

        for (uint32_t h = 0; h < total_hashes; ++h) {
            uint64_t local_seed = seed ^ h;
            uint64_t best_tag = 0;
            double min_val = std::numeric_limits<double>::infinity();

            for (uint32_t i = 0; i < len; ++i) {
                float weight = data[i];
                if (weight <= 0) continue;
                
                uint32_t dim = indices[i];
                uint64_t h1 = splitmix64(static_cast<uint64_t>(dim) ^ local_seed);
                uint64_t h2 = splitmix64(h1);
                
                double u = (h1 & 0xFFFFFFFFFFFFFFFULL) / static_cast<double>(0xFFFFFFFFFFFFFFFULL);
                if (u <= 0) u = 1e-15; 
                
                double val = -std::log(u) / static_cast<double>(weight);
                
                if (val < min_val) {
                    min_val = val;
                    best_tag = h2;
                }
            }
            prelim_result[h] = best_tag;
        }

        // Combine hashes if hpt > 1
        for (uint32_t table = 0; table < num_hashes; table++) {
            uint64_t combined = prelim_result[table * hashes_per_table];
            for (uint32_t h = 1; h < hashes_per_table; h++) {
                combined = combine_hashes(combined, prelim_result[table * hashes_per_table + h]);
            }
            result[table] = static_cast<uint32_t>(combined >> (64 - hash_range_pow));
        }
    }

    /**
     * @brief Hash a sparse vector (CSR row format).
     *
     * @param data    Pointer to the non-zero float weights.
     * @param indices Pointer to the column indices of the non-zero elements.
     * @param nnz     Number of non-zero elements in this row.
     * @return vector<uint32_t> Vector of `num_hashes` weighted min-hash values.
     */
    virtual vector<uint32_t> operator()(const float* data, const uint32_t* indices, uint32_t nnz) const override {
        vector<uint32_t> res(num_hashes);
        hash_internal(res.data(), data, indices, nnz);
        return res;
    }

    /**
     * @brief Hash a dense query vector.
     *
     * Converts the dense vector to a sparse representation (positive entries only)
     * and delegates to hash_internal().
     *
     * @param q Dense Eigen query vector; only entries with value > 1e-9 are treated as non-zero.
     * @return vector<uint32_t> Vector of `num_hashes` weighted min-hash values.
     */
    virtual vector<uint32_t> operator()(const Eigen::VectorXf& q) const override {
        vector<uint32_t> indices;
        vector<float> data;
        for (int i = 0; i < q.size(); ++i) {
            if (q(i) > 1e-9) {
                indices.push_back(i);
                data.push_back(q(i));
            }
        }
        vector<uint32_t> res(num_hashes);
        hash_internal(res.data(), data.data(), indices.data(), (uint32_t)indices.size());
        return res;
    }
};

#endif

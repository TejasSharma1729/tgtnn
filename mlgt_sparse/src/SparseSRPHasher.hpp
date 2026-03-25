#ifndef D8EEFC4F_15A3_4114_B209_56D079040F36
#define D8EEFC4F_15A3_4114_B209_56D079040F36

#include "headers.hpp"
#include "BaseHasher.hpp"

/**
 * @brief Sparse Signed Random Projection (SRP) hasher.
 * 
 * Sparse SRP projects high-dimensional sparse vectors onto a lower-dimensional 
 * space using a random projection matrix. It estimates cosine similarity between 
 * vectors. This implementation generates weights on-the-fly to save memory.
 */
class SparseSRPHasher : public BaseHasher {
public:
    /** @brief Number of bits to pack into each integer hash value. */
    uint32_t num_bits;

    /**
     * @brief Construct a new Sparse SRP Hasher.
     * @param b Number of projection bits per hash (default 16).
     * @param s Random seed.
     * @param nh Number of independent hash values to generate.
     */
    SparseSRPHasher(uint32_t b = 16, uint32_t s = 42, uint32_t nh = 1) 
        : BaseHasher(nh, s), num_bits(b) {}

    /**
     * @brief Implementation of sparse hashing interface.
     * 
     * @param data Pointer to CSR non-zero float values.
     * @param indices Pointer to CSR column indices.
     * @param nnz Number of non-zero elements.
     * @return vector<uint32_t> The computed set of hashes.
     */
    virtual vector<uint32_t> operator()(const float* data, const uint32_t* indices, uint32_t nnz) const override {
        vector<uint32_t> res(num_hashes);
        for (uint32_t h = 0; h < num_hashes; ++h) {
            std::vector<float> sums(num_bits, 0.0f);
            uint32_t local_seed = seed ^ h;

            for (uint32_t i = 0; i < nnz; ++i) {
                uint32_t feature_idx = indices[i];
                float val = data[i];

                uint64_t random_bits = feature_idx ^ local_seed;
                random_bits = splitmix64(random_bits);

                for (uint32_t b = 0; b < num_bits; ++b) {
                    if ((random_bits >> (b % 64)) & 1) sums[b] += val;
                    else sums[b] -= val;
                    if (b % 64 == 63 && b + 1 < num_bits) random_bits = splitmix64(random_bits);
                }
            }

            uint32_t hash_val = 0;
            for (uint32_t b = 0; b < num_bits; ++b) {
                if (sums[b] > 0) hash_val |= (1 << b);
            }
            res[h] = hash_val;
        }
        return res;
    }

    /**
     * @brief Implementation of dense hashing interface.
     * 
     * @param q Dense query vector.
     * @return vector<uint32_t> The computed set of hashes.
     */
    virtual vector<uint32_t> operator()(const Eigen::VectorXf& q) const override {
        vector<uint32_t> res(num_hashes);
        for (uint32_t h = 0; h < num_hashes; ++h) {
            std::vector<float> sums(num_bits, 0.0f);
            uint32_t local_seed = seed ^ h;

            for (uint32_t d = 0; d < (uint32_t)q.size(); ++d) {
                float val = q(d);
                if (std::abs(val) < 1e-9) continue;

                uint64_t random_bits = d ^ local_seed;
                random_bits = splitmix64(random_bits);

                for (uint32_t b = 0; b < num_bits; ++b) {
                    if ((random_bits >> (b % 64)) & 1) sums[b] += val;
                    else sums[b] -= val;
                    if (b % 64 == 63 && b + 1 < num_bits) random_bits = splitmix64(random_bits);
                }
            }

            uint32_t hash_val = 0;
            for (uint32_t b = 0; b < num_bits; ++b) {
                if (sums[b] > 0) hash_val |= (1 << b);
            }
            res[h] = hash_val;
        }
        return res;
    }

    /**
     * @brief Legacy hash method for backward compatibility.
     * 
     * @param result Pointer to output array (uint64_t).
     * @param data Pointer to CSR non-zero values.
     * @param indices Pointer to CSR column indices.
     * @param nnz Number of non-zeros.
     */
    void hash(uint64_t *result, const float *data, const uint32_t *indices, uint32_t nnz) const {
        vector<uint32_t> res = operator()(data, indices, nnz);
        *result = res[0];
    }
};

#endif /* D8EEFC4F_15A3_4114_B209_56D079040F36 */

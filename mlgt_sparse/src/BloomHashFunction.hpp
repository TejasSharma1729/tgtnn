#ifndef A3D011D9_855C_4F02_8D21_8B13C3A4D315
#define A3D011D9_855C_4F02_8D21_8B13C3A4D315

#include "headers.hpp"
#include "BaseHasher.hpp"

const uint BLOOM_HASH_BITS = 20; 
const uint BLOOM_NUM_HASHES = 50; 
const uint BLOOM_THRESHOLD = 20; 

/**
 * @brief Bloom Filter-inspired hasher generating multiple compound hash values.
 * 
 * This hasher uses Sparse Signed Random Projection to map vectors into multiple 
 * integer hash values. It is the primary hasher used by the standard MLGT 
 * implementation.
 */
class BloomHashFunction : public BaseHasher {
public:
    /** @brief Number of bits packed into each compound SRP hash value. */
    uint num_bits;
    /** @brief Expected dimensionality of the input vectors. */
    uint dimension;
    /** @brief Minimum number of hash matches required for identification (mirrors MLGTSaffron threshold). */
    uint threshold;
    /** @brief Debug verbosity level (0 = silent). */
    int debug;

public:
    /**
     * @brief Construct a new Bloom Hash Function.
     *
     * @param dimension   Dimensionality of the input vectors.
     * @param num_hashes  Number of independent compound hash functions to compute.
     * @param num_bits    Number of SRP projection bits packed into each hash value.
     * @param threshold   Minimum hash-match count for candidate identification (informational; used by callers).
     * @param debug       Debug verbosity level (0 = silent).
     * @param seed        Random seed; if 0, a hardware random seed is drawn automatically.
     */
    BloomHashFunction(
        uint dimension,
        uint num_hashes = BLOOM_NUM_HASHES, 
        uint num_bits = BLOOM_HASH_BITS,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0,
        uint seed = 0
    ) :
        BaseHasher(num_hashes, seed),
        num_bits(num_bits),
        dimension(dimension),
        threshold(threshold),
        debug(debug)
    {
        if (seed == 0) {
            std::random_device rd;
            this->seed = rd();
        }
    }

    /**
     * @brief Internal helper to compute a single compound hash value.
     * 
     * @param data Pointer to CSR float values.
     * @param indices Pointer to CSR column indices.
     * @param nnz Number of non-zero elements.
     * @param hash_idx Index of the hash function to use.
     * @return uint The computed hash value.
     */
    inline uint compute_single_hash(
        const float* data, 
        const uint32_t* indices, 
        uint32_t nnz, 
        uint32_t hash_idx
    ) const {
        vector<float> sums(num_bits, 0.0f);
        uint32_t local_seed = seed ^ hash_idx;

        for (uint32_t i = 0; i < nnz; ++i) {
            uint32_t feature_idx = indices[i];
            float val = data[i];

            uint64_t random_bits = feature_idx ^ local_seed;
            random_bits = (random_bits ^ (random_bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
            random_bits = (random_bits ^ (random_bits >> 27)) * 0x94d049bb133111ebULL;
            random_bits = random_bits ^ (random_bits >> 31);

            for (uint32_t b = 0; b < num_bits; ++b) {
                if ((random_bits >> (b % 64)) & 1) {
                    sums[b] += val;
                } else {
                    sums[b] -= val;
                }
                if (b % 64 == 63 && b + 1 < num_bits) {
                    random_bits = (random_bits ^ (random_bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
                }
            }
        }

        uint hash_val = 0;
        for (uint32_t b = 0; b < num_bits; ++b) {
            if (sums[b] > 0) {
                hash_val |= (1 << b);
            }
        }
        return hash_val;
    }

    /**
     * @brief Implementation of sparse hashing interface.
     * 
     * @param data Pointer to CSR non-zero float values.
     * @param indices Pointer to CSR column indices.
     * @param nnz Number of non-zero elements.
     * @return vector<uint32_t> The computed set of compound hashes.
     */
    virtual vector<uint> operator()(
        const float* data, 
        const uint32_t* indices, 
        uint32_t nnz
    ) const override {
        vector<uint> res(num_hashes);
        for (uint h = 0; h < num_hashes; ++h) {
            res[h] = compute_single_hash(data, indices, nnz, h);
        }
        return res;
    }

    /**
     * @brief Implementation of dense hashing interface.
     * 
     * @param q Dense query vector.
     * @return vector<uint32_t> The computed set of compound hashes.
     */
    virtual vector<uint> operator()(const Eigen::VectorXf& q) const override {
        vector<uint> res(num_hashes);
        for (uint h = 0; h < num_hashes; ++h) {
            vector<float> sums(num_bits, 0.0f);
            uint32_t local_seed = seed ^ h;
            
            for (uint32_t d = 0; d < (uint32_t)q.size(); ++d) {
                float val = q(d);
                if (val == 0) continue;

                uint64_t random_bits = d ^ local_seed;
                random_bits = (random_bits ^ (random_bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
                random_bits = (random_bits ^ (random_bits >> 27)) * 0x94d049bb133111ebULL;
                random_bits = random_bits ^ (random_bits >> 31);

                for (uint32_t b = 0; b < num_bits; ++b) {
                    if ((random_bits >> (b % 64)) & 1) {
                        sums[b] += val;
                    } else {
                        sums[b] -= val;
                    }
                    if (b % 64 == 63 && b + 1 < num_bits) {
                        random_bits = (random_bits ^ (random_bits >> 30)) * 0xbf58476d1ce4e5b9ULL;
                    }
                }
            }

            uint hash_val = 0;
            for (uint32_t b = 0; b < num_bits; ++b) {
                if (sums[b] > 0) {
                    hash_val |= (1 << b);
                }
            }
            res[h] = hash_val;
        }
        return res;
    }
};

#endif /* A3D011D9_855C_4F02_8D21_8B13C3A4D315 */

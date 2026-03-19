#ifndef F7F18F5B_4350_4678_A004_3EE22E17C64E
#define F7F18F5B_4350_4678_A004_3EE22E17C64E

#include "headers.hpp"
#include "BaseHasher.hpp"

/**
 * @brief Dense Signed Random Projection (SRP).
 * For dense input vectors. Stores a dense projection matrix for speed if requested,
 * or generates on-the-fly.
 */
class DenseSRPHasher : public BaseHasher {
public:
    uint32_t num_bits;
    uint32_t dimension;
    std::vector<std::vector<std::vector<int8_t>>> stored_weights; // [num_hashes][num_bits][dimension]

    DenseSRPHasher(uint32_t b = 16, uint32_t d = 0, uint32_t s = 42, uint32_t nh = 1, bool store = false) 
        : BaseHasher(nh, s), num_bits(b), dimension(d) {
        if (store && d > 0) {
            stored_weights.resize(num_hashes, std::vector<std::vector<int8_t>>(num_bits, std::vector<int8_t>(dimension)));
            for (uint32_t h = 0; h < num_hashes; ++h) {
                for (uint32_t i = 0; i < num_bits; ++i) {
                    uint32_t bit_seed = seed ^ h ^ i;
                    std::mt19937 gen(bit_seed);
                    std::bernoulli_distribution dist(0.5);
                    for (uint32_t j = 0; j < dimension; ++j) {
                        stored_weights[h][i][j] = dist(gen) ? 1 : -1;
                    }
                }
            }
        }
    }

    inline vector<uint32_t> hash_dense(const float* data) const {
        vector<uint32_t> res(num_hashes);
        for (uint32_t h = 0; h < num_hashes; ++h) {
            uint32_t hash_val = 0;
            if (!stored_weights.empty()) {
                for (uint32_t b = 0; b < num_bits; ++b) {
                    float sum = 0;
                    const int8_t* weights = stored_weights[h][b].data();
                    for (uint32_t d = 0; d < dimension; ++d) {
                        sum += data[d] * weights[d];
                    }
                    if (sum > 0) hash_val |= (1 << b);
                }
            } else {
                std::vector<float> sums(num_bits, 0.0f);
                uint32_t local_seed = seed ^ h;
                for (uint32_t d = 0; d < dimension; ++d) {
                    float val = data[d];
                    if (std::abs(val) < 1e-9) continue;
                    uint64_t random_bits = splitmix64(d ^ local_seed);
                    for (uint32_t b = 0; b < num_bits; ++b) {
                        if ((random_bits >> (b % 64)) & 1) sums[b] += val;
                        else sums[b] -= val;
                        if (b % 64 == 63 && b + 1 < num_bits) random_bits = splitmix64(random_bits);
                    }
                }
                for (uint32_t b = 0; b < num_bits; ++b) {
                    if (sums[b] > 0) hash_val |= (1 << b);
                }
            }
            res[h] = hash_val;
        }
        return res;
    }

    virtual vector<uint32_t> operator()(const float* data, const uint32_t* indices, uint32_t nnz) const override {
        // Even though it's a dense hasher, we can implement sparse input support
        vector<uint32_t> res(num_hashes);
        for (uint32_t h = 0; h < num_hashes; ++h) {
            uint32_t hash_val = 0;
            if (!stored_weights.empty()) {
                for (uint32_t b = 0; b < num_bits; ++b) {
                    float sum = 0;
                    const int8_t* weights = stored_weights[h][b].data();
                    for (uint32_t i = 0; i < nnz; ++i) {
                        sum += data[i] * weights[indices[i]];
                    }
                    if (sum > 0) hash_val |= (1 << b);
                }
            } else {
                std::vector<float> sums(num_bits, 0.0f);
                uint32_t local_seed = seed ^ h;
                for (uint32_t i = 0; i < nnz; ++i) {
                    float val = data[i];
                    uint32_t d = indices[i];
                    uint64_t random_bits = splitmix64(d ^ local_seed);
                    for (uint32_t b = 0; b < num_bits; ++b) {
                        if ((random_bits >> (b % 64)) & 1) sums[b] += val;
                        else sums[b] -= val;
                        if (b % 64 == 63 && b + 1 < num_bits) random_bits = splitmix64(random_bits);
                    }
                }
                for (uint32_t b = 0; b < num_bits; ++b) {
                    if (sums[b] > 0) hash_val |= (1 << b);
                }
            }
            res[h] = hash_val;
        }
        return res;
    }

    virtual vector<uint32_t> operator()(const Eigen::VectorXf& q) const override {
        return hash_dense(q.data());
    }

    // Original hash method for backward compatibility/pybind
    void hash(uint64_t *result, const float *data) const {
        vector<uint32_t> res = hash_dense(data);
        *result = res[0];
    }
};

#endif /* F7F18F5B_4350_4678_A004_3EE22E17C64E */

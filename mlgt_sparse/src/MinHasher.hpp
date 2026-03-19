#ifndef E74FF110_A8B3_4DF0_8E7F_4A5094D0BD66
#define E74FF110_A8B3_4DF0_8E7F_4A5094D0BD66

#include "headers.hpp"

#include "BaseHasher.hpp"

/**
 * @brief MinHash implementation for sets/sparse binary vectors.
 * Does not check values (assumes all non-zero values are 1).
 */
class MinHasher : public BaseHasher {
public:
    uint32_t hashes_per_table;
    uint32_t hash_range_pow;

    MinHasher(uint32_t nh = 1, uint32_t hpt = 1, uint32_t hrp = 16, uint32_t s = 42)
        : BaseHasher(nh, s), hashes_per_table(hpt), hash_range_pow(hrp) {}

    template<typename T>
    void hash_internal(uint32_t *result, const T *indices, uint32_t len) const {
        uint32_t num_hashes_to_generate = num_hashes * hashes_per_table;
        std::vector<uint64_t> prelim_result(num_hashes_to_generate, UINT64_MAX);
        uint64_t binsize = (UINT64_MAX / num_hashes_to_generate) + 1;

        for (uint32_t i = 0; i < len; i++) {
            uint64_t val = (uint64_t)indices[i] ^ seed;
            val = splitmix64(val);
            uint32_t binid = (uint32_t)std::min(val / binsize, (uint64_t)num_hashes_to_generate - 1);
            if (prelim_result[binid] > val) {
                prelim_result[binid] = val;
            }
        }

        // Densification
        for (uint32_t i = 0; i < num_hashes_to_generate; i++) {
            if (prelim_result[i] != UINT64_MAX) continue;
            uint64_t count = 0;
            uint64_t next = UINT64_MAX;
            while (next == UINT64_MAX && count < 100) {
                count++;
                uint32_t index = (uint32_t)(combine_hashes(i, count) % num_hashes_to_generate);
                next = prelim_result[index];
            }
            prelim_result[i] = (next == UINT64_MAX) ? 0 : next;
        }

        // Combine
        for (uint32_t table = 0; table < num_hashes; table++) {
            uint64_t combined = prelim_result[table * hashes_per_table];
            for (uint32_t h = 1; h < hashes_per_table; h++) {
                combined = combine_hashes(combined, prelim_result[table * hashes_per_table + h]);
            }
            result[table] = (uint32_t)(combined >> (64 - hash_range_pow));
        }
    }

    virtual vector<uint32_t> operator()(const float* data, const uint32_t* indices, uint32_t nnz) const override {
        vector<uint32_t> res(num_hashes);
        hash_internal(res.data(), indices, nnz);
        return res;
    }

    virtual vector<uint32_t> operator()(const Eigen::VectorXf& q) const override {
        // For dense queries in MinHash, we treat non-zero (or above epsilon) elements as the set.
        vector<uint32_t> indices;
        for (int i = 0; i < q.size(); ++i) {
            if (std::abs(q(i)) > 1e-9) indices.push_back(i);
        }
        vector<uint32_t> res(num_hashes);
        hash_internal(res.data(), indices.data(), (uint32_t)indices.size());
        return res;
    }
    
    // Original hash method for backward compatibility/pybind
    void hash(uint64_t *result, const uint64_t *indices, uint64_t len) const {
        vector<uint32_t> res(num_hashes);
        hash_internal(res.data(), indices, (uint32_t)len);
        for(uint32_t i=0; i<num_hashes; ++i) result[i] = res[i];
    }
};

#endif /* E74FF110_A8B3_4DF0_8E7F_4A5094D0BD66 */

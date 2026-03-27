#ifndef C0A80101_WEIGHTED_MINHASHER_HPP
#define C0A80101_WEIGHTED_MINHASHER_HPP

#include "headers.hpp"
#include "BaseHasher.hpp"

/**
 * @brief Weighted MinHash implementation using a robust BagMinHash-inspired approach.
 */
class WeightedMinHasher : public BaseHasher {
public:
    uint32_t hashes_per_table;
    uint32_t hash_range_pow;

    WeightedMinHasher(uint32_t nh = 1, uint32_t hpt = 1, uint32_t hrp = 16, uint32_t s = 42)
        : BaseHasher(nh, s), hashes_per_table(hpt), hash_range_pow(hrp) {}

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

    virtual vector<uint32_t> operator()(const float* data, const uint32_t* indices, uint32_t nnz) const override {
        vector<uint32_t> res(num_hashes);
        hash_internal(res.data(), data, indices, nnz);
        return res;
    }

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

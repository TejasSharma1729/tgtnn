#ifndef DA5CB10E_B4E9_4C3E_8AAC_48136FD60AC3
#define DA5CB10E_B4E9_4C3E_8AAC_48136FD60AC3

#include "data_structures.hpp"

class SimHash {
public:
    using HashValue = int32_t;

    SimHash(uint num_tables, uint num_bits, uint dimension, int debug = 0)
        : num_tables_(num_tables), num_bits_(num_bits), dimension_(dimension)
    {
        if (num_bits_ == 0 || num_tables_ == 0 || dimension_ == 0) {
            throw std::invalid_argument("SimHash: num_tables, num_bits and dimension must be > 0");
        }
        // allocate planes: flattened (table, bit, dim)
        planes_.resize(static_cast<size_t>(num_tables_) * num_bits_ * dimension_);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 1.0);

        for (size_t i = 0; i < planes_.size(); ++i) {
            planes_[i] = dist(gen);
        }
    }

    // Hash a single vector and return integer hash per table
    std::vector<HashValue> hash_vector(const std::vector<float>& vec, int debug = 0) const {
        if (vec.size() != dimension_) throw std::runtime_error("SimHash: input vector has wrong dimension");
        std::vector<HashValue> out(static_cast<size_t>(num_tables_), 0);

        for (uint t = 0; t < num_tables_; ++t) {
            int32_t value = 0;
            for (uint b = 0; b < num_bits_; ++b) {
                double dot = 0.0;
                size_t base = (static_cast<size_t>(t) * num_bits_ + b) * dimension_;
                for (uint d = 0; d < dimension_; ++d) {
                    dot += static_cast<double>(vec[d]) * planes_[base + d];
                }
                if (dot >= 0.0) {
                    value |= (1 << b);
                }
            }
            out[t] = static_cast<HashValue>(value);
        }
        return out;
    }

    // Hash many features
    std::vector<std::vector<HashValue>> hash_features(const std::vector<std::vector<float32_t>>& features, int debug = 0) const {
        std::vector<std::vector<HashValue>> result;
        result.reserve(features.size());
        for (const auto& f : features) {
            // convert float32_t to float for hash_vector signature
            std::vector<float> vf(f.begin(), f.end());
            result.push_back(hash_vector(vf, debug));
        }
        return result;
    }

    uint num_tables() const { return num_tables_; }
    uint num_bits() const { return num_bits_; }
    uint dimension() const { return dimension_; }

private:
    uint num_tables_;
    uint num_bits_;
    uint dimension_;
    std::vector<double> planes_;
};

#endif /* DA5CB10E_B4E9_4C3E_8AAC_48136FD60AC3 */

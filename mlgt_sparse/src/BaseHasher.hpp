#ifndef C0A80101_HASH_BASE_HPP
#define C0A80101_HASH_BASE_HPP

#include "headers.hpp"

/**
 * @brief Abstract base class for all MLGT hashers.
 * Ensures a consistent interface for sparse datasets and dense queries.
 */
class BaseHasher {
public:
    uint32_t num_hashes;
    uint32_t seed;

    BaseHasher(uint32_t nh, uint32_t s) : num_hashes(nh), seed(s) {}
    virtual ~BaseHasher() = default;

    /**
     * @brief Hash a sparse vector.
     */
    virtual vector<uint32_t> operator()(
        const float* data, 
        const uint32_t* indices, 
        uint32_t nnz
    ) const = 0;

    /**
     * @brief Hash a dense query vector.
     */
    virtual vector<uint32_t> operator()(const Eigen::VectorXf& q) const = 0;
};

template <typename H>
concept HasherType = std::is_base_of_v<BaseHasher, H>;

#endif

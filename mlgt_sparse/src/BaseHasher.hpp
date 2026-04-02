#ifndef C0A80101_HASH_BASE_HPP
#define C0A80101_HASH_BASE_HPP

#include "headers.hpp"

/**
 * @brief Abstract base class for all MLGT hashers.
 * 
 * A Hasher's role is to project high-dimensional vectors (either sparse or dense) 
 * into a set of discrete integer hash values. These hash values are used by the
 * PoolInvertedIndex to perform fast candidate identification.
 * 
 * All hashers in this module must support two interfaces:
 * 1. Sparse Input: Processing a single row of a CSR matrix (data, indices, nnz).
 * 2. Dense Input: Processing a dense Eigen query vector.
 */
class BaseHasher {
public:
    /** @brief Number of independent hash functions to compute per vector. */
    uint32_t num_hashes;
    /** @brief Seed for pseudo-random number generation. */
    uint32_t seed;

    /**
     * @brief Construct a new Base Hasher.
     * @param nh Number of hash values to generate.
     * @param s Random seed.
     */
    BaseHasher(uint32_t nh, uint32_t s) : num_hashes(nh), seed(s) {}
    virtual ~BaseHasher() = default;

    /**
     * @brief Hash a sparse vector (CSR row format).
     * 
     * @param data Pointer to the non-zero float values.
     * @param indices Pointer to the column indices of the non-zero values.
     * @param nnz Number of non-zero elements in this row.
     * @return vector<uint32_t> A vector of length `num_hashes` containing the computed hash values.
     */
    virtual vector<uint32_t> operator()(
        const float* data, 
        const uint32_t* indices, 
        uint32_t nnz
    ) const = 0;

    /**
     * @brief Hash a dense query vector.
     * 
     * @param q An Eigen::VectorXf representing the dense query.
     * @return vector<uint32_t> A vector of length `num_hashes` containing the computed hash values.
     */
    virtual vector<uint32_t> operator()(const Eigen::VectorXf& q) const = 0;
};

/**
 * @brief C++20 Concept to ensure a type inherits from BaseHasher.
 */
template <typename H>
concept HasherType = std::is_base_of_v<BaseHasher, H>;

#endif

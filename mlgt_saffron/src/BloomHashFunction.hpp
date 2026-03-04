#ifndef A3D011D9_855C_4F02_8D21_8B13C3A4D315
#define A3D011D9_855C_4F02_8D21_8B13C3A4D315

#include "headers.hpp"

const uint BLOOM_HASH_BITS = 16; 
const uint BLOOM_NUM_HASHES = 100; 
const uint BLOOM_THRESHOLD = 10; 

/**
 * @brief A Bloom Filter-inspired hash function that generates multiple compound hash values.
 * 
 * Uses random projections to generate multiple independent hash values for a given vector.
 * Each hash value is composed of multiple bits, where each bit is determined by the sign
 * of a random projection.
 */
class BloomHashFunction {
protected:
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> projection_matrix_; // [num_hashes * num_bits][dimension]
    uint num_hashes_; // Number of compound hash functions
    uint num_bits_; // Number of bits per hash function
    uint dimension_; // Input dimensionality
    uint threshold_; // Matching threshold (unused in hash generation)
    int debug_;

public:
    /**
     * @brief Initializes a BloomHashFunction.
     */
    BloomHashFunction(
        uint dimension,
        uint num_hashes = BLOOM_NUM_HASHES, 
        uint num_bits = BLOOM_HASH_BITS,
        uint threshold = BLOOM_THRESHOLD,
        int debug = 0
    ) :
        num_hashes_(num_hashes),
        num_bits_(num_bits),
        dimension_(dimension),
        threshold_(threshold),
        debug_(debug)
    {
        projection_matrix_.resize(num_hashes * num_bits, dimension);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0, 1.0);
        for (int i = 0; i < projection_matrix_.rows(); ++i) {
            for (int j = 0; j < projection_matrix_.cols(); ++j) {
                projection_matrix_(i, j) = dis(gen);
            }
        }
    }

    /**
     * @brief Computes multiple compound hash values for a given vector.
     * 
     * @param point The input vector as a float vector.
     * @return vector<uint> A vector of hash values (one per compound hash).
     */
    inline vector<uint> operator()(const vector<float>& point) const {
        assert(point.size() == dimension_ && "Point dimension mismatch");
        Eigen::Map<const Eigen::VectorXf> q(point.data(), dimension_);
        return operator()(q);
    }

    /**
     * @brief Computes multiple compound hash values for an Eigen vector.
     * 
     * Projects the input vector onto the internal projection matrix and 
     * generates multiple compact hash values.
     * 
     * @param q The input Eigen vector.
     * @return vector<uint> A vector of hash values (one per compound hash).
     */
    inline vector<uint> operator()(const Eigen::VectorXf& q) const {
        assert(q.size() == dimension_ && "Point dimension mismatch");
        Eigen::VectorXf projections = projection_matrix_ * q;
        vector<uint> res(num_hashes_, 0);
        for (uint h = 0; h < num_hashes_; ++h) {
            uint hash_val = 0;
            for (uint i = 0; i < num_bits_; ++i) {
                if (projections(h * num_bits_ + i) >= 0) {
                    hash_val |= (1 << i);
                }
            }
            res[h] = hash_val;
        }
        return res;
    }

    /**
     * @brief Computes multiple hash values for a given point (callable interface, numpy).
     * @param point_arr A 1D numpy array representing the point.
     * @return vector<uint> A vector of hash values (one per compound hash).
     */
    inline vector<uint> operator()(const pybind11::array_t<float> &point_arr) const {
        assert(point_arr.ndim() == 1 && "Point must be 1D");
        assert(point_arr.shape(0) == (ssize_t)dimension_ && "Point dimension mismatch");
        auto r = point_arr.unchecked<1>();
        Eigen::Map<const Eigen::VectorXf> q(r.data(0), dimension_);
        return operator()(q);
    }

    /**
     * @brief Returns the input dimensionality.
     * @return uint Dimension.
     */
    inline uint dimension() const { return dimension_; }

    /**
     * @brief Returns the target number of independent hash functions.
     * @return uint Count.
     */
    inline uint num_hashes() const { return num_hashes_; }

    /**
     * @brief Returns the number of bits packed into each compound hash value.
     * @return uint Bits per hash.
     */
    inline uint num_bits() const { return num_bits_; }
};

#endif /* A3D011D9_855C_4F02_8D21_8B13C3A4D315 */

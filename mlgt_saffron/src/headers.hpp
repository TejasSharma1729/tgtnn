#ifndef D06ED106_555D_4C4D_8C5F_06CF6C90B5F0
#define D06ED106_555D_4C4D_8C5F_06CF6C90B5F0

#include <bits/stdc++.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

typedef unsigned int uint;

const uint NUM_POOLS_COEFF = 10; // c_m; num_pools = K * c_m
const uint POOLS_PER_ITEM = 3; // c_d; each item appears in c_d pools
const uint SIGNATURE_COEFF = 6; // c_L; num_signature_bits =  ceil(log2(n)) * c_L
const uint NUM_HASH_BITS = 18; // Increased from 10 to 18 to reduce collisions in large datasets
using namespace std;


/**
 * @brief A Locality Sensitive Hashing (LSH) function using random projections.
 */
class HashFunction {
protected:
    vector<vector<float>> hash_bits_; // Each hash bit is defined by a random projection vector
    uint num_hash_bits_ = NUM_HASH_BITS; // Number of bits in the hash
    uint dimension_; // Dimensionality of the input points
    int debug_ = 0; // Debug level (0 for none, higher values for more verbose output)

    
public:
    /**
     * @brief Initializes a HashFunction with a given dimension and number of hash bits.
     * 
     * @param dimension The dimensionality of the input points.
     * @param num_hash_bits The number of bits in the generated hash (default: NUM_HASH_BITS).
     * @param debug Debug level (0 for none).
     */
    HashFunction(uint dimension, uint num_hash_bits = NUM_HASH_BITS, int debug = 0) :
        hash_bits_(num_hash_bits, vector<float>(dimension, 0.0f)),
        num_hash_bits_(num_hash_bits),
        dimension_(dimension),
        debug_(debug)
    {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dis(0.0, 1.0);
        for (uint i = 0; i < num_hash_bits_; ++i) {
            for (uint j = 0; j < dimension_; ++j) {
                hash_bits_[i][j] = dis(gen);
            }
        }
        if (debug_ > 0) {
            cout << "[HashFunction] Initialized with dimension=" << dimension_
                 << ", num_hash_bits=" << num_hash_bits_ << endl;
        }
    }

    /**
     * @brief Returns the dimension of the hash function.
     * @return uint Dimensionality.
     */
    inline uint dimension() const { return dimension_; }

    /**
     * @brief Returns the number of bits in the hash.
     * @return uint Number of bits.
     */
    inline uint num_hash_bits() const { return num_hash_bits_; }

    ~HashFunction() = default;

    /**
     * @brief Computes the hash value for a given point from a raw pointer.
     * 
     * @param data Pointer to the float array.
     * @param size Size of the array.
     * @return uint The computed hash value.
     */
    inline uint computeHash(const float* data, size_t size) const {
        assert(size == dimension_ && "Point dimension mismatch");
        uint hash_value = 0;
        for (uint i = 0; i < num_hash_bits_; ++i) {
            float dot_product = 0;
            for (uint j = 0; j < dimension_; ++j) {
                dot_product += hash_bits_[i][j] * data[j];
            }
            if (dot_product >= 0) {
                hash_value |= (1 << i);
            }
        }
        return hash_value;
    }

    /**
     * @brief Computes the hash value for a given point (vector version).
     * 
     * @param point A vector of floats representing the point.
     * @return uint The computed hash value.
     */
    inline uint computeHash(const vector<float>& point) const {
        return computeHash(point.data(), point.size());
    }

    /**
     * @brief Computes the hash value for a given point (numpy version).
     * 
     * @param point_arr A 1D numpy array of floats representing the point.
     * @return uint The computed hash value as an integer.
     */
    inline uint computeHash(pybind11::array_t<float> point_arr) const {
        auto r = point_arr.unchecked<1>();
        return computeHash(r.data(0), r.shape(0));
    }

    /**
     * @brief Computes the hash value for a given point (callable interface, numpy).
     * 
     * @param point_arr A 1D numpy array of floats representing the point.
     * @return uint The computed hash value.
     */
    inline uint operator()(pybind11::array_t<float> point_arr) const {
        return computeHash(point_arr);
    }

    /**
     * @brief Computes the hash value for a given point (callable interface, vector).
     * 
     * @param point A vector of floats representing the point.
     * @return uint The computed hash value.
     */
    inline uint operator()(const vector<float>& point) const {
        return computeHash(point);
    }
};


#endif /* D06ED106_555D_4C4D_8C5F_06CF6C90B5F0 */

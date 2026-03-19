#ifndef AE4F9A3D_7F73_445D_96D6_51A9705F40CF
#define AE4F9A3D_7F73_445D_96D6_51A9705F40CF

#include "headers.hpp"

const uint NUM_HASH_BITS = 18; 


/**
 * @brief A Locality Sensitive Hashing (LSH) function using random projections.
 * 
 * Maps high-dimensional vectors into integer hash values where similar vectors 
 * have a higher probability of sharing the same hash bit.
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
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dis(0.0, 1.0);
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
     * @brief Returns the dimension of the input points.
     * @return uint Dimensionality.
     */
    inline uint dimension() const { return dimension_; }

    /**
     * @brief Returns the number of bits in the generated hash.
     * @return uint Number of bits.
     */
    inline uint num_hash_bits() const { return num_hash_bits_; }

    ~HashFunction() = default;

protected:
    /**
     * @brief Computes the hash value for a given point from a raw pointer.
     * 
     * Uses random projections and bit-packing to generate a compact integer hash.
     * 
     * @param data Pointer to the float array representing the point.
     * @return uint The computed hash value as a bit-packed integer.
     */
    inline uint computeHash(const float* data) const {
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

public:
    /**
     * @brief Computes the hash value for a given point (callable interface, numpy).
     * 
     * @param point_arr A 1D numpy array of floats representing the point.
     * @return uint The computed hash value as a bit-packed integer.
     */
    inline uint operator()(pybind11::array_t<float> point_arr) const {
        assert(point_arr.shape(0) == dimension_ && "Point dimension mismatch");
        auto r = point_arr.unchecked<1>();
        return computeHash(r.data(0));
    }

    /**
     * @brief Computes the hash value for a given point (callable interface, vector).
     * 
     * @param point A vector of floats representing the point.
     * @return uint The computed hash value.
     */
    inline uint operator()(const vector<float>& point) const {
        assert(point.size() == dimension_ && "Point dimension mismatch");
        return computeHash(point.data());
    }
};




#endif /* AE4F9A3D_7F73_445D_96D6_51A9705F40CF */

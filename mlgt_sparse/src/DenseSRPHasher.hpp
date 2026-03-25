#ifndef F7F18F5B_4350_4678_A004_3EE22E17C64E
#define F7F18F5B_4350_4678_A004_3EE22E17C64E

#include "headers.hpp"
#include "BaseHasher.hpp"

/**
 * @brief Dense Signed Random Projection (SRP) hasher.
 * 
 * Stores a full projection matrix for high-precision locality sensitive hashing.
 * Optimized for dense vectors using Eigen's matrix multiplication, but supports
 * sparse inputs by indexing into the matrix rows.
 */
class DenseSRPHasher : public BaseHasher {
public:
    /** @brief Number of bits to pack into each integer hash value. */
    uint32_t num_bits;
    /** @brief Expected dimensionality of the input vectors. */
    uint32_t dimension;
    /** @brief The projection matrix [num_hashes * num_bits][dimension]. */
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> projection_matrix;

    /**
     * @brief Construct a new Dense SRP Hasher.
     * @param b Number of projection bits per hash.
     * @param d Dimensionality of input points.
     * @param s Random seed.
     * @param nh Number of independent hash values to generate.
     * @param store Unused (matrix is always stored in this robust version).
     */
    DenseSRPHasher(uint32_t b = 16, uint32_t d = 0, uint32_t s = 42, uint32_t nh = 1, bool store = true) 
        : BaseHasher(nh, s), num_bits(b), dimension(d) {
        if (d > 0) {
            projection_matrix.resize(num_hashes * num_bits, dimension);
            std::mt19937 gen(seed);
            std::normal_distribution<float> dist(0.0, 1.0);
            for (int i = 0; i < projection_matrix.rows(); ++i) {
                for (int j = 0; j < projection_matrix.cols(); ++j) {
                    projection_matrix(i, j) = dist(gen);
                }
            }
        }
    }

    /**
     * @brief Internal helper to hash a raw float pointer (dense).
     */
    inline vector<uint32_t> hash_dense(const float* data) const {
        Eigen::Map<const Eigen::VectorXf> q(data, dimension);
        return operator()(q);
    }

    /**
     * @brief Implementation of sparse hashing interface.
     */
    virtual vector<uint32_t> operator()(const float* data, const uint32_t* indices, uint32_t nnz) const override {
        vector<uint32_t> res(num_hashes, 0);
        vector<float> projections(num_hashes * num_bits, 0.0f);

        // Sparse-dense dot product against every row of the projection matrix
        for (uint32_t i = 0; i < nnz; ++i) {
            uint32_t col = indices[i];
            float val = data[i];
            for (uint32_t r = 0; r < num_hashes * num_bits; ++r) {
                projections[r] += val * projection_matrix(r, col);
            }
        }

        for (uint32_t h = 0; h < num_hashes; ++h) {
            uint32_t hash_val = 0;
            for (uint32_t b = 0; b < num_bits; ++b) {
                if (projections[h * num_bits + b] >= 0) {
                    hash_val |= (1 << b);
                }
            }
            res[h] = hash_val;
        }
        return res;
    }

    /**
     * @brief Implementation of dense hashing interface using Eigen BLAS.
     * 
     * @param q The query vector (sparse or dense)
     * @return vector<uint32_t> the computed set of hashes
     */
    virtual vector<uint32_t> operator()(const Eigen::VectorXf& q) const override {
        assert(q.size() == (int)dimension && "Dimension mismatch in DenseSRPHasher");
        Eigen::VectorXf projections = projection_matrix * q;
        
        vector<uint32_t> res(num_hashes, 0);
        for (uint32_t h = 0; h < num_hashes; ++h) {
            uint32_t hash_val = 0;
            for (uint32_t b = 0; b < num_bits; ++b) {
                if (projections(h * num_bits + b) >= 0) {
                    hash_val |= (1 << b);
                }
            }
            res[h] = hash_val;
        }
        return res;
    }

    /**
     * @brief Legacy hash method for backward compatibility.
     */
    void hash(uint64_t *result, const float *data) const {
        vector<uint32_t> res = hash_dense(data);
        *result = res[0];
    }
};

#endif /* F7F18F5B_4350_4678_A004_3EE22E17C64E */

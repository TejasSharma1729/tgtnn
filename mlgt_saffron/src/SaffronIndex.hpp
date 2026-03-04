#ifndef D36A5974_6993_4859_84C8_7DDFC47F3EC9
#define D36A5974_6993_4859_84C8_7DDFC47F3EC9

#include "headers.hpp"
#include "Saffron.hpp"
#include "HashFunction.hpp"
#include "ParityBuckets.hpp"


/**
 * @brief SaffronIndex implementation using parity buckets for rapid nearest neighbor search.
 * 
 * Combines standard SAFFRON principles with LSH parity buckets to enable
 * sub-linear time similarity search in high-dimensional vector spaces.
 */
class SaffronIndex : public Saffron {
protected:
    vector<HashFunction> hash_functions_; // One hash function per pool
    vector<vector<ParityBuckets>> bucket_indices_; // For each pool and bit position, the parity buckets
    vector<vector<float>> data_points_; // Original data points for scoring
    uint num_hash_bits_; // Number of bits for the hash functions
    uint dimension_; // Dimensionality of the data points
    bool normalize_; // Whether to normalize the data points

public:
    /**
     * @brief Initializes SaffronIndex with data points and indexing parameters.
     * 
     * @param data_points_arr A 2D numpy array of floats (num_points x dimension).
     * @param num_neighbors The target number of neighbors to recover (k).
     * @param num_hash_bits Number of bits for the hash functions.
     * @param debug Debug level.
     */
    SaffronIndex(
        pybind11::array_t<float> data_points_arr, 
        uint num_neighbors = 100, 
        uint num_hash_bits = NUM_HASH_BITS, 
        int debug = 0,
        bool normalize = true
    ) :
        Saffron(data_points_arr.shape(0), num_neighbors, debug),
        num_hash_bits_(num_hash_bits),
        dimension_(data_points_arr.shape(0) > 0 ? data_points_arr.shape(1) : 0),
        normalize_(normalize)
    {
        data_points_ = normalizeDataset(data_points_arr, normalize_);

        hash_functions_.reserve(num_pools_);
        for (uint i = 0; i < num_pools_; ++i) {
            hash_functions_.emplace_back(dimension_, num_hash_bits_, debug);
        }
        
        bucket_indices_.resize(num_pools_, vector<ParityBuckets>(signature_length_));
        for (uint item_idx = 0; item_idx < num_features_; ++item_idx) {
            vector<bool> signature = getSignature(item_idx, signature_length_);
            for (uint pool_idx : pools_.items_to_pools[item_idx]) {
                uint hash_val = hash_functions_[pool_idx](data_points_[item_idx]);
                for (uint bit_idx = 0; bit_idx < signature_length_; ++bit_idx) {
                    if (signature[bit_idx]) {
                        bucket_indices_[pool_idx][bit_idx].insert(hash_val);
                    }
                }
            }
            if (item_idx % 100000 == 0 && debug >= 1) {
                cout << "Indexed " << item_idx << " / " << num_features_ << " points..." << endl;
            }
        }
    }

    /**
     * @brief Returns the dimensionality of the data points.
     * @return uint 
     */
    inline uint dimension() const { return dimension_; }

    /**
     * @brief Returns the k value (sparsity/target neighbors).
     * @return uint 
     */
    inline uint k_val() const { return sparsity_; }

    ~SaffronIndex() = default;

protected:
    /**
     * @brief Computes parity-bucket residuals for a query vector.
     * 
     * Projects the query through each pool's hash function and checks the 
     * corresponding buckets in the parity indices.
     * 
     * @param query The normalized query vector.
     * @return vector<vector<bool>> residuals.
     */
    inline vector<vector<bool>> getResiduals(vector<float> &query) const {
        assert(query.size() == dimension_ && "Query dimension mismatch");
        vector<vector<bool>> residuals(num_pools_, vector<bool>(signature_length_, false));
        for (uint i = 0; i < num_pools_; ++i) {
            uint hash_val = hash_functions_[i](query);
            for (uint j = 0; j < signature_length_; ++j) {
                residuals[i][j] = bucket_indices_[i][j].get_parity(hash_val);
            }
        }
        if (debug_ > 0) {
            cout << "Computed residuals for query." << endl;
        }
        return residuals;
    }

    /**
     * @brief Scores identified items and returns the top K based on inner product.
     * 
     * @param query The query vector (numpy array).
     * @param identified A set of identified item indices.
     * @return vector<uint> Top K item indices.
     */
    inline vector<uint> getTopK(vector<float> &query, const set<uint> &identified) const {
        assert(query.size() == dimension_ && "Query dimension mismatch");
        vector<pair<float, uint>> scores;
        for (uint datapoint: identified) {
            float dp = std::inner_product(
                query.begin(), 
                query.end(), 
                data_points_[datapoint].begin(), 
                0.0f
            );
            scores.push_back({dp, datapoint});
        }
        sort(scores.rbegin(), scores.rend());
        vector<uint> topK;
        for (size_t i = 0; i < std::min((size_t)sparsity_, scores.size()); ++i) {
            topK.push_back(scores[i].second);
        }
        sort(topK.begin(), topK.end());
        if (debug_ > 0) {
            cout << "Scored " << identified.size() << " identified items." << endl;
        }
        return topK;
    }

public:
    /**
     * @brief Performs a full search for the given query vector.
     * 
     * @param query_arr The query point (numpy array).
     * @return vector<uint> Top K nearest neighbor indices.
     */
    inline vector<uint> search(pybind11::array_t<float> query_arr) {
        vector<float> query = normalizeDataPoint(query_arr, normalize_);
        assert(query.size() == dimension_ && "Query dimension mismatch");
        vector<vector<bool>> residuals = getResiduals(query);
        set<uint> identified = peelingAlgorithm(residuals);
        return getTopK(query, identified);
    }

    /**
     * @brief Performs a search using the callable interface.
     * 
     * @param query_arr The query point (numpy array).
     * @return vector<uint> Top K nearest neighbor indices.
     */
    inline vector<uint> operator()(pybind11::array_t<float> query_arr) {
        return search(query_arr);
    }
};


#endif /* D36A5974_6993_4859_84C8_7DDFC47F3EC9 */

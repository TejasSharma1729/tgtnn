#ifndef D36A5974_6993_4859_84C8_7DDFC47F3EC9
#define D36A5974_6993_4859_84C8_7DDFC47F3EC9

#include "headers.hpp"
#include "saffron.hpp"


/**
 * @brief A memory-efficient parity bucket for Saffron.
 */
struct ParityBuckets {
    vector<bool> buckets; // Using vector<bool> for memory efficiency; each bucket is a single bit
    static const uint num_buckets = (1 << NUM_HASH_BITS); // 2^num_hash_bits buckets

    ParityBuckets() : buckets(num_buckets, false) {}

    inline void insert(uint hash_value) {
        buckets[hash_value] = !buckets[hash_value];
    }

    inline bool get_parity(uint hash_value) const {
        return buckets[hash_value];
    }
};

/**
 * @brief An indexing structure combining Saffron with Parity Buckets for fast nearest neighbor search.
 */
class __attribute__((visibility("hidden"))) SaffronIndex : public Saffron {
    // The attribute visibility("hidden") is used to prevent symbol export in shared libraries, which can reduce binary size and improve load times when this class is not intended to be used outside of this module.
private:
    pybind11::array_t<float> data_points_arr_; // The data points

protected:
    vector<HashFunction> hash_functions_; // One hash function per pool
    vector<vector<ParityBuckets>> bucket_indices_; // For each pool and bit position, the parity buckets
    uint num_hash_bits_; // Number of bits for the hash functions
    uint dimension_; // Dimensionality of the data points

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
        data_points_arr_(data_points_arr),
        num_hash_bits_(num_hash_bits),
        dimension_(data_points_arr.shape(0) > 0 ? data_points_arr.shape(1) : 0)
    {
        auto r = data_points_arr_.mutable_unchecked<2>();
        
        if (normalize && num_features_ > 0) {
            if (debug_ >= 1) cout << "Normalizing data points..." << endl;
            for (uint i = 0; i < num_features_; ++i) {
                float norm_sq = 0;
                for (uint j = 0; j < dimension_; ++j) {
                    norm_sq += r(i, j) * r(i, j);
                }
                float norm = sqrt(norm_sq);
                if (norm > 1e-9) {
                    for (uint j = 0; j < dimension_; ++j) {
                        r(i, j) /= norm;
                    }
                }
            }
        }

        hash_functions_.reserve(num_pools_);
        for (uint i = 0; i < num_pools_; ++i) {
            hash_functions_.emplace_back(dimension_, num_hash_bits_, debug);
        }
        
        bucket_indices_.resize(num_pools_, vector<ParityBuckets>(signature_length_));
        for (uint item_idx = 0; item_idx < num_features_; ++item_idx) {
            vector<bool> signature = getSignature(item_idx, signature_length_);
            const float* pt_ptr = r.data(item_idx, 0);
            for (uint pool_idx : pools_.items_to_pools[item_idx]) {
                uint hash_val = hash_functions_[pool_idx].computeHash(pt_ptr, dimension_);
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
     * @brief Returns the number of data points.
     * @return uint 
     */
    inline uint size() const { return num_features_; }

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

    /**
     * @brief Computes the residuals for a given query vector.
     * 
     * @param query_arr The query point as a 1D numpy array of floats.
     * @return vector<vector<bool>> residuals.
     */
    inline vector<vector<bool>> getResiduals(pybind11::array_t<float> query_arr) const {
        auto r = query_arr.unchecked<1>();
        assert(r.shape(0) == dimension_ && "Query dimension mismatch");
        vector<vector<bool>> residuals(num_pools_, vector<bool>(signature_length_, false));
        for (uint i = 0; i < num_pools_; ++i) {
            uint hash_val = hash_functions_[i].computeHash(r.data(0), r.shape(0));
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
     * @param query_arr The query vector (numpy array).
     * @param identified A set of identified item indices.
     * @return vector<uint> Top K item indices.
     */
    inline vector<uint> getTopK(pybind11::array_t<float> query_arr, const set<uint> &identified) const {
        auto r = query_arr.unchecked<1>();
        auto data_proxy = data_points_arr_.unchecked<2>();
        assert(r.shape(0) == dimension_ && "Query dimension mismatch");
        vector<pair<float, uint>> scores;
        for (uint datapoint: identified) {
            float dp = 0;
            for (uint i = 0; i < dimension_; ++i) {
                dp += r(i) * data_proxy(datapoint, i);
            }
            scores.push_back({dp, datapoint});
        }
        sort(scores.rbegin(), scores.rend());
        vector<uint> topK;
        for (size_t i = 0; i < min((size_t)sparsity_, scores.size()); ++i) {
            topK.push_back(scores[i].second);
        }
        sort(topK.begin(), topK.end());
        if (debug_ > 0) {
            cout << "Scored " << identified.size() << " identified items." << endl;
        }
        return topK;
    }

    /**
     * @brief Performs a full search for the given query vector.
     * 
     * @param query_arr The query point (numpy array).
     * @return vector<uint> Top K nearest neighbor indices.
     */
    inline vector<uint> search(pybind11::array_t<float> query_arr) {
        auto r_orig = query_arr.mutable_unchecked<1>();
        const uint dim = r_orig.shape(0);
        
        // Normalize query
        vector<float> q_normed(dim);
        float q_norm_sq = 0;
        for (uint i = 0; i < dim; ++i) q_norm_sq += r_orig(i) * r_orig(i);
        float q_norm = sqrt(q_norm_sq);
        for (uint i = 0; i < dim; ++i) q_normed[i] = (q_norm > 1e-9) ? (r_orig(i) / q_norm) : r_orig(i);
        
        const float* q_ptr = q_normed.data();

        // Map from pool_idx to his hash value for the query
        vector<uint> query_hashes(num_pools_);
        // Map from pool_idx to the corresponding residual bits
        vector<vector<bool>> residuals(num_pools_, vector<bool>(signature_length_, false));

        for (uint i = 0; i < num_pools_; ++i) {
            query_hashes[i] = hash_functions_[i].computeHash(q_ptr, dimension_);
            for (uint j = 0; j < signature_length_; ++j) {
                residuals[i][j] = bucket_indices_[i][j].get_parity(query_hashes[i]);
            }
        }

        set<uint> identified;
        queue<Candidate> candidates;

        // Initial scan for candidates
        for (uint pool_idx = 0; pool_idx < num_pools_; ++pool_idx) {
            vector<uint> decoded = decodeSignature(residuals[pool_idx], signature_length_);
            for (uint item_idx : decoded) {
                if (item_idx < num_features_) {
                    candidates.push({pool_idx, item_idx});
                }
            }
        }

        auto data_proxy = data_points_arr_.unchecked<2>();

        while (!candidates.empty()) {
            Candidate cand = candidates.front();
            candidates.pop();

            if (identified.count(cand.item_idx)) {
                continue; 
            }
            identified.insert(cand.item_idx);

            const float* item_ptr = data_proxy.data(cand.item_idx, 0);

            // Peel this item from all pools where it collided with the query
            for (uint pool_idx : pools_.items_to_pools[cand.item_idx]) {
                // If the item lands in the same bucket as the query in this pool
                if (hash_functions_[pool_idx].computeHash(item_ptr, dimension_) == query_hashes[pool_idx]) {
                    vector<bool> signature = getSignature(cand.item_idx, signature_length_);
                    for (uint bit_idx = 0; bit_idx < signature_length_; ++bit_idx) {
                        if (signature[bit_idx]) {
                            residuals[pool_idx][bit_idx].flip();
                        }
                    }

                    // Check if this pool just became or still is a decodable pool
                    vector<uint> decoded = decodeSignature(residuals[pool_idx], signature_length_);
                    for (uint new_item_idx : decoded) {
                        if (new_item_idx < num_features_) {
                            candidates.push({pool_idx, new_item_idx});
                        }
                    }
                }
            }
        }

        return getTopK(query_arr, identified);
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

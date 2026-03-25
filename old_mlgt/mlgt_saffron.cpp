#include "data_structures.hpp"
#include "inverted_index.hpp"
#include "saffron.hpp"
#include "mlgt_saffron.hpp"
#include "simhash.hpp"
#include "mlgt.hpp"


/**
 * Convert numpy array to std::vector
 * Handles automatic extraction of buffer data and conversion
 */
template<typename T>
inline std::vector<T> numpy_to_vector(pybind11::array_t<T> arr) {
    auto buf = arr.request();
    auto* ptr = static_cast<T*>(buf.ptr);
    return std::vector<T>(ptr, ptr + buf.shape[0]);
}

/**
 * Convert std::vector to numpy array for return values
 */
template<typename T>
inline pybind11::array_t<T> vector_to_numpy(const std::vector<T>& vec) {
    return pybind11::array_t<T>(vec.size(), vec.data());
}

/**
 * Convert 2D vector of floats to numpy array
 */
inline pybind11::array_t<float> matrix_to_numpy(const std::vector<std::vector<float>>& mat) {
    if (mat.empty()) return pybind11::array_t<float>(0);
    pybind11::array_t<float> arr({mat.size(), mat[0].size()});
    auto buf = arr.request();
    auto* ptr = static_cast<float*>(buf.ptr);
    for (size_t i = 0; i < mat.size(); ++i) {
        std::copy(mat[i].begin(), mat[i].end(), ptr + i * mat[i].size());
    }
    return arr;
}


PYBIND11_MODULE(mlgt_saffron, m) {
    m.doc() = "SAFFRON Group Testing and MLGT C++ Implementation with optimized hash index queries";

    pybind11::class_<Saffron>(m, "Saffron",
        "SAFFRON Group Testing Decoder\n\n"
        "Non-adaptive group testing algorithm using sparse-graph codes.\n"
        "Recovers defective items from binary test results using a peeling algorithm.\n"
        "Theory: Lee, K., Pedarsani, R., & Ramchandran, K. (2015).\n"
        "'SAFFRON: A Fast, Efficient, and Robust Framework for Group Testing based on Sparse-Graph Codes'")
        .def(pybind11::init<unsigned int, unsigned int, int>(),
             pybind11::arg("num_features") = 1000000,
             pybind11::arg("sparsity") = 100,
             pybind11::arg("debug") = 0,
             "Initialize SAFFRON decoder with num_features items and sparsity parameter")
        
        // ===== Public Data Members =====
        .def_readonly("num_features", &Saffron::num_features,
                     "Number of items/features (left nodes of bipartite graph)")
        .def_readonly("sparsity", &Saffron::sparsity,
                     "Expected number of defective items (K parameter)")
        .def_readonly("num_buckets", &Saffron::num_buckets,
                     "Number of buckets/right nodes (4 * sparsity)")
        .def_readonly("degree", &Saffron::degree,
                     "Left-degree in bipartite graph (ceil(log2(K)))")
        .def_readonly("num_bits", &Saffron::num_bits,
                     "Bits per signature section (ceil(log2(n) / 2))")
        .def_readonly("num_tests", &Saffron::num_tests,
                     "Total test sections (6 * num_bits)")
        .def_readonly("num_pools", &Saffron::num_pools,
                     "Total number of pools (num_buckets * num_tests)")
        .def("T_matrix", [](Saffron& self) {
            // Convert BitMatrix to numpy bool array
            auto py_T = pybind11::array_t<bool>(
                std::vector<size_t>{(size_t)self.num_buckets, (size_t)self.num_features}
            );
            auto buf = py_T.request();
            auto* ptr = static_cast<bool*>(buf.ptr);
            for (uint i = 0; i < self.num_buckets; ++i) {
                for (uint j = 0; j < self.num_features; ++j) {
                    ptr[i * self.num_features + j] = self.T_matrix[i][j];
                }
            }
            return py_T;
        }, "T-matrix (bipartite graph incidence matrix).\n"
           "Shape: (num_buckets, num_features)\n"
           "Each feature connects to exactly degree buckets")
        .def("U_matrix", [](Saffron& self) {
            // Convert BitMatrix to numpy bool array
            uint rows = 6 * self.num_bits;
            auto py_U = pybind11::array_t<bool>(
                std::vector<size_t>{(size_t)rows, (size_t)self.num_features}
            );
            auto buf = py_U.request();
            auto* ptr = static_cast<bool*>(buf.ptr);
            for (uint i = 0; i < rows; ++i) {
                for (uint j = 0; j < self.num_features; ++j) {
                    ptr[i * self.num_features + j] = self.U_matrix[i][j];
                }
            }
            return py_U;
        }, "U-matrix (signature matrix) for singleton/doubleton detection.\n"
           "Shape: (6*num_bits, num_features)\n"
           "Contains 6 sections: U1, ~U1, U2, ~U2, U1^U2, U1^~U2")
        
        .def("pools", [](Saffron& self, int debug) {
            auto pools_data = self.pools(debug);
            pybind11::list py_pools;
            for (const auto& pool : pools_data) {
                pybind11::list py_pool;
                for (uint item : pool) {
                    py_pool.append(item);
                }
                py_pools.append(py_pool);
            }
            return py_pools;
        }, pybind11::arg("debug") = 0,
           "Generate list of all pools with constituent items.\n\n"
           "Returns: List of pools, where each pool is a list of item indices")
        
        .def("solve", [](Saffron& self, pybind11::array_t<uint8_t> measurements, int debug) {
            auto result = self.solve(numpy_to_vector(measurements), debug);
            return vector_to_numpy(result);
        }, pybind11::arg("measurements"), pybind11::arg("debug") = 0,
           "Decode measurements to recover defective items using peeling algorithm.\n\n"
           "Args:\n"
           "    measurements: Binary vector of test results (length = num_pools)\n\n"
           "Returns:\n"
           "    Binary vector indicating identified defective items");
    
    pybind11::class_<InvertedIndex>(m, "InvertedIndex",
        "Optimized inverted index for fast hash lookups.\n\n"
        "Uses unordered_map-based structure for scalability.\n"
        "Maps hash values to lists of point IDs efficiently with O(1) average lookup.\n"
        "Per-pool and per-table organization for hierarchical lookups.")
        .def(pybind11::init<const std::vector<std::vector<int32_t>>&,
                      const std::vector<std::vector<uint>>&, int>(),             pybind11::arg("hash_features"),
             pybind11::arg("pools"),
             pybind11::arg("debug") = 0,             "Initialize inverted index from hash features and pool assignments.\n\n"
             "Args:\n"
             "    hash_features: Matrix of shape (num_features, num_tables) with hash values\n"
             "    pools: Matrix of shape (num_pools, points_per_pool) with point assignments")
        
        .def("query", [](InvertedIndex& self, uint pool_id, pybind11::array_t<int32_t> query_hashes, uint min_tables) {
            std::vector<InvertedIndex::InPoolIndex> res = self.query(pool_id, numpy_to_vector(query_hashes), min_tables);
            std::vector<uint> res_uint(res.begin(), res.end());
            return vector_to_numpy(res_uint);
        }, pybind11::arg("pool_id"), pybind11::arg("query_hashes"), pybind11::arg("min_tables") = 1,
           "Query for points matching hash in pool across all tables.\n\n"
           "Args:\n"
           "    pool_id: Pool index\n"
           "    query_hashes: Vector of hash values (one per table)\n"
           "    min_tables: Minimum number of matching tables\n\n"
           "Returns:\n"
           "    Array of matching InPoolIndex")
        
        .def("query_multi_table", [](InvertedIndex& self, uint pool_id,
                                      pybind11::array_t<int32_t> query_hashes, uint threshold, int debug) {
            auto result = self.query_multi_table(pool_id, numpy_to_vector(query_hashes), threshold, debug);
            return vector_to_numpy(result);
        }, pybind11::arg("pool_id"), pybind11::arg("query_hashes"), pybind11::arg("threshold"), pybind11::arg("debug") = 0,
           "Query across multiple tables in a pool.\n\n"
           "Args:\n"
           "    pool_id: Pool index\n"
           "    query_hashes: Vector of hash values (one per table)\n"
           "    threshold: Minimum number of matching tables\n\n"
           "Returns:\n"
           "    Array of point IDs matching at least threshold tables")
        
        .def("memory_usage", &InvertedIndex::memory_usage,
             "Get total memory usage in bytes.\n\n"
             "Returns: Estimated memory usage including unordered_map overhead")
        
        .def("statistics", [](InvertedIndex& self) {
            auto stats = self.statistics();
            pybind11::dict py_stats;
            for (const auto& [key, val] : stats) {
                py_stats[pybind11::str(key)] = pybind11::int_(val);
            }
            return py_stats;
        }, "Get index statistics.\n\n"
           "Returns: Dictionary with keys:\n"
           "    - num_pools: Total number of pools\n"
           "    - num_tables: Total number of tables\n"
           "    - num_features: Total number of features\n"
           "    - total_tables: Non-empty tables count\n"
           "    - total_hashes: Total unique hash values\n"
           "    - total_points: Total point references\n"
           "    - avg_hashes_per_table: Average hash density");
    
    pybind11::class_<MLGTSaffronIndex>(m, "MLGTSaffronIndex",
        "MLGT with SAFFRON Index Wrapper - dual index structure\n\n"
        "Has an index for T-matrix pools --> tables --> hash value --> point IDs\n"
        "Has an index for U-matrix pools --> tables --> hash value --> point IDs\n"
        "Enables efficient intersection queries across both indices.")
        .def(pybind11::init<const std::vector<std::vector<int32_t>>&, Saffron&, int>(),
             pybind11::arg("hash_features"),
             pybind11::arg("saffron"),
             pybind11::arg("debug") = 0,
             "Initialize MLGT SAFFRON Index with T and U indices\n\n"
             "Args:\n"
             "    hash_features: Hash features matrix (num_features x num_tables)\n"
             "    saffron: Saffron decoder instance for pool structure\n"
             "    debug: Enable debug output")
        
        // ===== Public Data Members =====
        .def_readonly("num_buckets", &MLGTSaffronIndex::num_buckets,
                     "Number of buckets in the T-matrix (4*sparsity)")
        .def_readonly("num_tests", &MLGTSaffronIndex::num_tests,
                     "Number of tests (6*num_bits)")
        .def_readonly("num_tables", &MLGTSaffronIndex::num_tables,
                     "Number of hash tables")
        .def_readonly("num_features", &MLGTSaffronIndex::num_features,
                     "Total number of features")
        
        // ===== Query Methods =====
        .def("query", [](MLGTSaffronIndex& self, uint pool_id, pybind11::array_t<int32_t> query_hashes, 
                         uint min_matching_tables) {
            auto result = self.query(pool_id, numpy_to_vector(query_hashes), min_matching_tables);
            return vector_to_numpy(result);
        }, pybind11::arg("pool_id"), pybind11::arg("query_hashes"), 
           pybind11::arg("min_matching_tables") = 1,
           "Query for candidates matching both T and U indices.\n\n"
           "Args:\n"
           "    pool_id: Combined pool ID (bucket * num_tests + test)\n"
           "    query_hashes: Query hashes vector\n"
           "    min_matching_tables: Minimum number of matching tables\n\n"
           "Returns:\n"
           "    Array of matching point IDs (intersection of both indices)")
        
        .def("memory_usage", &MLGTSaffronIndex::memory_usage,
             "Get total memory usage of both T and U indices in bytes.\n\n"
             "Returns: Total memory usage (T_index + U_index)")
        
        .def("statistics", [](MLGTSaffronIndex& self) {
            auto stats = self.statistics();
            pybind11::dict py_stats;
            for (const auto& [key, val] : stats) {
                py_stats[pybind11::str(key)] = pybind11::int_(val);
            }
            return py_stats;
        }, "Get combined statistics from both T and U indices.\n\n"
           "Returns: Dictionary with prefixed keys:\n"
           "    - T_num_pools, T_num_tables, T_num_features, ...\n"
           "    - U_num_pools, U_num_tables, U_num_features, ...");

    
    pybind11::class_<MLGTsaffron>(m, "MLGTsaffron",
        "MLGT with SAFFRON group testing.\n\n"
        "Combines SimHash locality-sensitive hashing with SAFFRON group testing.\n"
        "Provides efficient nearest neighbor queries using hash-based pools.\n"
        "Features:\n"
        "  - L2 normalization of features\n"
        "  - Multi-table hash lookups\n"
        "  - Pool-based candidate aggregation\n"
        "  - Score-ranked retrieval")
        .def(pybind11::init([](unsigned int num_tables, unsigned int num_bits, 
                               unsigned int dimension, unsigned int num_neighbors,
                               pybind11::array_t<float> features_arr,
                               int debug) {
            // Convert numpy array to 2D vector
            auto buf = features_arr.request();
            auto* ptr = static_cast<float*>(buf.ptr);
            
            size_t num_features = buf.shape[0];
            if (buf.ndim != 2 || buf.shape[1] != dimension) {
                throw std::runtime_error("Features must be a 2D array with shape (num_features, dimension)");
            }
            
            std::vector<std::vector<float>> features_vec(num_features, std::vector<float>(dimension));
            for (size_t i = 0; i < num_features; ++i) {
                for (size_t j = 0; j < dimension; ++j) {
                    features_vec[i][j] = ptr[i * dimension + j];
                }
            }
            
            return new MLGTsaffron(num_tables, num_bits, dimension, num_neighbors, features_vec, debug);
        }),
             pybind11::arg("num_tables"),
             pybind11::arg("num_bits"),
             pybind11::arg("dimension"),
             pybind11::arg("num_neighbors"),
             pybind11::arg("features"),
             pybind11::arg("debug") = 0,
             "Initialize MLGT with SAFFRON.\n\n"
             "Args:\n"
             "    num_tables: Number of hash tables\n"
             "    num_bits: Bits per hash value\n"
             "    dimension: Feature vector dimension\n"
             "    num_neighbors: Expected number of neighbors (sparsity for SAFFRON)\n"
             "    features: Feature matrix as numpy array, shape (num_features, dimension) - will be L2 normalized")
        
        .def_readonly("num_tables", &MLGTsaffron::num_tables,
                     "Number of hash tables")
        .def_readonly("num_bits", &MLGTsaffron::num_bits,
                     "Bits per hash value")
        .def_readonly("dimension", &MLGTsaffron::dimension,
                     "Feature vector dimension")
        .def_readonly("num_neighbors", &MLGTsaffron::num_neighbors,
                     "Expected number of neighbors (sparsity for SAFFRON)")
        .def_readonly("num_features", &MLGTsaffron::num_features,
                     "Total number of features/data points")
        
        .def("get_saffron", [](MLGTsaffron& self) {
            return self.saffron;
        }, pybind11::return_value_policy::reference_internal,
           "Get the underlying Saffron decoder instance.\n\n"
           "Returns: Reference to Saffron object")
        
        .def("get_features", [](MLGTsaffron& self) {
            return matrix_to_numpy(
                std::vector<std::vector<float>>(
                    self.features.begin(), 
                    self.features.end()
                )
            );
        }, "Get normalized feature matrix.\n\n"
           "Returns: Feature matrix as numpy array, shape (num_features, dimension)")
        
        .def("get_pools", [](MLGTsaffron& self) {
            pybind11::list py_pools;
            for (const auto& pool : self.pool_list) {
                pybind11::list py_pool;
                for (uint item : pool) {
                    py_pool.append(item);
                }
                py_pools.append(py_pool);
            }
            return py_pools;
        }, "Get all SAFFRON pools (feature assignments).\n\n"
           "Returns: List of pools, each pool is a list of feature indices")
        
        .def("query", [](MLGTsaffron& self, pybind11::array_t<float> query_feature, uint k, int debug) {
            auto result = self.query(numpy_to_vector(query_feature), k, debug);
            return vector_to_numpy(result);
        }, pybind11::arg("query_feature"), pybind11::arg("k"), pybind11::arg("debug") = 0,
           "Query for k nearest neighbors using hash-based pooling.\n\n"
           "Generates hashes for query, queries all pools, aggregates by frequency,\n"
           "and returns top-k by match frequency.\n\n"
           "Args:\n"
           "    query_feature: Query feature vector (will be L2 normalized)\n"
           "    k: Number of neighbors to return\n"
           "    debug: Enable debug output\n\n"
           "Returns:\n"
           "    Array of up to k neighbor indices")
        
        .def("get_top_k_hash_neighbors", [](MLGTsaffron& self, pybind11::array_t<float> query_feature, 
                                            uint k, uint min_matching_tables, int debug) {
            auto result = self.get_top_k_hash_neighbors(numpy_to_vector(query_feature), k, min_matching_tables, debug);
            return vector_to_numpy(result);
        }, pybind11::arg("query_feature"), pybind11::arg("k"), pybind11::arg("min_matching_tables") = 1, pybind11::arg("debug") = 0,
           "Get top K hash neighbors for a query using direct hash comparison.\n\n"
           "Returns candidates that match query hashes directly.\n"
           "Candidates are ranked by number of matching hash tables (descending).\n"
           "This method bypasses the MLGT pool structure and scans all features.\n\n"
           "Args:\n"
           "    query_feature: Query feature vector (will be L2 normalized)\n"
           "    k: Number of neighbors to return\n"
           "    min_matching_tables: Minimum hash table matches required (default: 1)\n"
           "    debug: Enable debug output\n\n"
           "Returns:\n"
           "    Array of up to k neighbor indices sorted by match count (descending)")
        
        .def("get_hash_neighbors_with_scores", [](MLGTsaffron& self, pybind11::array_t<float> query_feature,
                                                   uint k, uint min_matching_tables, int debug) {
            auto result = self.get_hash_neighbors_with_scores(numpy_to_vector(query_feature), k, min_matching_tables, debug);
            
            pybind11::list py_result;
            for (const auto& [neighbor_id, score] : result) {
                py_result.append(pybind11::make_tuple(neighbor_id, score));
            }
            return py_result;
        }, pybind11::arg("query_feature"), pybind11::arg("k"), pybind11::arg("min_matching_tables") = 1, pybind11::arg("debug") = 0,
           "Get hash neighbors with score information using direct hash comparison.\n\n"
           "Returns candidates matched by hashes with their match scores.\n"
           "Score indicates the number of matching hash tables.\n"
           "This method bypasses the MLGT pool structure and scans all features.\n\n"
           "Args:\n"
           "    query_feature: Query feature vector (will be L2 normalized)\n"
           "    k: Number of neighbors to return\n"
           "    min_matching_tables: Minimum hash table matches required (default: 1)\n"
           "    debug: Enable debug output\n\n"
           "Returns:\n"
           "    List of tuples (neighbor_id, match_score) sorted by score descending")
        
        .def("get_pool_hash_neighbors", [](MLGTsaffron& self, pybind11::array_t<float> query_feature,
                                                         uint pool_id, uint k, uint min_matching_tables, int debug) {
                auto result = self.get_pool_hash_neighbors(numpy_to_vector(query_feature), pool_id, k, min_matching_tables, debug);
            return vector_to_numpy(result);
          }, pybind11::arg("query_feature"), pybind11::arg("pool_id"), pybind11::arg("k"), 
              pybind11::arg("min_matching_tables") = 1, pybind11::arg("debug") = 0,
           "Get hash neighbors from a specific SAFFRON pool.\n\n"
           "Queries only a single pool for hash neighbors matching the query.\n"
           "Useful for analyzing pool-specific matches and pool composition.\n\n"
           "Args:\n"
           "    query_feature: Query feature vector (will be L2 normalized)\n"
           "    pool_id: Pool index to query (0 to num_pools-1)\n"
           "    k: Number of neighbors to return\n"
           "    min_matching_tables: Minimum hash table matches required (default: 1)\n"
           "    debug: Enable debug output\n\n"
           "Returns:\n"
           "    Array of up to k neighbor indices in the specified pool")
        
        .def("get_index", [](MLGTsaffron& self) {
            return self.index_.get();
        }, pybind11::return_value_policy::reference_internal,
           "Get the underlying MLGTSaffronIndex instance.\n\n"
           "Returns: Reference to MLGTSaffronIndex for advanced operations")
;

    
    pybind11::class_<SimHash>(m, "SimHash",
        "SimHash: Random-projection based signed-hash (per-table integer hashes)")
        .def(pybind11::init<uint, uint, uint, uint>(),
             pybind11::arg("num_tables"), pybind11::arg("num_bits"), pybind11::arg("threshold"), pybind11::arg("dimension"))
        .def("hash_vector", [](SimHash& self, pybind11::array_t<float> vec) {
            auto buf = vec.request();
            if (buf.ndim != 1) throw std::runtime_error("hash_vector expects 1D float array");
            if ((uint)buf.shape[0] != self.dimension()) throw std::runtime_error("vector dimension mismatch");
            auto* ptr = static_cast<float*>(buf.ptr);
            std::vector<float> v(ptr, ptr + buf.shape[0]);
            auto out = self.hash_vector(v); // vector<int32_t>
            return vector_to_numpy<int32_t>(out);
        }, pybind11::arg("vector"))
        .def("hash_features", [](SimHash& self, pybind11::array_t<float> mat) {
            auto buf = mat.request();
            if (buf.ndim != 2) throw std::runtime_error("hash_features expects 2D float array");
            uint rows = static_cast<uint>(buf.shape[0]);
            uint cols = static_cast<uint>(buf.shape[1]);
            if (cols != self.dimension()) throw std::runtime_error("feature dimension mismatch");
            auto* ptr = static_cast<float*>(buf.ptr);
            // call hash_features by converting to vector<vector<float32_t>>
            std::vector<std::vector<float32_t>> features;
            features.reserve(rows);
            for (uint i = 0; i < rows; ++i) {
                std::vector<float32_t> row(cols);
                for (uint j = 0; j < cols; ++j) row[j] = ptr[i * cols + j];
                features.push_back(std::move(row));
            }
            auto hashes = self.hash_features(features); // vector<vector<int32_t>>
            // convert to numpy array (rows x num_tables)
            pybind11::array_t<int32_t> out(std::vector<size_t>{(size_t)rows, hashes[0].size()});
            auto outbuf = out.request();
            int32_t* outptr = static_cast<int32_t*>(outbuf.ptr);
            for (uint i = 0; i < rows; ++i) {
                for (size_t j = 0; j < hashes[i].size(); ++j) {
                    outptr[i * hashes[i].size() + j] = hashes[i][j];
                }
            }
            return out;
        }, pybind11::arg("features"));
    
    
    // ========================================================================
    // Module-level Functions
    // ========================================================================
    m.def("construct_T_matrix", [](uint num_features, uint degree, uint num_buckets, bool debug) {
        auto T = construct_T_matrix(num_features, degree, num_buckets, debug);
        auto py_T = pybind11::array_t<bool>(
            std::vector<size_t>{(size_t)num_buckets, (size_t)num_features}
        );
        auto buf = py_T.request();
        auto* ptr = static_cast<bool*>(buf.ptr);
        for (uint i = 0; i < num_buckets; ++i) {
            for (uint j = 0; j < num_features; ++j) {
                ptr[i * num_features + j] = T[i][j];
            }
        }
        return py_T;
    }, pybind11::arg("num_features"), pybind11::arg("degree"), pybind11::arg("num_buckets"), pybind11::arg("debug") = false,
       "Construct T matrix (bipartite graph incidence matrix).\n\n"
       "Creates a left-regular bipartite graph where each feature connects\n"
       "to exactly `degree` buckets chosen uniformly at random.\n\n"
       "Args:\n"
       "    num_features: Number of items/features (left nodes)\n"
       "    degree: Left-degree (connections per feature), typically ceil(log2(K))\n"
       "    num_buckets: Number of buckets (right nodes), typically 4*K\n\n"
       "Returns:\n"
       "    Boolean matrix of shape (num_buckets, num_features)");
    
    m.def("construct_U_matrix", [](uint num_features, uint num_bits, bool debug) {
        auto U = construct_U_matrix(num_features, num_bits, debug);
        uint rows = 6 * num_bits;
        auto py_U = pybind11::array_t<bool>(
            std::vector<size_t>{(size_t)rows, (size_t)num_features}
        );
        auto buf = py_U.request();
        auto* ptr = static_cast<bool*>(buf.ptr);
        for (uint i = 0; i < rows; ++i) {
            for (uint j = 0; j < num_features; ++j) {
                ptr[i * num_features + j] = U[i][j];
            }
        }
        return py_U;
    }, pybind11::arg("num_features"), pybind11::arg("num_bits"), pybind11::arg("debug") = false,
       "Construct U matrix (signature matrix).\n\n"
       "Creates 6 signature sections stacked vertically for singleton and\n"
       "doubleton detection/resolution. Total size: (6*num_bits) x num_features.\n\n"
       "Sections:\n"
       "  - U1: Binary representation of feature indices (first half of bits)\n"
       "  - U1_complement: Bitwise complement of U1\n"
       "  - U2: Binary representation using second half of bits\n"
       "  - U2_complement: Bitwise complement of U2\n"
       "  - U3: XOR of U1 and U2 (parity check)\n"
       "  - U4: XOR of U1 and U2_complement\n\n"
       "Args:\n"
       "    num_features: Number of items\n"
       "    num_bits: Bits per section, typically ceil(log2(n) / 2)\n\n"
       "Returns:\n"
       "    Boolean matrix of shape (6*num_bits, num_features)");

    // --- MLGT Class Bindings ---

    pybind11::class_<MLGT>(m, "MLGT",
        "Multi-Level Group Testing (MLGT) Index.\n\n"
        "Combines SimHash locality-sensitive hashing with group testing theory for efficient ANN search.\n"
        "Uses 'my_matching_algo' for candidate recovery.")
        .def(pybind11::init([](uint num_tables, uint num_bits, uint dimension, 
                               uint num_pools, uint pools_per_point, uint points_per_pool,
                               uint match_threshold, uint num_neighbors,
                               pybind11::array_t<float> features_arr, int debug) {
            
            auto buf = features_arr.request();
            if (buf.ndim != 2 || buf.shape[1] != dimension) {
                throw std::runtime_error("Features must be a 2D array with shape (num_features, dimension)");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            size_t num_features = buf.shape[0];
            
            std::vector<std::vector<float>> features_vec(num_features, std::vector<float>(dimension));
            for (size_t i = 0; i < num_features; ++i) {
                std::copy(ptr + i * dimension, ptr + (i + 1) * dimension, features_vec[i].begin());
            }
            
            return new MLGT(num_tables, num_bits, dimension, num_pools, 
                           pools_per_point, points_per_pool, match_threshold, 
                           num_neighbors, features_vec, debug);
        }),
        pybind11::arg("num_tables"), pybind11::arg("num_bits"), pybind11::arg("dimension"),
        pybind11::arg("num_pools"), pybind11::arg("pools_per_point"), pybind11::arg("points_per_pool"),
        pybind11::arg("match_threshold"), pybind11::arg("num_neighbors"),
        pybind11::arg("features"), pybind11::arg("debug") = 0,
        "Initialize MLGT index.\n\n"
        "Args:\n"
        "    num_tables: Number of hash tables\n"
        "    num_bits: Bits per hash\n"
        "    dimension: Input vector dimension\n"
        "    num_pools: Number of pools\n"
        "    pools_per_point: Pools per point\n"
        "    points_per_pool: Points per pool\n"
        "    match_threshold: Min table matches for positive pool\n"
        "    num_neighbors: Number of neighbors to return\n"
        "    features: Feature matrix (N x D)\n"
        "    debug: Debug level")
        
        .def("query", [](MLGT& self, pybind11::array_t<float> query_arr) {
            auto buf = query_arr.request();
            if (buf.ndim != 1 || buf.shape[0] != self.dimension) {
                throw std::runtime_error("Query must be a 1D array of size 'dimension'");
            }
            float* ptr = static_cast<float*>(buf.ptr);
            std::vector<float> query_vec(ptr, ptr + buf.shape[0]);
            
            Vec results = self.query(query_vec);
            return vector_to_numpy(results);
        }, pybind11::arg("query_vector"),
           "Query the index for nearest neighbors.\n\n"
           "Args:\n"
           "    query_vector: input query vector\n\n"
           "Returns:\n"
           "    List of nearest neighbor indices")
        
        // Read-only properties
        .def_readonly("num_features", &MLGT::num_features, "Number of features in index");

    // --- Standalone Functions ---

    m.def("obtain_pools", [](uint num_features, uint num_pools, uint pools_per_point, uint points_per_pool, int debug) {
        auto result = obtain_pools(num_features, num_pools, pools_per_point, points_per_pool, debug);
        Matrix& pools = result.first;
        BitMatrix& pooling_matrix = result.second;
        
        // Convert pools to list of lists
        pybind11::list py_pools;
        for (const auto& row : pools) {
            pybind11::list py_row;
            for (uint val : row) py_row.append(val);
            py_pools.append(py_row);
        }
        
        // Convert pooling_matrix to numpy bool array
        auto py_pooling_matrix = pybind11::array_t<bool>(
            std::vector<size_t>{(size_t)num_pools, (size_t)num_features}
        );
        auto buf = py_pooling_matrix.request();
        bool* ptr = static_cast<bool*>(buf.ptr);
        for (uint i = 0; i < num_pools; ++i) {
            for (uint j = 0; j < num_features; ++j) {
                ptr[i * num_features + j] = pooling_matrix[i][j];
            }
        }
        
        return pybind11::make_tuple(py_pools, py_pooling_matrix);
    }, pybind11::arg("num_features"), pybind11::arg("num_pools"), 
       pybind11::arg("pools_per_point"), pybind11::arg("points_per_pool"), 
       pybind11::arg("debug") = 0,
       "Generate random pool assignments.\n\n"
       "Returns:\n"
       "    Tuple(pools_list, pooling_matrix_bool)");

    m.def("my_matching_algo", [](pybind11::array_t<bool> pooling_matrix_arr, pybind11::array_t<uint> positive_pools_arr) {
        // Convert pooling_matrix
        auto pm_buf = pooling_matrix_arr.request();
        uint num_pools = pm_buf.shape[0];
        uint num_features = pm_buf.shape[1];
        bool* pm_ptr = static_cast<bool*>(pm_buf.ptr);
        
        BitMatrix pooling_matrix(num_pools, std::vector<bool>(num_features));
        for (uint i = 0; i < num_pools; ++i) {
            for (uint j = 0; j < num_features; ++j) {
                pooling_matrix[i][j] = pm_ptr[i * num_features + j];
            }
        }
        
        // Convert positive_pools
        std::vector<uint> positive_pools = numpy_to_vector(positive_pools_arr);
        
        Vec candidates = my_matching_algo(pooling_matrix, positive_pools);
        return vector_to_numpy(candidates);
    }, pybind11::arg("pooling_matrix"), pybind11::arg("positive_pools"),
       "Recover candidates using greedy group testing algorithm.\n\n"
       "Args:\n"
       "    pooling_matrix: Boolean matrix (pools x features)\n"
       "    positive_pools: Vector of positive pool indices (counts)\n\n"
       "Returns:\n"
       "    List of candidate feature indices");

}

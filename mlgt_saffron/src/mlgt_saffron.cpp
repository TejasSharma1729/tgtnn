#include "headers.hpp"
#include "HashFunction.hpp"
#include "BloomHashFunction.hpp"
#include "PoolingMatrix.hpp"
#include "Saffron.hpp"
#include "ParityBuckets.hpp"
#include "SaffronIndex.hpp"
#include "GlobalInvertedIndex.hpp"
#include "MLGTSaffron.hpp"
#include "BloomHashIndex.hpp"
#include "BloomGroupTestingSaffron.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mlgt_saffron, m) {
    m.doc() = "mlgt_saffron python module using SAFFRON for fast nearest neighbor search.";

    // Export Constants
    m.attr("NUM_POOLS_COEFF") = NUM_POOLS_COEFF;
    m.attr("POOLS_PER_ITEM") = POOLS_PER_ITEM;
    m.attr("SIGNATURE_COEFF") = SIGNATURE_COEFF;
    m.attr("NUM_HASH_BITS") = NUM_HASH_BITS;
    m.attr("BLOOM_HASH_BITS") = BLOOM_HASH_BITS;
    m.attr("BLOOM_NUM_HASHES") = BLOOM_NUM_HASHES;
    m.attr("BLOOM_THRESHOLD") = BLOOM_THRESHOLD;

    // HashFunction
    py::class_<HashFunction>(m, "HashFunction", "A Locality Sensitive Hashing (LSH) function using random projections.")
        .def(py::init<uint, uint, int>(), 
             py::arg("dimension"), 
             py::arg("num_hash_bits") = NUM_HASH_BITS, 
             py::arg("debug") = 0,
             "Initializes a HashFunction with a given dimension and number of hash bits.")
        .def("dimension", &HashFunction::dimension, "Returns the dimension of the hash function.")
        .def("num_hash_bits", &HashFunction::num_hash_bits, "Returns the number of bits in the hash.")
        .def("__call__", (uint (HashFunction::*)(py::array_t<float>) const) &HashFunction::operator(), py::arg("point"), 
             "Computes the hash value for a given point (callable interface, numpy).");

    // BloomHashFunction
    py::class_<BloomHashFunction>(m, "BloomHashFunction", "A Bloom Filter-inspired hash function generating multiple compound hash values.")
        .def(py::init<uint, uint, uint, uint, int>(),
             py::arg("dimension"),
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("num_bits") = BLOOM_HASH_BITS,
             py::arg("threshold") = BLOOM_THRESHOLD,
             py::arg("debug") = 0,
             "Initializes a BloomHashFunction.")
        .def("dimension", &BloomHashFunction::dimension, "Returns the input dimensionality.")
        .def("num_hashes", &BloomHashFunction::num_hashes, "Returns the number of compound hash functions.")
        .def("num_bits", &BloomHashFunction::num_bits, "Returns the number of bits per hash.")
        .def("__call__", (vector<uint> (BloomHashFunction::*)(const py::array_t<float>&) const) &BloomHashFunction::operator(), py::arg("point"),
             "Computes multiple hash values for a given point (callable interface, numpy).");

    // PoolingMatrix
    py::class_<PoolingMatrix>(m, "PoolingMatrix", "Represents the mapping between items and pools for the SAFFRON algorithm.")
        .def_readwrite("pools_to_items", &PoolingMatrix::pools_to_items, "List of vectors, where each vector contains the item indices in that pool.")
        .def_readwrite("items_to_pools", &PoolingMatrix::items_to_pools, "List of vectors, where each vector contains the pool indices the item belongs to.")
        .def_readwrite("num_features", &PoolingMatrix::num_features, "Total number of features/items (n).")
        .def_readwrite("num_pools", &PoolingMatrix::num_pools, "Total number of pools (m).");

    // Candidate
    py::class_<Candidate>(m, "Candidate", "Represents a candidate item found in a pool.")
        .def_readwrite("pool_idx", &Candidate::pool_idx, "Index of the pool.")
        .def_readwrite("item_idx", &Candidate::item_idx, "Index of the item in the pool.");

    // Saffron
    py::class_<Saffron>(m, "Saffron", "Core SAFFRON algorithm implementation for sparse recovery.")
        .def(py::init<uint, uint, int>(), 
             py::arg("num_features"), 
             py::arg("sparsity"), 
             py::arg("debug") = 0,
             "Initializes the Saffron algorithm setup.")
        .def("num_features", &Saffron::num_features, "Returns the number of features (n).")
        .def("sparsity", &Saffron::sparsity, "Returns the expected sparsity level (k).")
        .def("num_pools", &Saffron::num_pools, "Returns the number of pools.")
        .def("signature_length", &Saffron::signature_length, "Returns the length of the signatures.")
        .def("peelingAlgorithm", &Saffron::peelingAlgorithm, 
             py::arg("residuals"), 
             py::arg("debug") = 0,
             "Executes the peeling algorithm to recover identified items from residuals.");
     
    // ParityBuckets
     py::class_<ParityBuckets>(m, "ParityBuckets", "A memory-efficient parity bucket for Saffron.")
        .def_readwrite("buckets", &ParityBuckets::buckets, "Bitset representing bucket parities.")
        .def_readwrite("num_buckets", &ParityBuckets::num_buckets, "Number of buckets.")
        .def(py::init<>(), "Construct a new Parity Buckets object.")
        .def("insert", &ParityBuckets::insert, py::arg("hash_val"), "Inserts a hash value into the parity buckets.")
        .def("get_parity", &ParityBuckets::get_parity, py::arg("hash_val"), "Checks if a hash value is likely present based on parity.");
     
    // GlobalInvertedIndex
    py::class_<GlobalInvertedIndex>(m, "GlobalInvertedIndex", "A memory-efficient global inverted index for all items.")
        .def(py::init<uint, uint>(),
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("threshold") = BLOOM_THRESHOLD,
             "Initializes the GlobalInvertedIndex.")
        .def("num_hashes", &GlobalInvertedIndex::num_hashes, "Returns the number of hash functions used.")
        .def("build", &GlobalInvertedIndex::build, py::arg("all_hashes"), py::arg("item_indices"),
             "Builds the index from a list of pre-computed hash values.")
        .def("get_matches", &GlobalInvertedIndex::get_matches, py::arg("query_hashes"), 
             "Returns the indices of items that match the query hashes.");

    // SaffronIndex
    py::class_<SaffronIndex, Saffron>(m, "SaffronIndex", "An indexing structure combining Saffron with Parity Buckets.")
        .def(py::init<py::array_t<float>, uint, uint, int, bool>(), 
             py::arg("data_points"), 
             py::arg("num_neighbors") = 100, 
             py::arg("num_hash_bits") = NUM_HASH_BITS, 
             py::arg("debug") = 0,
             py::arg("normalize") = true)
        .def("search", &SaffronIndex::search, py::arg("query"), "Searches for the top K nearest neighbors.")
        .def("__call__", &SaffronIndex::operator(), py::arg("query"), "Searches for the top K nearest neighbors.");

    // BloomHashIndex
    py::class_<BloomHashIndex, BloomHashFunction>(m, "BloomHashIndex", "A memory-efficient inverted index using multiple hash functions.")
        .def(py::init<py::array_t<float>, uint, uint, uint, int>(),
             py::arg("data_points"),
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("num_bits") = BLOOM_HASH_BITS,
             py::arg("threshold") = BLOOM_THRESHOLD,
             py::arg("debug") = 0)
        .def("matches", &BloomHashIndex::matches, py::arg("query_hashes"), "Checks if any item matches the query hashes.")
        .def("get_matches", &BloomHashIndex::get_matches, py::arg("query_hashes"), "Returns indices of matching items.")
        .def("__call__", (bool (BloomHashIndex::*)(const vector<float>&) const) &BloomHashIndex::operator(), py::arg("query"), "Checks if any item matches the query.");

    // MLGTSaffron
    py::class_<MLGTSaffron, Saffron>(m, "MLGTSaffron", "MLGT Saffron nearest neighbor search implementation using BloomHashIndex.")
        .def(py::init<py::array_t<float>, uint, uint, uint, uint, int, bool>(),
             py::arg("data_points"),
             py::arg("num_neighbors") = 100,
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("hash_bits") = BLOOM_HASH_BITS,
             py::arg("threshold") = BLOOM_THRESHOLD,
             py::arg("debug") = 0,
             py::arg("normalize") = true)
        .def("search", &MLGTSaffron::search, py::arg("query"), "Searches for the top K nearest neighbors.")
        .def("__call__", &MLGTSaffron::operator(), py::arg("query"), "Searches for the top K nearest neighbors.");

    // BloomGroupTestingSaffron
    py::class_<BloomGroupTestingSaffron, Saffron>(m, "BloomGroupTestingSaffron", "Saffron variant using one BloomHashIndex per (pool, test) pair.")
        .def(py::init<py::array_t<float>, uint, uint, uint, uint, int, bool>(),
             py::arg("data_points"),
             py::arg("num_neighbors") = 100,
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("hash_bits") = BLOOM_HASH_BITS,
             py::arg("threshold") = BLOOM_THRESHOLD,
             py::arg("debug") = 0,
             py::arg("normalize") = true)
        .def("search", &BloomGroupTestingSaffron::search, py::arg("query"), "Searches for the top K nearest neighbors.");

    // Free functions
    m.def("computePools", &computePools, 
          py::arg("num_features"), 
          py::arg("sparsity"), 
          py::arg("debug") = 0,
          "Computes the pooling matrix for SAFFRON.");

    m.def("getSignature", &getSignature, 
          py::arg("j"), 
          py::arg("signature_length"),
          "Generates a robust 6L SAFFRON signature for singleton and doubleton recovery.");

    m.def("decodeSignature", &decodeSignature, 
          py::arg("measurement"), 
          py::arg("signature_length"),
          "Decodes a signature from a measurement vector, supporting doubleton recovery.");
     
    m.def("normalizeDataPoint", &normalizeDataPoint, 
          py::arg("data_point"), 
          py::arg("normalize") = true,
          "Normalizes a single data point if requested.");

    m.def("normalizeDataset", &normalizeDataset, 
          py::arg("data_points"), 
          py::arg("normalize") = true,
          "Normalizes a dataset if requested.");
}


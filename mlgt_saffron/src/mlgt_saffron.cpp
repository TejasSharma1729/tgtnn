#include "headers.hpp"
#include "saffron.hpp"
#include "saffron_index.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mlgt_saffron, m) {
    m.doc() = "mlgt_saffron python module using SAFFRON for fast nearest neighbor search.";

    // Export Constants
    m.attr("NUM_POOLS_COEFF") = NUM_POOLS_COEFF;
    m.attr("POOLS_PER_ITEM") = POOLS_PER_ITEM;
    m.attr("SIGNATURE_COEFF") = SIGNATURE_COEFF;
    m.attr("NUM_HASH_BITS") = NUM_HASH_BITS;

    // HashFunction
    py::class_<HashFunction>(m, "HashFunction", "A Locality Sensitive Hashing (LSH) function using random projections.")
        .def(py::init<uint, uint, int>(), 
             py::arg("dimension"), 
             py::arg("num_hash_bits") = NUM_HASH_BITS, 
             py::arg("debug") = 0,
             "Initializes a HashFunction with a given dimension and number of hash bits.\n\n"
             ":param dimension: The dimensionality of the input points.\n"
             ":param num_hash_bits: The number of bits in the generated hash (default: NUM_HASH_BITS).\n"
             ":param debug: Debug level (0 for none).")
        .def("dimension", &HashFunction::dimension, "Returns the dimension of the hash function.\n\n:return: Dimensionality as uint.")
        .def("num_hash_bits", &HashFunction::num_hash_bits, "Returns the number of bits in the hash.\n\n:return: Number of bits as uint.")
        .def("computeHash", (uint (HashFunction::*)(py::array_t<float>) const) &HashFunction::computeHash, py::arg("point"), 
             "Computes the hash value for a given point.\n\n"
             ":param point: A 1D numpy array of floats representing the point.\n"
             ":return: The computed hash value as an integer.")
        .def("__call__", (uint (HashFunction::*)(py::array_t<float>) const) &HashFunction::operator(), py::arg("point"), 
             "Computes the hash value for a given point (callable interface).\n\n"
             ":param point: A 1D numpy array of floats representing the point.\n"
             ":return: The computed hash value as an integer.");

    // PoolingMatrix
    py::class_<PoolingMatrix>(m, "PoolingMatrix", "Represents the mapping between items and pools for the SAFFRON algorithm.")
        .def_readwrite("pools_to_items", &PoolingMatrix::pools_to_items, "List of vectors, where each vector contains the item indices in that pool.")
        .def_readwrite("items_to_pools", &PoolingMatrix::items_to_pools, "List of vectors, where each vector contains the pool indices the item belongs to.")
        .def_readwrite("num_features", &PoolingMatrix::num_features, "Total number of features/items.")
        .def_readwrite("num_pools", &PoolingMatrix::num_pools, "Total number of pools.");

    // Candidate (formerly SingletonPool)
    py::class_<Candidate>(m, "Candidate", "Represents a candidate item found in a pool.")
        .def_readwrite("pool_idx", &Candidate::pool_idx, "Index of the pool.")
        .def_readwrite("item_idx", &Candidate::item_idx, "Index of the item in the pool.");

    // Saffron
    py::class_<Saffron>(m, "Saffron", "Core SAFFRON algorithm implementation for sparse recovery.")
        .def(py::init<uint, uint, int>(), 
             py::arg("num_features"), 
             py::arg("sparsity"), 
             py::arg("debug") = 0,
             "Initializes the Saffron algorithm.\n\n"
             ":param num_features: Total number of features (n).\n"
             ":param sparsity: Expected sparsity level (k).\n"
             ":param debug: Debug level.")
        .def("num_features", &Saffron::num_features, "Returns the number of features.\n\n:return: uint.")
        .def("sparsity", &Saffron::sparsity, "Returns the sparsity level.\n\n:return: uint.")
        .def("num_pools", &Saffron::num_pools, "Returns the number of pools.\n\n:return: uint.")
        .def("signature_length", &Saffron::signature_length, "Returns the length of the signatures.\n\n:return: uint.")
        .def("peelingAlgorithm", &Saffron::peelingAlgorithm, 
             py::arg("residuals"), 
             py::arg("debug") = 0,
             "Executes the peeling algorithm to recover identified items from residuals.\n\n"
             ":param residuals: A 2D vector of booleans (num_pools x signature_bits).\n"
             ":param debug: Debug level.\n"
             ":return: A set of identified item indices.");

    // SaffronIndex
    py::class_<SaffronIndex, Saffron>(m, "SaffronIndex", "An indexing structure combining Saffron with Parity Buckets for fast nearest neighbor search.")
        .def(py::init<py::array_t<float>, uint, uint, int, bool>(), 
             py::arg("data_points"), 
             py::arg("num_neighbors") = 100, 
             py::arg("num_hash_bits") = NUM_HASH_BITS, 
             py::arg("debug") = 0,
             py::arg("normalize") = true,
             "Initializes SaffronIndex with data points and indexing parameters.\n\n"
             ":param data_points: A 2D numpy array of floats (num_points x dimension).\n"
             ":param num_neighbors: The target number of neighbors to recover (k).\n"
             ":param num_hash_bits: Number of bits for the hash functions.\n"
             ":param debug: Debug level.\n"
             ":param normalize: Whether to normalize data points to unit sphere (default: true).")
        .def("size", &SaffronIndex::size, "Returns the number of data points.\n\n:return: uint.")
        .def("dimension", &SaffronIndex::dimension, "Returns the dimensionality of the data points.\n\n:return: uint.")
        .def("k_val", &SaffronIndex::k_val, "Returns the k value (sparsity/target neighbors).\n\n:return: uint.")
        .def("getResiduals", &SaffronIndex::getResiduals, py::arg("query"), 
             "Computes the residuals for a given query vector.\n\n"
             ":param query: The query point as a 1D numpy array of floats.\n"
             ":return: A 2D vector of booleans representing residuals.")
        .def("getTopK", &SaffronIndex::getTopK, py::arg("query"), py::arg("identified"), 
             "Scores identified items and returns the top K based on inner product.\n\n"
             ":param query: The query vector (numpy array).\n"
             ":param identified: A set of identified item indices.\n"
             ":return: A list of the top K item indices.")
        .def("search", &SaffronIndex::search, py::arg("query"), 
             "Performs a full search for the given query vector using the peeling algorithm.\n\n"
             ":param query: The query point (numpy array).\n"
             ":return: A list of the top K nearest neighbor indices.")
        .def("__call__", &SaffronIndex::operator(), py::arg("query"), 
             "Performs a search using the callable interface.\n\n"
             ":param query: The query point (numpy array).\n"
             ":return: A list of the top K nearest neighbor indices.");

    // Free functions
    m.def("computePools", &computePools, 
          py::arg("num_features"), 
          py::arg("sparsity"), 
          py::arg("debug") = 0,
          "Computes the pooling matrix for SAFFRON.\n\n"
          ":param num_features: Total number of features.\n"
          ":param sparsity: Expected sparsity level.\n"
          ":param debug: Debug level.\n"
          ":return: A PoolingMatrix object.");

    m.def("getSignature", &getSignature, 
          py::arg("j"), 
          py::arg("signature_length"),
          "Generates a signature for a given item index.\n\n"
          ":param j: The item index.\n"
          ":param signature_length: Length of the desired signature.\n"
          ":return: A boolean vector signature.");

    m.def("decodeSignature", &decodeSignature, 
          py::arg("measurement"), 
          py::arg("signature_length"),
          "Decodes a signature from a measurement vector, supporting doubleton recovery.\n\n"
          ":param measurement: A boolean vector of measurement bits.\n"
          ":param signature_length: Length of the signature.\n"
          ":return: A list of discovered item indices (empty, 1 item, or 2 items).");
}


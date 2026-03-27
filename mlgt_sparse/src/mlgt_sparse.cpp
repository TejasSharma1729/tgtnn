#include "headers.hpp"
#include "BaseHasher.hpp"
#include "MinHasher.hpp"
#include "WeightedMinHasher.hpp"
#include "SparseSRPHasher.hpp"
#include "DenseSRPHasher.hpp"
#include "BloomHashFunction.hpp"
#include "PoolingMatrix.hpp"
#include "Saffron.hpp"
#include "GlobalInvertedIndex.hpp"
#include "MLGTSaffron.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mlgt_sparse, m) {
    m.doc() = "Multi-Label Group Testing (MLGT) module for sub-linear nearest neighbor search.";

    // Export Constants
    m.attr("NUM_POOLS_COEFF") = NUM_POOLS_COEFF;
    m.attr("POOLS_PER_ITEM") = POOLS_PER_ITEM;
    m.attr("SIGNATURE_COEFF") = SIGNATURE_COEFF;
    m.attr("BLOOM_HASH_BITS") = BLOOM_HASH_BITS;
    m.attr("BLOOM_NUM_HASHES") = BLOOM_NUM_HASHES;
    m.attr("BLOOM_THRESHOLD") = BLOOM_THRESHOLD;

    // MinHasher
    py::class_<MinHasher>(m, "MinHasher", 
        "MinHash implementation for Jaccard similarity estimation on sparse sets.")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("num_hashes") = 1,
             py::arg("hashes_per_table") = 1,
             py::arg("hash_range_pow") = 16,
             py::arg("seed") = 42)
        .def("__call__", [](const MinHasher& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"))
        .def("__call__", [](const MinHasher& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"))
        .def_readwrite("num_hashes", &MinHasher::num_hashes)
        .def_readwrite("seed", &MinHasher::seed);

    // WeightedMinHasher
    py::class_<WeightedMinHasher>(m, "WeightedMinHasher",
        "Weighted MinHash implementation for Weighted Jaccard similarity estimation.")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("num_hashes") = 1,
             py::arg("hashes_per_table") = 1,
             py::arg("hash_range_pow") = 16,
             py::arg("seed") = 42)
        .def("__call__", [](const WeightedMinHasher& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"))
        .def("__call__", [](const WeightedMinHasher& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"))
        .def_readwrite("num_hashes", &WeightedMinHasher::num_hashes)
        .def_readwrite("seed", &WeightedMinHasher::seed);

    // SparseSRPHasher
    py::class_<SparseSRPHasher>(m, "SparseSRPHasher",
        "Sparse Signed Random Projection (SRP) for cosine similarity estimation.")
        .def(py::init<uint32_t, uint32_t, uint32_t>(),
             py::arg("num_bits") = 16,
             py::arg("seed") = 42,
             py::arg("num_hashes") = 1)
        .def("__call__", [](const SparseSRPHasher& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"))
        .def("__call__", [](const SparseSRPHasher& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"))
        .def_readwrite("num_hashes", &SparseSRPHasher::num_hashes)
        .def_readwrite("seed", &SparseSRPHasher::seed);

    // DenseSRPHasher
    py::class_<DenseSRPHasher>(m, "DenseSRPHasher",
        "Dense Signed Random Projection (SRP) optimized for dense input vectors.")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, bool>(),
             py::arg("num_bits") = 16,
             py::arg("dimension") = 0,
             py::arg("seed") = 42,
             py::arg("num_hashes") = 1,
             py::arg("store") = false)
        .def("__call__", [](const DenseSRPHasher& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"))
        .def_readwrite("num_hashes", &DenseSRPHasher::num_hashes)
        .def_readwrite("seed", &DenseSRPHasher::seed);

    // BloomHashFunction
    py::class_<BloomHashFunction>(m, "BloomHashFunction",
        "Compound Sparse SRP hasher used by the default MLGT implementation.")
        .def(py::init<uint, uint, uint, uint, int, uint>(),
             py::arg("dimension"),
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("num_bits") = BLOOM_HASH_BITS,
             py::arg("threshold") = BLOOM_THRESHOLD,
             py::arg("debug") = 0,
             py::arg("seed") = 0)
        .def("__call__", [](const BloomHashFunction& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"))
        .def("__call__", [](const BloomHashFunction& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"))
        .def_readwrite("num_hashes", &BloomHashFunction::num_hashes)
        .def_readwrite("seed", &BloomHashFunction::seed);


    // Helper to register MLGTSaffron variants
    auto register_mlgt = [&](auto& module, const char* name, auto type_ptr) {
        using T = typename std::remove_pointer<decltype(type_ptr)>::type;
        py::class_<T>(module, name, "MLGT Nearest Neighbor search engine using the SAFFRON scheme.")
            .def(py::init<py::array_t<float>, py::array_t<uint32_t>, py::array_t<uint64_t>, uint32_t, typename T::HasherAlias, uint, uint, int, bool>(),
                 py::arg("data"),
                 py::arg("indices"),
                 py::arg("indptr"),
                 py::arg("num_cols"),
                 py::arg("hasher"),
                 py::arg("num_neighbors") = 100,
                 py::arg("threshold") = BLOOM_THRESHOLD,
                 py::arg("debug") = 0,
                 py::arg("normalize") = true,
                 "Construct an MLGT search index.\n\n"
                 "Args:\n"
                 "    data: CSR values array (float32).\n"
                 "    indices: CSR column indices array (uint32).\n"
                 "    indptr: CSR row pointers array (uint64).\n"
                 "    num_cols: Total number of columns (features).\n"
                 "    hasher: An instance of a compatible Hasher.\n"
                 "    num_neighbors: Top K neighbors to recover.\n"
                 "    threshold: Matching threshold for recovery.\n"
                 "    debug: Debug level.\n"
                 "    normalize: Whether to L2-normalize input vectors.")
            .def("search", &T::search, py::arg("query"), 
                 "Perform a nearest neighbor search for a dense query.\n\n"
                 "Args:\n"
                 "    query: 1D numpy array representing the dense query vector.\n"
                 "Returns:\n"
                 "    List of indices of the recovered nearest neighbors.")
            .def("__call__", &T::operator(), py::arg("query"), "Alias for search().");
    };

    register_mlgt(m, "MLGTSaffronBloom", (MLGTSaffronBloom*)nullptr);
    register_mlgt(m, "MLGTSaffronMinHash", (MLGTSaffronMinHash*)nullptr);
    register_mlgt(m, "MLGTSaffronWeightedMinHash", (MLGTSaffronWeightedMinHash*)nullptr);
    register_mlgt(m, "MLGTSaffronSparseSRP", (MLGTSaffronSparseSRP*)nullptr);
    register_mlgt(m, "MLGTSaffronDenseSRP", (MLGTSaffronDenseSRP*)nullptr);

    // Alias for backward compatibility
    m.attr("MLGTSaffron") = m.attr("MLGTSaffronBloom");
}

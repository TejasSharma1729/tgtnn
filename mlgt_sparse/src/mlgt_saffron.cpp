#include "headers.hpp"
#include "BaseHasher.hpp"
#include "MinHasher.hpp"
#include "SparseSRPHasher.hpp"
#include "DenseSRPHasher.hpp"
#include "BloomHashFunction.hpp"
#include "PoolingMatrix.hpp"
#include "Saffron.hpp"
#include "GlobalInvertedIndex.hpp"
#include "MLGTSaffron.hpp"
#include "BloomGroupTestingSaffron.hpp"
#include "SaffronIndex.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mlgt_saffron, m) {
    m.doc() = "mlgt_saffron python module using SAFFRON for sparse nearest neighbor search.";

    // Export Constants
    m.attr("NUM_POOLS_COEFF") = NUM_POOLS_COEFF;
    m.attr("POOLS_PER_ITEM") = POOLS_PER_ITEM;
    m.attr("SIGNATURE_COEFF") = SIGNATURE_COEFF;
    m.attr("BLOOM_HASH_BITS") = BLOOM_HASH_BITS;
    m.attr("BLOOM_NUM_HASHES") = BLOOM_NUM_HASHES;
    m.attr("BLOOM_THRESHOLD") = BLOOM_THRESHOLD;

    // MinHasher
    py::class_<MinHasher>(m, "MinHasher")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("num_hashes") = 1,
             py::arg("hashes_per_table") = 1,
             py::arg("hash_range_pow") = 16,
             py::arg("seed") = 42);

    // SparseSRPHasher
    py::class_<SparseSRPHasher>(m, "SparseSRPHasher")
        .def(py::init<uint32_t, uint32_t, uint32_t>(),
             py::arg("num_bits") = 16,
             py::arg("seed") = 42,
             py::arg("num_hashes") = 1);

    // DenseSRPHasher
    py::class_<DenseSRPHasher>(m, "DenseSRPHasher")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, bool>(),
             py::arg("num_bits") = 16,
             py::arg("dimension") = 0,
             py::arg("seed") = 42,
             py::arg("num_hashes") = 1,
             py::arg("store") = false);

    // BloomHashFunction
    py::class_<BloomHashFunction>(m, "BloomHashFunction")
        .def(py::init<uint, uint, uint, uint, int>(),
             py::arg("dimension"),
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("num_bits") = BLOOM_HASH_BITS,
             py::arg("threshold") = BLOOM_THRESHOLD,
             py::arg("debug") = 0);

    // Helper to register MLGTSaffron variants
    auto register_mlgt = [&](auto& m, const char* name, auto type_ptr) {
        using T = typename std::remove_pointer<decltype(type_ptr)>::type;
        py::class_<T>(m, name)
            .def(py::init<py::array_t<float>, py::array_t<uint32_t>, py::array_t<uint64_t>, uint32_t, typename T::HasherAlias, uint, uint, int, bool>(),
                 py::arg("data"),
                 py::arg("indices"),
                 py::arg("indptr"),
                 py::arg("num_cols"),
                 py::arg("hasher"),
                 py::arg("num_neighbors") = 100,
                 py::arg("threshold") = BLOOM_THRESHOLD,
                 py::arg("debug") = 0,
                 py::arg("normalize") = true)
            .def("search", &T::search, py::arg("query"))
            .def("__call__", &T::operator(), py::arg("query"));
    };

    register_mlgt(m, "MLGTSaffronBloom", (MLGTSaffronBloom*)nullptr);
    register_mlgt(m, "MLGTSaffronMinHash", (MLGTSaffronMinHash*)nullptr);
    register_mlgt(m, "MLGTSaffronSparseSRP", (MLGTSaffronSparseSRP*)nullptr);
    register_mlgt(m, "MLGTSaffronDenseSRP", (MLGTSaffronDenseSRP*)nullptr);

    // Alias for backward compatibility
    m.attr("MLGTSaffron") = m.attr("MLGTSaffronBloom");

    // BloomGroupTestingSaffron
    py::class_<BloomGroupTestingSaffron>(m, "BloomGroupTestingSaffron")
        .def(py::init<py::array_t<float>, uint, uint, uint, uint, int, bool>(),
             py::arg("data"),
             py::arg("num_neighbors") = 100,
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("hash_bits") = BLOOM_HASH_BITS,
             py::arg("threshold") = BLOOM_THRESHOLD,
             py::arg("debug") = 0,
             py::arg("normalize") = true)
        .def("search", &BloomGroupTestingSaffron::search, py::arg("query"))
        .def("__call__", &BloomGroupTestingSaffron::operator(), py::arg("query"));

    // SaffronIndex
    py::class_<SaffronIndex>(m, "SaffronIndex")
        .def(py::init<py::array_t<float>, uint, uint, int, bool>(),
             py::arg("data"),
             py::arg("num_neighbors") = 100,
             py::arg("num_hash_bits") = NUM_HASH_BITS,
             py::arg("debug") = 0,
             py::arg("normalize") = true)
        .def("search", &SaffronIndex::search, py::arg("query"))
        .def("__call__", &SaffronIndex::operator(), py::arg("query"));
}

#include "headers.hpp"
#include "BaseHasher.hpp"
#include "MinHasher.hpp"
#include "WeightedMinHasher.hpp"
#include "SparseSRPHasher.hpp"
#include "DenseSRPHasher.hpp"
#include "BloomHashFunction.hpp"
#include "PoolingMatrix.hpp"
#include "Saffron.hpp"
#include "PoolInvertedIndex.hpp"
#include "MLGTSaffron.hpp"
#include "MLGTGlobal.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mlgt_sparse, m) {
    m.doc() =
        "Multi-Label Group Testing (MLGT) module for sub-linear approximate nearest\n"
        "neighbour search.\n\n"
        "The module implements the SAFFRON sparse-recovery scheme on top of\n"
        "locality-sensitive hashing.  A dataset is indexed once; queries are answered\n"
        "in sub-linear time by identifying sparse components through pool residuals and\n"
        "an iterative peeling algorithm.\n\n"
        "Exported constants\n"
        "------------------\n"
        "NUM_POOLS_COEFF   : Default multiplier for computing the number of pools (m = k * NUM_POOLS_COEFF).\n"
        "POOLS_PER_ITEM    : Default number of pools each item is assigned to.\n"
        "SIGNATURE_COEFF   : Coefficient used in SAFFRON signature-length calculation.\n"
        "BLOOM_HASH_BITS   : Default number of SRP projection bits per compound hash.\n"
        "BLOOM_NUM_HASHES  : Default number of compound hash functions in BloomHashFunction.\n"
        "BLOOM_THRESHOLD   : Default minimum hash-match threshold for candidate identification.";

    // Export Constants
    m.attr("NUM_POOLS_COEFF") = NUM_POOLS_COEFF;
    m.attr("POOLS_PER_ITEM") = POOLS_PER_ITEM;
    m.attr("SIGNATURE_COEFF") = SIGNATURE_COEFF;
    m.attr("BLOOM_HASH_BITS") = BLOOM_HASH_BITS;
    m.attr("BLOOM_NUM_HASHES") = BLOOM_NUM_HASHES;
    m.attr("BLOOM_THRESHOLD") = BLOOM_THRESHOLD;

    // Abstract base class for all hashers
    py::class_<BaseHasher>(m, "Hasher",
        "Abstract base class for all MLGT hashers.\n\n"
        "All concrete hasher types (MinHasher, WeightedMinHasher, SparseSRPHasher,\n"
        "DenseSRPHasher, BloomHashFunction) inherit from this class.  Use it for\n"
        "type annotations and isinstance() checks:\n\n"
        "    h: Hasher = MinHasher(num_hashes=50)\n"
        "    assert isinstance(h, Hasher)\n\n"
        "Members\n"
        "-------\n"
        "num_hashes : Number of independent hash values returned per call.\n"
        "seed       : Random seed used for hash function generation.")
        .def_readwrite("num_hashes", &BaseHasher::num_hashes,
             "Number of independent hash values returned per call.")
        .def_readwrite("seed", &BaseHasher::seed,
             "Random seed used for hash function generation.")
        .def("__call__", [](const BaseHasher& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"),
             "Hash a sparse CSR row.\n\n"
             "Args:\n"
             "    data   : 1-D float32 numpy array of non-zero values.\n"
             "    indices: 1-D uint32 numpy array of column indices of the non-zero elements.\n"
             "    nnz    : Number of non-zero elements.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def("__call__", [](const BaseHasher& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"),
             "Hash a dense query vector.\n\n"
             "Args:\n"
             "    query: Dense Eigen/numpy vector.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.");

    // MinHasher
    py::class_<MinHasher, BaseHasher>(m, "MinHasher",
        "MinHash locality-sensitive hasher for unweighted Jaccard similarity.\n\n"
        "Projects a sparse binary set (the support of a sparse vector) into\n"
        "`num_hashes` integer hash values by computing the minimum of a random\n"
        "permutation over the set's indices.  Multiple internal hashes can be\n"
        "concatenated per table to reduce the false-positive rate.\n\n"
        "Members\n"
        "-------\n"
        "num_hashes      : Number of independent hash values returned per call.\n"
        "seed            : Random seed used for hash function generation.")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("num_hashes") = 1,
             py::arg("hashes_per_table") = 1,
             py::arg("hash_range_pow") = 16,
             py::arg("seed") = 42,
             "Construct a MinHasher.\n\n"
             "Args:\n"
             "    num_hashes      : Number of independent hash tables (final hash values returned).\n"
             "    hashes_per_table: Number of min-hashes concatenated per table (higher → fewer collisions).\n"
             "    hash_range_pow  : Output range is [0, 2**hash_range_pow).\n"
             "    seed            : Random seed.")
        .def("__call__", [](const MinHasher& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"),
             "Hash a sparse CSR row.\n\n"
             "Args:\n"
             "    data   : 1-D float32 numpy array of non-zero values (ignored; only indices matter).\n"
             "    indices: 1-D uint32 numpy array of column indices of the non-zero elements.\n"
             "    nnz    : Number of non-zero elements.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def("__call__", [](const MinHasher& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"),
             "Hash a dense query vector.\n\n"
             "Args:\n"
             "    query: Dense Eigen/numpy vector; entries with |value| > 1e-9 are treated as present.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def_readwrite("num_hashes", &MinHasher::num_hashes,
             "Number of independent hash values returned per call.")
        .def_readwrite("seed", &MinHasher::seed,
             "Random seed used for hash function generation.");

    // WeightedMinHasher
    py::class_<WeightedMinHasher, BaseHasher>(m, "WeightedMinHasher",
        "Weighted MinHash locality-sensitive hasher for Weighted Jaccard similarity.\n\n"
        "Each non-zero feature is treated as a weighted element.  The minimum of\n"
        "-log(u)/weight (exponential racing) determines the representative for each\n"
        "hash function, making collision probability proportional to the Weighted\n"
        "Jaccard coefficient.\n\n"
        "Members\n"
        "-------\n"
        "num_hashes      : Number of independent hash values returned per call.\n"
        "seed            : Random seed used for hash function generation.")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("num_hashes") = 1,
             py::arg("hashes_per_table") = 1,
             py::arg("hash_range_pow") = 16,
             py::arg("seed") = 42,
             "Construct a WeightedMinHasher.\n\n"
             "Args:\n"
             "    num_hashes      : Number of independent hash tables (final hash values returned).\n"
             "    hashes_per_table: Number of internal hash values combined per table.\n"
             "    hash_range_pow  : Output range is [0, 2**hash_range_pow).\n"
             "    seed            : Random seed.")
        .def("__call__", [](const WeightedMinHasher& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"),
             "Hash a sparse CSR row using feature weights.\n\n"
             "Args:\n"
             "    data   : 1-D float32 numpy array of non-zero weights.\n"
             "    indices: 1-D uint32 numpy array of column indices of the non-zero elements.\n"
             "    nnz    : Number of non-zero elements.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def("__call__", [](const WeightedMinHasher& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"),
             "Hash a dense query vector.\n\n"
             "Args:\n"
             "    query: Dense Eigen/numpy vector; only entries > 1e-9 contribute.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def_readwrite("num_hashes", &WeightedMinHasher::num_hashes,
             "Number of independent hash values returned per call.")
        .def_readwrite("seed", &WeightedMinHasher::seed,
             "Random seed used for hash function generation.");

    // SparseSRPHasher
    py::class_<SparseSRPHasher, BaseHasher>(m, "SparseSRPHasher",
        "Sparse Signed Random Projection (SRP) hasher for cosine similarity.\n\n"
        "Projects a sparse vector onto random ±1 hyperplanes generated on-the-fly\n"
        "from a seed (no matrix stored).  Packs `num_bits` sign bits per hash value.\n"
        "Collision probability is proportional to the cosine similarity between vectors.\n\n"
        "Members\n"
        "-------\n"
        "num_hashes: Number of independent hash values returned per call.\n"
        "seed      : Random seed used for projection generation.")
        .def(py::init<uint32_t, uint32_t, uint32_t>(),
             py::arg("num_bits") = 16,
             py::arg("seed") = 42,
             py::arg("num_hashes") = 1,
             "Construct a SparseSRPHasher.\n\n"
             "Args:\n"
             "    num_bits  : Number of SRP projection bits packed into each integer hash value.\n"
             "    seed      : Random seed for projection generation.\n"
             "    num_hashes: Number of independent hash values to compute.")
        .def("__call__", [](const SparseSRPHasher& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"),
             "Hash a sparse CSR row.\n\n"
             "Args:\n"
             "    data   : 1-D float32 numpy array of non-zero values.\n"
             "    indices: 1-D uint32 numpy array of column indices of the non-zero elements.\n"
             "    nnz    : Number of non-zero elements.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def("__call__", [](const SparseSRPHasher& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"),
             "Hash a dense query vector.\n\n"
             "Args:\n"
             "    query: Dense Eigen/numpy vector.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def_readwrite("num_hashes", &SparseSRPHasher::num_hashes,
             "Number of independent hash values returned per call.")
        .def_readwrite("seed", &SparseSRPHasher::seed,
             "Random seed for projection generation.");

    // DenseSRPHasher
    py::class_<DenseSRPHasher, BaseHasher>(m, "DenseSRPHasher",
        "Dense Signed Random Projection (SRP) hasher optimised for dense query vectors.\n\n"
        "Stores an explicit projection matrix of shape (num_hashes * num_bits, dimension)\n"
        "sampled from N(0,1).  At query time a single Eigen matrix-vector product yields\n"
        "all projections; sign bits are packed into integer hash values.\n"
        "Also supports sparse inputs via an explicit sparse-dense dot product.\n\n"
        "Members\n"
        "-------\n"
        "num_hashes: Number of independent hash values returned per call.\n"
        "seed      : Random seed used to generate the projection matrix.")
        .def(py::init<uint32_t, uint32_t, uint32_t, uint32_t, bool>(),
             py::arg("num_bits") = 16,
             py::arg("dimension") = 0,
             py::arg("seed") = 42,
             py::arg("num_hashes") = 1,
             py::arg("store") = false,
             "Construct a DenseSRPHasher.\n\n"
             "Args:\n"
             "    num_bits  : Number of projection bits packed into each integer hash value.\n"
             "    dimension : Dimensionality of the input vectors (required to build the matrix).\n"
             "    seed      : Random seed for projection matrix generation.\n"
             "    num_hashes: Number of independent hash values to compute.\n"
             "    store     : Unused parameter (matrix is always stored); kept for API compatibility.")
        .def("__call__", [](const DenseSRPHasher& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"),
             "Hash a sparse CSR row using the stored projection matrix.\n\n"
             "Args:\n"
             "    data   : 1-D float32 numpy array of non-zero values.\n"
             "    indices: 1-D uint32 numpy array of column indices of the non-zero elements.\n"
             "    nnz    : Number of non-zero elements.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def("__call__", [](const DenseSRPHasher& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"),
             "Hash a dense query vector using a matrix-vector product.\n\n"
             "Args:\n"
             "    query: Dense Eigen/numpy vector of length `dimension`.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def_readwrite("num_hashes", &DenseSRPHasher::num_hashes,
             "Number of independent hash values returned per call.")
        .def_readwrite("seed", &DenseSRPHasher::seed,
             "Random seed used to generate the projection matrix.");

    // BloomHashFunction
    py::class_<BloomHashFunction, BaseHasher>(m, "BloomHashFunction",
        "Bloom-filter-inspired compound SRP hasher; the default hasher for MLGTSaffronBloom.\n\n"
        "Generates `num_hashes` independent Sparse Signed Random Projection values, each\n"
        "encoding `num_bits` sign bits.  Projection weights are computed on-the-fly via\n"
        "SplitMix64, so no projection matrix is stored.  This is the hasher used by the\n"
        "standard MLGTSaffron (BloomHashFunction) variant.\n\n"
        "Members\n"
        "-------\n"
        "num_hashes: Number of compound hash values returned per call.\n"
        "seed      : Random seed; 0 means a hardware-random seed is drawn at construction.")
        .def(py::init<uint, uint, uint, uint, int, uint>(),
             py::arg("dimension"),
             py::arg("num_hashes") = BLOOM_NUM_HASHES,
             py::arg("num_bits") = BLOOM_HASH_BITS,
             py::arg("threshold") = BLOOM_THRESHOLD,
             py::arg("debug") = 0,
             py::arg("seed") = 0,
             "Construct a BloomHashFunction.\n\n"
             "Args:\n"
             "    dimension  : Dimensionality of the input vectors.\n"
             "    num_hashes : Number of independent compound hash functions (default BLOOM_NUM_HASHES).\n"
             "    num_bits   : SRP projection bits packed into each hash value (default BLOOM_HASH_BITS).\n"
             "    threshold  : Informational threshold passed to callers (default BLOOM_THRESHOLD).\n"
             "    debug      : Debug verbosity level (0 = silent).\n"
             "    seed       : Random seed; 0 triggers automatic hardware-random seeding.")
        .def("__call__", [](const BloomHashFunction& h, py::array_t<float> data, py::array_t<uint32_t> indices, uint32_t nnz) {
            return h(data.data(), indices.data(), nnz);
        }, py::arg("data"), py::arg("indices"), py::arg("nnz"),
             "Hash a sparse CSR row.\n\n"
             "Args:\n"
             "    data   : 1-D float32 numpy array of non-zero values.\n"
             "    indices: 1-D uint32 numpy array of column indices of the non-zero elements.\n"
             "    nnz    : Number of non-zero elements.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def("__call__", [](const BloomHashFunction& h, const Eigen::VectorXf& q) {
            return h(q);
        }, py::arg("query"),
             "Hash a dense query vector.\n\n"
             "Args:\n"
             "    query: Dense Eigen/numpy vector.\n"
             "Returns:\n"
             "    List[int] of length num_hashes.")
        .def_readwrite("num_hashes", &BloomHashFunction::num_hashes,
             "Number of compound hash values returned per call.")
        .def_readwrite("seed", &BloomHashFunction::seed,
             "Random seed; 0 means a hardware-random seed was drawn at construction.");


    // Abstract base class for all MLGTSaffron index variants
    py::class_<MLGTSaffronBase>(m, "MLGTSaffron",
        "Abstract base class for all MLGT nearest-neighbour index variants.\n\n"
        "Use for type annotations and isinstance() checks:\n\n"
        "    idx: MLGTSaffron = MLGTSaffronMinHash(...)\n"
        "    assert isinstance(idx, MLGTSaffron)\n\n"
        "Concrete variants and the similarity measure each targets:\n"
        "  MLGTSaffronBloom           – cosine  (Bloom/SRP, default)\n"
        "  MLGTSaffronMinHash         – Jaccard\n"
        "  MLGTSaffronWeightedMinHash – Weighted Jaccard\n"
        "  MLGTSaffronSparseSRP       – cosine  (sparse, matrix-free)\n"
        "  MLGTSaffronDenseSRP        – cosine  (stored projection matrix)")
        .def("search", &MLGTSaffronBase::search, py::arg("query"),
             "Search for the approximate nearest neighbours of a query vector.\n\n"
             "Args:\n"
             "    query: 1-D float32 numpy array of length num_cols.\n"
             "Returns:\n"
             "    List[int]: Up to num_neighbors item indices (sorted by global index).")
        .def("__call__", [](MLGTSaffronBase& self, py::array_t<float> q) { return self(q); },
             py::arg("query"),
             "Alias for search(query); see search() for full documentation.");
     
     // Abstract base class for all MLGTSaffron index variants
    py::class_<MLGTGlobalBase>(m, "MLGTGlobal",
        "Abstract base class for all MLGT nearest-neighbour index variants.\n\n"
        "Use for type annotations and isinstance() checks:\n\n"
        "    idx: MLGTGlobal = MLGTGlobalMinHash(...)\n"
        "    assert isinstance(idx, MLGTGlobal)\n\n"
        "Concrete variants and the similarity measure each targets:\n"
        "  MLGTGlobalBloom           – cosine  (Bloom/SRP, default)\n"
        "  MLGTGlobalMinHash         – Jaccard\n"
        "  MLGTGlobalWeightedMinHash – Weighted Jaccard\n"
        "  MLGTGlobalSparseSRP       – cosine  (sparse, matrix-free)\n"
        "  MLGTGlobalDenseSRP        – cosine  (stored projection matrix)")
        .def("search", &MLGTGlobalBase::search, py::arg("query"),
             "Search for the approximate nearest neighbours of a query vector.\n\n"
             "Args:\n"
             "    query: 1-D float32 numpy array of length num_cols.\n"
             "Returns:\n"
             "    List[int]: Up to num_neighbors item indices (sorted by global index).")
        .def("__call__", [](MLGTGlobalBase& self, py::array_t<float> q) { return self(q); },
             py::arg("query"),
             "Alias for search(query); see search() for full documentation.");

    // Helper to register concrete MLGTSaffron variants (all inherit MLGTSaffronBase)
    auto register_mlgt = [&](auto& module, const char* name, auto type_ptr) {
        using T = typename std::remove_pointer<decltype(type_ptr)>::type;
        py::class_<T, MLGTSaffronBase>(module, name,
            "MLGT approximate nearest-neighbour index using the SAFFRON recovery scheme.\n\n"
            "Indexes a sparse dataset once at construction time.  Each query is answered\n"
            "in sub-linear time by:\n"
            "  1. Hashing the query with the configured Hasher.\n"
            "  2. Querying each pool's PoolInvertedIndex to obtain a binary residual.\n"
            "  3. Running the SAFFRON peeling algorithm over the residuals to identify\n"
            "     candidate item indices.\n"
            "  4. Re-scoring candidates by exact dot product and returning the top-k.\n\n"
            "search() and __call__() are inherited from MLGTSaffron.")
            .def(py::init<py::array_t<float>, py::array_t<uint32_t>, py::array_t<uint64_t>, uint32_t, typename T::HasherAlias, uint, uint, uint, uint, int, bool>(),
                 py::arg("data"),
                 py::arg("indices"),
                 py::arg("indptr"),
                 py::arg("num_cols"),
                 py::arg("hasher"),
                 py::arg("num_neighbors") = 100,
                 py::arg("num_pools") = 0,
                 py::arg("pools_per_item") = POOLS_PER_ITEM,
                 py::arg("threshold") = BLOOM_THRESHOLD,
                 py::arg("debug") = 0,
                 py::arg("normalize") = true,
                 "Build an MLGT nearest-neighbour index.\n\n"
                 "Args:\n"
                 "    data        : 1-D float32 numpy array – the CSR data array of the dataset.\n"
                 "    indices     : 1-D uint32 numpy array – the CSR column-indices array.\n"
                 "    indptr      : 1-D uint64 numpy array – the CSR row-pointer array (length n+1).\n"
                 "    num_cols    : Total number of feature dimensions (columns).\n"
                 "    hasher      : An initialised Hasher instance (MinHasher, BloomHashFunction, etc.).\n"
                 "    num_neighbors: Number of nearest neighbours to recover per query (k).\n"
                 "    num_pools   : Number of SAFFRON measurement pools; 0 = k * NUM_POOLS_COEFF.\n"
                 "    pools_per_item: Number of pools each item is assigned to.\n"
                 "    threshold   : Minimum hash-match count for a pool hit to count as a candidate.\n"
                 "    debug       : Debug verbosity level (0 = silent).\n"
                 "    normalize   : If True, all dataset vectors and queries are L2-normalised before processing.");
    };

    // Helper to register concrete MLGTGlobal variants (all inherit MLGTGlobalBase)
    auto register_mlgt_global = [&](auto& module, const char* name, auto type_ptr) {
        using T = typename std::remove_pointer<decltype(type_ptr)>::type;
        py::class_<T, MLGTGlobalBase>(module, name,
            "MLGT approximate nearest-neighbour index using the a hash-value inverted index.\n\n"
            "Indexes a sparse dataset once at construction time.  Each query is answered\n"
            "in sub-linear time by:\n"
            "  1. Hashing the query with the configured Hasher.\n"
            "  2. Querying each pool's PoolInvertedIndex to obtain a binary residual.\n"
            "  3. Running the SAFFRON peeling algorithm over the residuals to identify\n"
            "     candidate item indices.\n"
            "  4. Re-scoring candidates by exact dot product and returning the top-k.\n\n"
            "search() and __call__() are inherited from MLGTSaffron.")
            .def(py::init<py::array_t<float>, py::array_t<uint32_t>, py::array_t<uint64_t>, uint32_t, typename T::HasherAlias, uint, uint, uint, int, bool>(),
                 py::arg("data"),
                 py::arg("indices"),
                 py::arg("indptr"),
                 py::arg("num_cols"),
                 py::arg("hasher"),
                 py::arg("num_neighbors") = 100,
                 py::arg("num_pools") = 0,
                 py::arg("threshold") = BLOOM_THRESHOLD,
                 py::arg("debug") = 0,
                 py::arg("normalize") = true,
                 "Build an MLGT nearest-neighbour index.\n\n"
                 "Args:\n"
                 "    data        : 1-D float32 numpy array – the CSR data array of the dataset.\n"
                 "    indices     : 1-D uint32 numpy array – the CSR column-indices array.\n"
                 "    indptr      : 1-D uint64 numpy array – the CSR row-pointer array (length n+1).\n"
                 "    num_cols    : Total number of feature dimensions (columns).\n"
                 "    hasher      : An initialised Hasher instance (MinHasher, BloomHashFunction, etc.).\n"
                 "    num_neighbors: Number of nearest neighbours to recover per query (k).\n"
                 "    num_pools   : Number of SAFFRON measurement pools; 0 = k * NUM_POOLS_COEFF.\n"
                 "    threshold   : Minimum hash-match count for a pool hit to count as a candidate.\n"
                 "    debug       : Debug verbosity level (0 = silent).\n"
                 "    normalize   : If True, all dataset vectors and queries are L2-normalised before processing.");
    };

    register_mlgt(m, "MLGTSaffronBloom", (MLGTSaffronBloom*)nullptr);
    register_mlgt(m, "MLGTSaffronMinHash", (MLGTSaffronMinHash*)nullptr);
    register_mlgt(m, "MLGTSaffronWeightedMinHash", (MLGTSaffronWeightedMinHash*)nullptr);
    register_mlgt(m, "MLGTSaffronSparseSRP", (MLGTSaffronSparseSRP*)nullptr);
    register_mlgt(m, "MLGTSaffronDenseSRP", (MLGTSaffronDenseSRP*)nullptr);

     register_mlgt_global(m, "MLGTGlobalBloom", (MLGTGlobalBloom*)nullptr);
     register_mlgt_global(m, "MLGTGlobalMinHash", (MLGTGlobalMinHash*)nullptr);
     register_mlgt_global(m, "MLGTGlobalWeightedMinHash", (MLGTGlobalWeightedMinHash*)nullptr);
     register_mlgt_global(m, "MLGTGlobalSparseSRP", (MLGTGlobalSparseSRP*)nullptr);
     register_mlgt_global(m, "MLGTGlobalDenseSRP", (MLGTGlobalDenseSRP*)nullptr);
}
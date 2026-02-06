#include "naive_search.hpp"
#include "linscan.hpp"
#include "knn_index.hpp"
#include "threshold_index.hpp"
#include "threshold_index_randomized.hpp"
#include "sparse_types.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

PYBIND11_MODULE(sparse_gtnn, m) {
	m.doc() = "GTNN: Sparse Graph-based Nearest Neighbor Search Library";

	// --- sparse_types.hpp ---
	pybind11::class_<SparseElem>(m, "SparseElem", "Represents a single element in a sparse vector")
		.def(pybind11::init<>(), "Initialize an empty sparse element")
		.def_readwrite("index", &SparseElem::index, "Column index of the non-zero element")
		.def_readwrite("value", &SparseElem::value, "Value of the non-zero element");
	
	// Export sparse types as type aliases with docstrings
	// SparseVec is truly List[SparseElem] - use typing annotations
	pybind11::object typing = pybind11::module_::import("typing");
	pybind11::object SparseVec_type = typing.attr("List")[pybind11::type::of<SparseElem>()];
	SparseVec_type.attr("__doc__") = 
		"Type alias for List[SparseElem] representing a single sparse vector.\n\n"
		"Each sparse vector contains only non-zero elements with their column indices.";
	m.attr("SparseVec") = SparseVec_type;
	
	// SparseMat is truly List[SparseVec] - reference the alias we just created
	pybind11::object SparseMat_type = typing.attr("List")[SparseVec_type];
	SparseMat_type.attr("__doc__") = 
		"Type alias for List[SparseVec] representing a sparse matrix.\n\n"
		"Each row is a SparseVec, allowing for efficient storage of sparse data.";
	m.attr("SparseMat") = SparseMat_type;
	
	pybind11::class_<SparseVecOptimized>(
		m, "SparseVecOptimized", "Optimized sparse vector using Structure of Arrays (SoA)"
	)
		.def(pybind11::init<>())
		.def("reserve", &SparseVecOptimized::reserve, "Reserve space for elements", pybind11::arg("size"))
		.def("push_back", &SparseVecOptimized::push_back, "Add a new element to the sparse vector", 
			pybind11::arg("idx"), pybind11::arg("val"))
		.def("size", &SparseVecOptimized::size, "Get the number of non-zero elements")
		.def_readwrite("indices", &SparseVecOptimized::indices, "Column indices of non-zero elements")
		.def_readwrite("values", &SparseVecOptimized::values, "Values of non-zero elements");
	
		pybind11::class_<SparseMatOptimized>(m, "SparseMatOptimized", "Optimized sparse matrix using SoA structure")
		.def(pybind11::init<>()
	);
	
	pybind11::class_<CSRMatrix>(m, "CSRMatrix", "Compressed Sparse Row (CSR) matrix format")
		.def(pybind11::init<>()
	);
	
	m.def("check_file", (bool(*)(std::ofstream&)) &check_file, 
		"Check if an output file stream is open"
	);
	m.def("check_file", (bool(*)(std::ifstream&)) &check_file, 
		"Check if an input file stream is open"
	);
	m.def("check_file", (bool(*)(FILE*)) &check_file, 
		"Check if a C file pointer is valid"
	);
	m.def("check_file", (bool(*)(const std::string&)) &check_file, 
		"Check if a file exists at the given path", 
		pybind11::arg("file_name") = "Path to file to check"
	);
	m.def("check_file", (bool(*)(const char *)) &check_file, 
		"Check if a file exists at the given path", 
		pybind11::arg("file_name") = "Path to file to check");
	m.def("read_sparse_matrix", &read_sparse_matrix, 
		"Read a sparse matrix from a binary CSR file\n\nArgs:\n  file_name: Path to the binary CSR matrix file\n\nReturns:\n  Pair of (sparse_matrix, dimensionality)", 
		pybind11::arg("file_name") = "Path to binary CSR file");
	m.def("add_sparse", &add_sparse, 
		"Element-wise addition of two sparse vectors\n\nArgs:\n  one: First sparse vector\n  two: Second sparse vector\n\nReturns:\n  Sparse vector containing element-wise sum", 
		pybind11::arg("one") = "First sparse vector", 
		pybind11::arg("two") = "Second sparse vector");
	m.def("dot_product", &dot_product, 
		"Compute the dot product of two sparse vectors\n\nArgs:\n  one: First sparse vector\n  two: Second sparse vector\n\nReturns:\n  Dot product as a double", 
		pybind11::arg("one") = "First sparse vector", 
		pybind11::arg("two") = "Second sparse vector");
	m.def("compare_sparse", &compare_sparse, 
		"Compare two sparse vectors for approximate equality\n\nArgs:\n  one: First sparse vector\n  two: Second sparse vector\n\nReturns:\n  True if vectors are approximately equal", 
		pybind11::arg("one") = "First sparse vector", 
		pybind11::arg("two") = "Second sparse vector");
	m.def("compare_float", &compare_float, 
		"Compare two float values for approximate equality (tolerance: 1e-5)\n\nArgs:\n  one: First float value\n  two: Second float value\n\nReturns:\n  True if values are approximately equal", 
		pybind11::arg("one") = "First float value", 
		pybind11::arg("two") = "Second float value");
	m.def("create_random_csr_matrix", &create_random_csr_matrix, 
		"Create a random CSR matrix with specified sparsity\n\nArgs:\n  num_rows: Number of rows\n  num_cols: Number of columns\n  sparsity: Sparsity parameter for Poisson distribution\n\nReturns:\n  CSR matrix with unit-norm rows", 
		pybind11::arg("num_rows") = "Number of rows", 
		pybind11::arg("num_cols") = "Number of columns", 
		pybind11::arg("sparsity") = "Sparsity parameter (controls density)");
	m.def("read_csr_matrix", &read_csr_matrix, 
		"Read a CSR matrix from a binary file\n\nArgs:\n  filename: Path to binary file\n  num_rows: Output parameter for number of rows\n  num_cols: Output parameter for number of columns\n\nReturns:\n  CSR matrix as vector of (indices, values) pairs",
		pybind11::arg("filename") = "Path to binary CSR file",
		pybind11::arg("num_rows") = "Output: number of rows",
		pybind11::arg("num_cols") = "Output: number of columns");

	// --- naive_search.hpp ---
	pybind11::class_<NaiveSearch>(m, "NaiveSearch", "Naive brute-force search for both KNN and threshold-based search.\n\nUses exhaustive comparison against all dataset vectors. Supports both single and batch query processing.")
		.def(pybind11::init<SparseMat&, double, size_t, bool>(),
			"Initialize a NaiveSearch index\n\nArgs:\n  dataset: Sparse matrix containing data vectors\n  threshold: Similarity threshold for threshold-based search (default 0.8)\n  k: Number of nearest neighbors for KNN (default 1)\n  is_knn: If True, performs KNN; if False, performs threshold search (default True)",
			pybind11::arg("dataset") = "Sparse matrix with data vectors",
			pybind11::arg("threshold") = "Similarity threshold (default 0.8)",
			pybind11::arg("k") = "Number of neighbors (default 1)",
			pybind11::arg("is_knn") = "KNN mode if True, threshold mode if False (default True)")
		.def("search", &NaiveSearch::search, 
			"Search for neighbors of a single query vector\n\nArgs:\n  query: Query sparse vector\n\nReturns:\n  Tuple of (result_indices, num_dot_products_computed)",
			pybind11::arg("query") = "Query sparse vector")
		.def("search_multiple", &NaiveSearch::search_multiple,
			"Search for neighbors of multiple query vectors (batch)\n\nArgs:\n  queries: Batch of query sparse vectors\n\nReturns:\n  Tuple of (list of result_indices per query, total_dot_products_computed)",
			pybind11::arg("queries") = "Batch of query sparse vectors")
		.def("verify_results", &NaiveSearch::verify_results,
			"Verify search results and compute metrics\n\nArgs:\n  query: Query sparse vector\n  result: Indices returned by search algorithm\n\nReturns:\n  List [time_ms, recall, precision]",
			pybind11::arg("query") = "Query sparse vector",
			pybind11::arg("result") = "Result indices to verify");

	// --- linscan.hpp ---
	pybind11::class_<Linscan>(m, "Linscan", "Inverted index based threshold search using linear scans.\n\nBuilds an inverted index for efficient single-query threshold search. Supports batch queries.")
		.def(pybind11::init<SparseMat&, double>(),
			"Initialize a Linscan index\n\nArgs:\n  dataset: Sparse matrix containing data vectors\n  threshold: Similarity threshold for search (default 0.8)",
			pybind11::arg("dataset") = "Sparse matrix with data vectors",
			pybind11::arg("threshold") = "Similarity threshold (default 0.8)")
		.def("search", &Linscan::search,
			"Search for vectors above threshold for a single query\n\nArgs:\n  query: Query sparse vector\n\nReturns:\n  Tuple of (result_indices, num_dot_products_computed)",
			pybind11::arg("query") = "Query sparse vector")
		.def("search_multiple", &Linscan::search_multiple,
			"Search for vectors above threshold for multiple queries (batch)\n\nArgs:\n  queries: Batch of query sparse vectors\n\nReturns:\n  Tuple of (list of result_indices per query, total_dot_products_computed)",
			pybind11::arg("queries") = "Batch of query sparse vectors")
		.def("verify_results", &Linscan::verify_results,
			"Verify search results and compute metrics\n\nArgs:\n  query: Query sparse vector\n  result: Indices returned by search algorithm\n\nReturns:\n  List [time_ms, recall, precision]",
			pybind11::arg("query") = "Query sparse vector",
			pybind11::arg("result") = "Result indices to verify");

	// --- knn_index.hpp ---
	pybind11::class_<KNNIndexDataset>(m, "KNNIndexDataset", 
		"Hierarchical KNN search with double-group-testing for batch queries.\n\nBuilds a binary tree hierarchy on the dataset. Implements double-group-testing algorithm for efficient batch query processing.")
		.def(pybind11::init<SparseMat&, size_t, bool>(),
			"Initialize a KNNIndexDataset index\n\nArgs:\n  dataset: Sparse matrix containing data vectors\n  k: Number of nearest neighbors to find (default 1)\n  use_threading: Enable multi-threading for search (default False)",
			pybind11::arg("dataset") = "Sparse matrix with data vectors",
			pybind11::arg("k") = "Number of neighbors to find (default 1)",
			pybind11::arg("use_threading") = "Enable threading (default False)")
		.def("search", &KNNIndexDataset::search,
			"Search for k nearest neighbors of a single query\n\nArgs:\n  query: Query sparse vector\n\nReturns:\n  Tuple of (result_indices, num_dot_products_computed)",
			pybind11::arg("query") = "Query sparse vector")
		.def("search_multiple", &KNNIndexDataset::search_multiple,
			"Search for k nearest neighbors of multiple queries using double-group-testing\n\nArgs:\n  queries: Batch of query sparse vectors\n\nReturns:\n  Tuple of (list of result_indices per query, total_dot_products_computed)",
			pybind11::arg("queries") = "Batch of query sparse vectors")
		.def("verify_results", &KNNIndexDataset::verify_results,
			"Verify search results and compute metrics\n\nArgs:\n  query: Query sparse vector\n  result: Indices returned by search algorithm\n\nReturns:\n  List [time_ms, recall, precision]",
			pybind11::arg("query") = "Query sparse vector",
			pybind11::arg("result") = "Result indices to verify");

	// --- threshold_index.hpp ---
	pybind11::class_<ThresholdIndexDataset>(m, "ThresholdIndexDataset",
		"Hierarchical threshold search with dual-hierarchy recursion for batch queries.\n\nBuilds a binary tree hierarchy on both dataset and queries. Implements dual-hierarchy recursion for efficient batch processing.")
		.def(pybind11::init<SparseMat&, double, bool>(),
			"Initialize a ThresholdIndexDataset index\n\nArgs:\n  dataset: Sparse matrix containing data vectors\n  threshold: Similarity threshold for search (default 0.8)\n  use_threading: Enable multi-threading for search (default False)",
			pybind11::arg("dataset") = "Sparse matrix with data vectors",
			pybind11::arg("threshold") = "Similarity threshold (default 0.8)",
			pybind11::arg("use_threading") = "Enable threading (default False)")
		.def("search", &ThresholdIndexDataset::search,
			"Search for vectors above threshold for a single query\n\nArgs:\n  query: Query sparse vector\n\nReturns:\n  Tuple of (result_indices, num_dot_products_computed)",
			pybind11::arg("query") = "Query sparse vector")
		.def("search_multiple", &ThresholdIndexDataset::search_multiple,
			"Search for vectors above threshold for multiple queries using dual-hierarchy recursion\n\nArgs:\n  queries: Batch of query sparse vectors\n\nReturns:\n  Tuple of (list of result_indices per query, total_dot_products_computed)",
			pybind11::arg("queries") = "Batch of query sparse vectors")
		.def("verify_results", &ThresholdIndexDataset::verify_results,
			"Verify search results and compute metrics\n\nArgs:\n  query: Query sparse vector\n  result: Indices returned by search algorithm\n\nReturns:\n  List [time_ms, recall, precision]",
			pybind11::arg("query") = "Query sparse vector",
			pybind11::arg("result") = "Result indices to verify");

	// --- threshold_index_randomized_new.hpp ---
	pybind11::class_<ThresholdIndexRandomized>(m, "ThresholdIndexRandomized",
		"Page-based randomized threshold search with multi-threaded execution.\n\nOrganizes dataset into pages with cumulative vectors. Executes multi-threaded hierarchical search over pages.")
		.def(pybind11::init<SparseMat&, double>(),
			"Initialize a ThresholdIndexRandomized index\n\nArgs:\n  dataset: Sparse matrix containing data vectors\n  threshold: Similarity threshold for search (default 0.8)",
			pybind11::arg("dataset") = "Sparse matrix with data vectors",
			pybind11::arg("threshold") = "Similarity threshold (default 0.8)")
		.def("search", &ThresholdIndexRandomized::search,
			"Search for vectors above threshold using multi-threaded page-based search\n\nArgs:\n  query: Query sparse vector\n\nReturns:\n  Tuple of (result_indices, num_dot_products_computed)",
			pybind11::arg("query") = "Query sparse vector")
		.def("verify_results", &ThresholdIndexRandomized::verify_results,
			"Verify search results and compute metrics\n\nArgs:\n  query: Query sparse vector\n  result: Indices returned by search algorithm\n\nReturns:\n  List [time_ms, recall, precision]",
			pybind11::arg("query") = "Query sparse vector",
			pybind11::arg("result") = "Result indices to verify");
}

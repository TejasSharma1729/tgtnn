#include "header.hpp"
#include "knns_dataset.hpp"
#include "tnns_dataset.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gtnn, m) {
    m.doc() = "Optimized Group Testing Nearest Neighbors module"; 

    // Utility functions from header.hpp
    m.def("read_matrix", [](const string &file_name) {
        matrix_t mat;
        extract_matrix(file_name, mat);
        return mat;
    }, "Read a matrix from a binary file into a numpy array", 
    py::arg("file_name"));

    m.def("save_matrix", &save_matrix, "Save a numpy array/matrix to a binary file", 
          py::arg("file_name"), py::arg("matrix"));

    // KNNSIndexDataset bindings
    py::class_<KNNSIndexDataset>(m, "KNNSIndexDataset")
        .def(py::init<matrix_t &, size_t>(), 
             py::arg("mat"), py::arg("k_val") = 10,
             "Initialize the KNNS dataset structure.\n\n"
             "Args:\n"
             "    mat (np.ndarray): The dataset matrix.\n"
             "    k_val (int): The number of nearest neighbors to find (default: 10).")
             
        .def("search", &KNNSIndexDataset::search, 
             py::arg("query"),
             "Search for the K nearest neighbors of a single query vector.\n\n"
             "Args:\n"
             "    query (np.ndarray): The query vector.\n\n"
             "Returns:\n"
             "    tuple: (list of neighbor indices, number of dot products computed)")
             
        .def("search_multiple", &KNNSIndexDataset::search_multiple, 
             py::arg("query_set"),
             "Search for K nearest neighbors for a batch of queries using optimized double group testing.\n\n"
             "Args:\n"
             "    query_set (np.ndarray): The matrix of query vectors.\n\n"
             "Returns:\n"
             "    tuple: (map of {query_index: list of neighbor indices}, total dot products computed)")
             
        .def("verify_results", &KNNSIndexDataset::verify_results, 
             py::arg("query"), py::arg("result"),
             "Verify the accuracy of the search results against a brute-force ground truth.\n\n"
             "Args:\n"
             "    query (np.ndarray): The query vector.\n"
             "    result (list): The list of neighbor indices returned by search.\n\n"
             "Returns:\n"
             "    list: [time_taken_ms, recall/accuracy]");

    // ThresholdIndexDataset bindings
    py::class_<ThresholdIndexDataset>(m, "ThresholdIndexDataset")
        .def(py::init<matrix_t &, double>(), 
             py::arg("mat"), py::arg("threshold") = 0.8,
             "Initialize the Threshold dataset structure.\n\n"
             "Args:\n"
             "    mat (np.ndarray): The dataset matrix.\n"
             "    threshold (double): The cosine similarity threshold (default: 0.8).")
             
        .def("search", &ThresholdIndexDataset::search, 
             py::arg("query"),
             "Search for neighbors within threshold for a single query.\n\n"
             "Args:\n"
             "    query (np.ndarray): The query vector.\n\n"
             "Returns:\n"
             "    tuple: (list of neighbor indices, number of dot products computed)")
             
        .def("search_multiple", &ThresholdIndexDataset::search_multiple, 
             py::arg("queries"),
             "Search for neighbors for a batch of queries using optimized double group testing.\n\n"
             "Args:\n"
             "    queries (np.ndarray): The matrix of query vectors.\n\n"
             "Returns:\n"
             "    tuple: (list of lists of neighbor indices per query, total dot products computed)")
             
        .def("verify_results", &ThresholdIndexDataset::verify_results, 
             py::arg("query"), py::arg("results"),
             "Verify the accuracy of the search results against a brute-force ground truth.\n\n"
             "Args:\n"
             "    query (np.ndarray): The query vector.\n"
             "    results (list): The list of neighbor indices returned by search.\n\n"
             "Returns:\n"
             "    list: [time_taken_ms, precision, recall]");
}

#include "header.hpp"
#include "knns_dataset_reordered.hpp"

namespace py = pybind11;

// Register one (L, Lq) instantiation as a Python class named `name`.
template <uint L, uint Lq>
void register_dataset(py::module &m, const char *name) {
    using DS = KNNReorderedIndexDatasetT<L, Lq>;
    py::class_<DS>(m, name)
        .def(py::init<matrix_t &, size_t>(),
             py::arg("mat"), py::arg("k_val") = 10)
        .def("streaming_update",   &DS::streaming_update,   py::arg("mat"))
        .def("search",             &DS::search,             py::arg("query"), py::arg("use_threading") = true)
        .def("search_batch_binary",&DS::search_batch_binary,py::arg("query_set"), py::arg("use_threading") = true)
        .def("search_multiple",    &DS::search_multiple,    py::arg("query_set"), py::arg("use_threading") = true)
        .def("verify_results",     &DS::verify_results,     py::arg("query"), py::arg("result"));
}

PYBIND11_MODULE(gtnn_ablation, m) {
    m.doc() = "Ablation module: KNNReorderedIndexDatasetT<L,Lq> for varying L (KNNS_TREE_LEVELS) and Lq (KNNS_INVERTED_LEVELS)";

    // ── L ablation: Lq fixed at 5, L ∈ {6,7,8,9,10,11,12} ──────────────
    register_dataset< 6, 5>(m, "KNNReordered_L6_Lq5");
    register_dataset< 7, 5>(m, "KNNReordered_L7_Lq5");
    register_dataset< 8, 5>(m, "KNNReordered_L8_Lq5");
    register_dataset< 9, 5>(m, "KNNReordered_L9_Lq5");
    register_dataset<10, 5>(m, "KNNReordered_L10_Lq5");
    register_dataset<11, 5>(m, "KNNReordered_L11_Lq5");
    register_dataset<12, 5>(m, "KNNReordered_L12_Lq5");

    // ── Lq ablation: L fixed at 10, Lq ∈ {2,3,4,5,6,7,8} ──────────────
    // L10_Lq5 already registered above; skip duplicate
    register_dataset<10, 2>(m, "KNNReordered_L10_Lq2");
    register_dataset<10, 3>(m, "KNNReordered_L10_Lq3");
    register_dataset<10, 4>(m, "KNNReordered_L10_Lq4");
    register_dataset<10, 6>(m, "KNNReordered_L10_Lq6");
    register_dataset<10, 7>(m, "KNNReordered_L10_Lq7");
    register_dataset<10, 8>(m, "KNNReordered_L10_Lq8");
}

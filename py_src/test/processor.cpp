#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

class DataProcessor {
private:
    std::vector<float> data;
    size_t size;

public:
    // Constructor: Allocates memory for large data
    DataProcessor(size_t n) : size(n) {
        data.resize(n, 0.0f); // Initialize with zeros
        std::cout << "[C++] Constructor: Allocated " << n << " elements." << std::endl;
    }

    // Destructor: Clean up happens automatically with std::vector, 
    // but we add a print here to show lifecycle management.
    ~DataProcessor() {
        std::cout << "[C++] Destructor: Releasing memory." << std::endl;
    }

    // A method to simulate heavy processing
    void add_scalar(float value) {
        for (size_t i = 0; i < size; i++) {
            data[i] += value;
        }
    }

    // METHOD FOR NUMPY COMPATIBILITY
    // This returns a NumPy array wrapper around the C++ vector.
    // We use the "capsule" concept to ensure the C++ object isn't destroyed
    // while the NumPy array is still using its memory.
    py::array_t<float> get_data_view() {
        return py::array_t<float>(
            { size },               // Shape of the array
            { sizeof(float) },      // Strides (byte distance between elements)
            data.data(),            // Pointer to the data
            py::cast(this)          // The object that owns the memory
        );
    }
};

// This block defines the Python module name and bindings
PYBIND11_MODULE(fast_module, m) {
    m.doc() = "A C++ extension for high-performance data processing";

    py::class_<DataProcessor>(m, "DataProcessor")
        .def(py::init<size_t>())           // Bind the constructor
        .def("add_scalar", &DataProcessor::add_scalar)
        .def("get_data_view", &DataProcessor::get_data_view);
}

#include <bits/stdc++.h>
typedef uint32_t uint;
using CSRMatrix = std::vector<std::pair<std::vector<uint>, std::vector<float>>>;


// Function to read a CSR matrix from binary file
CSRMatrix read_csr_matrix(const std::string& filename, uint& num_rows, uint& num_cols) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file: " + filename);

    // Read matrix dimensions
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(uint));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(uint));
    

    // Read number of nonzero elements
    uint num_nonzero = 0;
    file.read(reinterpret_cast<char*>(&num_nonzero), sizeof(uint));
    std::cout << "num_rows: " << num_rows << " num_cols: " << num_cols << " num_nonzero: " << num_nonzero << std::endl;

    // Allocate space for CSR components
    std::vector<float> values(num_nonzero);
    std::vector<uint> indices(num_nonzero);
    std::vector<size_t> indptr(num_rows + 1);

    // Read binary data
    file.read(reinterpret_cast<char*>(values.data()), num_nonzero * sizeof(float));
    file.read(reinterpret_cast<char*>(indices.data()), num_nonzero * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(indptr.data()), (num_rows + 1) * sizeof(size_t));

    file.close();

    // Convert to `std::vector<std::pair<std::vector<uint32_t>, std::vector<float>>>`
    CSRMatrix csr_matrix(num_rows);

    for (size_t i = 0; i < num_rows; i++) {
        size_t start = indptr[i];
        size_t end = indptr[i + 1];

        csr_matrix[i].first.assign(indices.begin() + start, indices.begin() + end);
        csr_matrix[i].second.assign(values.begin() + start, values.begin() + end);
    }

    return csr_matrix;
}
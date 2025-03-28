#include <bits/stdc++.h>
using namespace std;
typedef unsigned int uint;
typedef vector<pair<vector<uint>, vector<float>>> CSRMatrix;

// Function to create a random CSR matrix
CSRMatrix create_random_csr_matrix(uint num_rows, uint num_cols, uint sparsity) {
    // Initialize random number generator
    random_device rd;
    mt19937 gen(rd());
    poission_distribution<uint> nnz_dist(sparsity);
    exponential_distribution<float> val_dist(1.0);

    // Create CSR matrix
    CSRMatrix csr_matrix(num_rows);

    for (uint i = 0; i < num_rows; i++) {
        uint start = 0;
        float norm = 0;
        while (true) {
            start += nnz_dist(gen);
            if (start >= num_cols) {
                break;
            }
            csr_matrix[i].first.push_back(start);
            float val = val_dist(gen);
            csr_matrix[i].second.push_back(val);
            norm += val * val;
        }
        for (uint j = 0; j < csr_matrix[i].second.size(); j++) {
            csr_matrix[i].second[j] /= sqrt(norm);
        }
    }

    return csr_matrix;
}
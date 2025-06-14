#include "GTnn/header.hpp"

int main(int argc, char **argv) {
    string file_name = argv[1];
    auto [data_set, dimension] = GTnn::read_sparse_matrix(file_name);
    vector<vector<uint>> correlations(dimension, vector<uint>(dimension, 0));
    for (size_t i = 0; i < data_set.size(); i++) {
        for (auto [dim1, val1] : data_set[i]) {
            for (auto [dim2, val2] : data_set[i]) {
                if (dim1 != dim2) {
                    correlations[dim1][dim2] += val1 * val2;
                    correlations[dim2][dim1] += val1 * val2;
                }
            }
        }
    }
    for (size_t i = 0; i < dimension; i++) {
        for (size_t j = 0; j < dimension; j++) {
            cout << i << " " << j << " " << correlations[i][j] << endl;
        }
        cout << endl;
    }
}
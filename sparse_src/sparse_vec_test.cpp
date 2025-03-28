#include "GTnn/SPARSE_GTNN_SUM.hpp"
#include <unistd.h>
#include <sys/wait.h>
using namespace std;

array<size_t, 1000> dot_products_matrix(size_t aa) {
    string arg = (aa < 10 ? "0" : "") + to_string(aa);
    int pid, status;
    if ((pid = fork()) == 0) {
        execl("/home/tejassharma/miniconda3/envs/ann/bin/python3", "python3", "./read_movielens.py", arg.c_str(), NULL);
        exit(127);
    }
    waitpid(pid, &status, 0);
    if (status != 0) {
        cerr << "Error reading movielens data, status " << status << endl;
        exit(1);
    }
    auto [matrix, dimention] = GT::read_sparse_matrix("movielens/shard_" + arg + "/X.txt");
    auto [query_set, _] = GT::read_sparse_matrix("movielens/shard_" + arg + "/Q.txt");
    array<size_t, 1000> num_dot_products;
    for (size_t i = 0; i < 1000; i++) {
        num_dot_products[i] = 0;
    }
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < query_set.size(); j++) {
            num_dot_products[min<size_t>(999, static_cast<size_t>(1000 * (matrix[i] * query_set[j])))]++;
        }
    }
    return num_dot_products;
}

int main(int argc, char **argv) {
    array<size_t, 1000> num_dot_products;
    for (size_t i = 0; i < 16; i++) {
        auto num_dot_products_i = dot_products_matrix(i);
        for (size_t j = 0; j < 1000; j++) {
            num_dot_products[j] += num_dot_products_i[j];
        }
    }
    fstream outfile("histogram_data.txt");
    for (size_t i = 0; i < 1000; i++) {
        outfile << "[" << (i / 1000.0) << ", " << ((i + 1) / 1000.0) << "]: " << setw(12) << num_dot_products[i] << endl;
    }
    int pid, status;
    if ((pid = fork()) == 0) {
        execl("/home/tejassharma/miniconda3/envs/ann/bin/python3", "python3", "./histogram_maker.py", NULL);
        exit(1);
    }
    waitpid(pid, &status, 0);
    return status;
}
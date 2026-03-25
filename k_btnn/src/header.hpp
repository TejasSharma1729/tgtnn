#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#if __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#include <Eigen/Sparse>
#else
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#endif

using namespace std;
using namespace std::chrono;
using namespace Eigen;

#ifndef __OPTIMIZED_GTNN_HEADER
#define __OPTIMIZED_GTNN_HEADER

#define NUM_THREADS 16
typedef Eigen::Matrix<double, 
            Eigen::Dynamic, 
            Eigen::Dynamic, 
            Eigen::RowMajor
> matrix_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> vector_t;

bool check_file(ifstream &file);
bool check_file(ofstream &file);
bool check_file(FILE *file);
bool check_file(const string &file_name);
bool check_file(const char *file_name);

string path_append(const string &path, const string &file_name);
bool recursive_mkdir(const string &path);
size_t extract_matrix(const string &file_name, matrix_t &matrix);
bool save_matrix(const string &file_name, const matrix_t &matrix);
bool compare_float(float one, float two) {
    return fabs(one - two) < 1.0e-5;
}

bool check_file(ifstream &file) {
    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        return false;
    }
    return true;
}

bool check_file(ofstream &file) {
    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        return false;
    }
    return true;
}

bool check_file(FILE *file) {
    if (file == nullptr) {
        cerr << "Error opening file" << endl;
        return false;
    }
    return true;
}

bool check_file(const string &file_name) {
    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return false;
    }
    return true;
}

bool check_file(const char *file_name) {
    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return false;
    }
    return true;
}

string path_append(const string &path, const string &file_name) {
    return path + "/" + file_name;
}

bool recursive_mkdir(const string &path) {
    istringstream iss(path);
    string dir;
    while (getline(iss, dir, '/')) {
        if (dir.empty()) continue; // Skip empty segments
        if (mkdir(dir.c_str(), 0777) == -1 && errno != EEXIST) {
            cerr << "Error creating directory: " << dir << endl;
            return false;
        }
        if (chdir(dir.c_str()) == -1) {
            cerr << "Error changing directory: " << dir << endl;
            return false;
        }
    }
    return true;
}

size_t extract_matrix(const string &file_name, matrix_t &matrix) {
    try {
        pybind11::module_ np = pybind11::module_::import("numpy");
        pybind11::object loaded = np.attr("load")(file_name);
        matrix = loaded.cast<matrix_t>();
        return matrix.cols();
    } catch (const std::exception &e) {
        cerr << "Error loading matrix from " << file_name << ": " << e.what() << endl;
        return 0;
    }
}

bool save_matrix(const string &file_name, const matrix_t &matrix) {
    try {
        pybind11::module_ np = pybind11::module_::import("numpy");
        np.attr("save")(file_name, matrix);
        return true;
    } catch (const std::exception &e) {
        cerr << "Error saving matrix to " << file_name << ": " << e.what() << endl;
        return false;
    }
}

#endif // __OPTIMIZED_GTNN_HEADER
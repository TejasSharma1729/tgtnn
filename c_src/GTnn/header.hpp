#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
using namespace std;
using namespace std::chrono;
using namespace Eigen;

#define NUM_THREADS 16
namespace GTnn {
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix_t;
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
}

bool GTnn::check_file(ifstream &file) {
    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        return false;
    }
    return true;
}

bool GTnn::check_file(ofstream &file) {
    if (!file.is_open()) {
        cerr << "Error opening file" << endl;
        return false;
    }
    return true;
}

bool GTnn::check_file(FILE *file) {
    if (file == nullptr) {
        cerr << "Error opening file" << endl;
        return false;
    }
    return true;
}

bool GTnn::check_file(const string &file_name) {
    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return false;
    }
    return true;
}

bool GTnn::check_file(const char *file_name) {
    ifstream file(file_name);
    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return false;
    }
    return true;
}

string GTnn::path_append(const string &path, const string &file_name) {
    return path + "/" + file_name;
}

bool GTnn::recursive_mkdir(const string &path) {
    istringstream iss(path);
    string dir;
    while (getline(iss, dir, '/')) {
        if (dir.empty()) continue; // Skip empty segments
        if (mkdir(dir.c_str(), 0777) == -1 && errno != EEXIST) {
            cerr << "Error creating directory: " << dir << endl;
            return false;
        }
        chdir(dir.c_str());
    }
    return true;
}

size_t GTnn::extract_matrix(const string &file_name, matrix_t &matrix) {
    ifstream file(file_name.c_str());
    if (!check_file(file)) return 0;
    
    vector<vector<double>> data;
    size_t idx = 0;
    double value;
    string line, temp;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<double> row_data;
        while (getline(iss, temp, ',')) {
            value = stod(temp);
            row_data.push_back(value);
        }
        data.push_back(row_data);
    }

    matrix.resize(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].size(); ++j) {
            matrix(i, j) = data[i][j];
        }
    }
    file.close();
    return data[0].size(); // Return number of columns
}

bool GTnn::save_matrix(const string &file_name, const matrix_t &matrix) {
    ofstream file(file_name.c_str());
    if (!check_file(file)) return false;
    
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j) << " ";
        }
        file << endl;
    }
    file.close();
    return true;
}
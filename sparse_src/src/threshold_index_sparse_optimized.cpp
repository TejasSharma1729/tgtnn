
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <Eigen/Sparse>
#include "sparse_types.hpp"

using namespace std;
using namespace std::chrono;

// Use the same SparseMat type from sparse_types.hpp
template <size_t N = 0, size_t Q = 0>
class ThresholdIndexSparseOptimized {
public:
    ThresholdIndexSparseOptimized(std::string &__file_directory, double __threshold);
    virtual ~ThresholdIndexSparseOptimized(void) = default;
    std::pair<double, size_t> search(void);
    std::pair<double, size_t> naive_search(void);
    size_t get_dimention(void);
    size_t get_query_set_size(void);
    size_t get_data_set_size(void);
    double get_threshold(void);
    std::pair<double, double> get_precision_and_recall(void);

protected:
    SparseMat data_set;
    SparseMat query_set;
    SparseMat data_index;
    SparseMat query_index;
    size_t dimention;
    double threshold;
    size_t num_dot_products = 0;
    const std::string algo_name = "ThresholdIndexSparseOptimized";

    std::vector<std::vector<uint>> search_res;
    std::vector<std::vector<uint>> naive_res;
    std::random_device rd;
    std::mt19937 gen;
    uint left_data;
    uint right_data;
    uint left_query;
    uint right_query;
    void search_single_data(double dot_prod);
    void search_single_query(double dot_prod);
    void search_subspans(double dot_prod);
};

// Constructor implementation
template <size_t N, size_t Q>
ThresholdIndexSparseOptimized<N, Q>::ThresholdIndexSparseOptimized(std::string &__file_directory, double __threshold) {
    std::string data_file = __file_directory + "/X.txt";
    std::string query_file = __file_directory + "/Q.txt";
    auto [__data_set, __dim_1] = read_sparse_matrix(data_file);
    auto [__query_set, __dim_2] = read_sparse_matrix(query_file);
    if (__dim_1 != __dim_2) {
        throw std::invalid_argument("Data and query set have different dimention");
    } else {
        this->dimention = __dim_1;
        this->data_set = std::move(__data_set);
        this->query_set = std::move(__query_set);
    }
    this->threshold = __threshold;
    // Initialize data_index using build_index template
    const uint num_vectors = this->data_set.size();
    const uint log_size = std::ceil(std::log2(num_vectors));
    const uint upper_size = 1 << log_size;
    this->data_index.resize(upper_size);
    for (uint i = 0; i < num_vectors; i++) {
        this->data_index[i] = this->data_set[i];
    }
    for (uint n = 1; n <= log_size - N; n++) {
        const uint step = 1 << n;
        for (uint i = 0; i < upper_size; i += step) {
            this->data_index[i] = add_sparse(this->data_index[i], this->data_index[i + step / 2]);
        }
    }
    this->gen = std::mt19937(this->rd());
}

// Getter implementations
template <size_t N, size_t Q>
size_t ThresholdIndexSparseOptimized<N, Q>::get_dimention(void) {
    return this->dimention;
}

template <size_t N, size_t Q>
size_t ThresholdIndexSparseOptimized<N, Q>::get_data_set_size(void) {
    return this->data_set.size();
}

template <size_t N, size_t Q>
size_t ThresholdIndexSparseOptimized<N, Q>::get_query_set_size(void) {
    return this->query_set.size();
}

template <size_t N, size_t Q>
double ThresholdIndexSparseOptimized<N, Q>::get_threshold(void) {
    return this->threshold;
}

// Main search implementation
template <size_t N, size_t Q>
std::pair<double, size_t> ThresholdIndexSparseOptimized<N, Q>::search(void) {
    this->search_res.clear();
    this->search_res.resize(static_cast<uint>(this->query_set.size()), std::vector<uint>());
    std::pair<double, size_t> time_and_num_dots = std::make_pair(0.0, 0UL);
    this->num_dot_products = (1 << N) * (1 << Q);
    auto start = std::chrono::high_resolution_clock::now();
    
    // Build query index
    const uint num_queries = this->query_set.size();
    const uint query_log_size = std::ceil(std::log2(num_queries));
    const uint query_upper_size = 1 << query_log_size;
    this->query_index.resize(query_upper_size);
    for (uint i = 0; i < num_queries; i++) {
        this->query_index[i] = this->query_set[i];
    }
    for (uint n = 1; n <= query_log_size - Q; n++) {
        const uint step = 1 << n;
        for (uint i = 0; i < query_upper_size; i += step) {
            this->query_index[i] = add_sparse(this->query_index[i], this->query_index[i + step / 2]);
        }
    }
    
    const uint dstep = 1 << static_cast<uint>(std::ceil(std::log2(this->data_index.size())) - N);
    const uint qstep = 1 << static_cast<uint>(std::ceil(std::log2(this->query_index.size())) - Q);
    
    for (size_t i = 0; i < this->data_index.size(); i += dstep) {
        for (size_t j = 0; j < this->query_index.size(); j += qstep) {
            this->left_data = i;
            this->right_data = i + dstep;
            this->left_query = j;
            this->right_query = j + qstep;
            double dot_prod = dot_product(this->data_index[i], this->query_index[j]);
            if (dot_prod >= this->threshold) {
                this->search_subspans(dot_prod);
            }
        }
    }
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    time_and_num_dots.first += duration.count() / 1.0e+3;
    time_and_num_dots.second += this->num_dot_products;
    return time_and_num_dots;
}

// Naive search implementation
template <size_t N, size_t Q>
std::pair<double, size_t> ThresholdIndexSparseOptimized<N, Q>::naive_search(void) {
    this->naive_res.clear();
    this->naive_res.resize(static_cast<uint>(this->query_set.size()), std::vector<uint>());
    std::pair<double, size_t> time_and_num_dots = std::make_pair(0.0, 0UL);
    
    for (uint j = 0; j < this->query_set.size(); j++) {
        this->num_dot_products = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < this->data_set.size(); i++) {
            this->num_dot_products++;
            if (dot_product(this->data_set[i], this->query_set[j]) >= this->threshold) {
                this->naive_res[j].push_back(i);
            }
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        time_and_num_dots.first += duration.count() / 1.0e+3;
        time_and_num_dots.second += this->num_dot_products;
    }
    return time_and_num_dots;
}

// Precision and recall calculation
template <size_t N, size_t Q>
std::pair<double, double> ThresholdIndexSparseOptimized<N, Q>::get_precision_and_recall(void) {
    double mean_precision = 0;
    double mean_recall = 0;
    
    for (unsigned int i = 0; i < this->naive_res.size(); i++) {
        std::unordered_set<size_t> truth_set(this->naive_res[i].begin(), this->naive_res[i].end());
        size_t true_positives = 0;
        for (const auto& pred : this->search_res[i]) {
            if (truth_set.find(pred) != truth_set.end()) {
                true_positives++;
            }
        }
        mean_precision += (this->search_res[i].size() == 0) ? 1.0 :
            (static_cast<double>(true_positives) / this->search_res[i].size());
        mean_recall += (this->naive_res[i].size() == 0) ? 1.0 :
            (static_cast<double>(true_positives) / this->naive_res[i].size());
    }
    mean_precision /= this->search_res.size();
    mean_recall /= this->naive_res.size();
    return std::make_pair(mean_precision, mean_recall);
}

// Search single data implementation
template <size_t N, size_t Q>
void ThresholdIndexSparseOptimized<N, Q>::search_single_data(double dot_prod) {
    if (this->left_query + 1 == this->right_query) {
        if (dot_prod >= this->threshold) {
            this->search_res[this->left_query].push_back(this->left_data);
        }
        return;
    }
    const uint mid_query = (this->left_query + this->right_query) / 2;
    uint lq = this->left_query;
    uint rq = this->right_query;
    double right_dot_prod = dot_product(this->data_set[this->left_data], this->query_set[mid_query]);
    dot_prod -= right_dot_prod;
    this->num_dot_products++;
    if (dot_prod >= this->threshold) {
        this->right_query = mid_query;
        this->search_single_data(dot_prod);
        this->right_query = 2 * mid_query - this->left_query;
    }
    if (right_dot_prod >= this->threshold) {
        this->left_query = mid_query;
        this->search_single_data(right_dot_prod);
        this->left_query = 2 * mid_query - this->right_query;
    }
    if (lq != this->left_query || rq != this->right_query) {
        throw std::runtime_error("Indices not restored");
    }
}

// Search single query implementation
template <size_t N, size_t Q>
void ThresholdIndexSparseOptimized<N, Q>::search_single_query(double dot_prod) {
    if (this->left_data + 1 == this->right_data) {
        if (dot_prod >= this->threshold) {
            this->search_res[this->left_query].push_back(this->left_data);
        }
        return;
    }
    const uint mid_data = (this->left_data + this->right_data) / 2;
    uint ld = this->left_data;
    uint rd = this->right_data;
    double right_dot_prod = dot_product(this->data_index[mid_data], this->query_index[this->left_query]);
    dot_prod -= right_dot_prod;
    this->num_dot_products++;
    if (dot_prod >= this->threshold) {
        this->right_data = mid_data;
        this->search_single_query(dot_prod);
        this->right_data = 2 * mid_data - this->left_data;
    }
    if (right_dot_prod >= this->threshold) {
        this->left_data = mid_data;
        this->search_single_query(right_dot_prod);
        this->left_data = 2 * mid_data - this->right_data;
    }
    if (ld != this->left_data || rd != this->right_data) {
        throw std::runtime_error("Indices not restored");
    }
}

// Search subspans implementation
template <size_t N, size_t Q>
void ThresholdIndexSparseOptimized<N, Q>::search_subspans(double dot_prod) {
    if (this->left_data + 1 == this->right_data) {
        return search_single_data(dot_prod);
    }
    if (this->left_query + 1 == this->right_query) {
        return search_single_query(dot_prod);
    }
    if (this->gen() % 3) {
        const uint mid_query = (this->left_query + this->right_query) / 2;
        uint lq = this->left_query;
        uint rq = this->right_query;
        double right_dot_prod = dot_product(this->data_index[this->left_data], this->query_index[mid_query]);
        dot_prod -= right_dot_prod;
        this->num_dot_products++;
        if (dot_prod >= this->threshold) {
            this->right_query = mid_query;
            this->search_subspans(dot_prod);
            this->right_query = 2 * mid_query - this->left_query;
        }
        if (right_dot_prod >= this->threshold) {
            this->left_query = mid_query;
            this->search_subspans(right_dot_prod);
            this->left_query = 2 * mid_query - this->right_query;
        }
        if (lq != this->left_query || rq != this->right_query) {
            throw std::runtime_error("Indices not restored");
        }
    } 
    else {
        const uint mid_data = (this->left_data + this->right_data) / 2;
        uint ld = this->left_data;
        uint rd = this->right_data;
        double right_dot_prod = dot_product(this->data_index[mid_data], this->query_index[this->left_query]);
        dot_prod -= right_dot_prod;
        this->num_dot_products++;
        if (dot_prod >= this->threshold) {
            this->right_data = mid_data;
            this->search_subspans(dot_prod);
            this->right_data = 2 * mid_data - this->left_data;
        }
        if (right_dot_prod >= this->threshold) {
            this->left_data = mid_data;
            this->search_subspans(right_dot_prod);
            this->left_data = 2 * mid_data - this->right_data;
        }
        if (ld != this->left_data || rd != this->right_data) {
            throw std::runtime_error("Indices not restored");
        }
    }
}


#include <immintrin.h>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace std::chrono;

namespace GT {
    struct sparse_vec_t {
        vector<double> values;
        vector<size_t> indices;
    };
    struct tree_node_t {
        sparse_vec_t sum_vec;
        size_t index;
        unique_ptr<tree_node_t> left;
        unique_ptr<tree_node_t> right;
    };

    class GroupTestingNN {
    public:
        GroupTestingNN(string &__file_directory, double __threshold);
        virtual ~GroupTestingNN(void) = default;
        template <size_t N = 0> pair<double, size_t> search(void);
        template <size_t N = 0, size_t Q = 0> pair<double, size_t> double_search(void);
        pair<double, size_t> naive_search(void);
        size_t get_dimention(void);
        size_t get_query_set_size(void);
        size_t get_data_set_size(void);
        double get_threshold(void);
        pair<double, double> get_precision_and_recall(void);
        pair<double, double> get_double_precision_and_recall(void);

    protected:
        vector<sparse_vec_t> data_set;
        vector<sparse_vec_t> query_set;
        unique_ptr<tree_node_t> root;
        size_t dimention;
        double threshold;
        size_t query_index;
        size_t num_dot_products = 0;
        const string algo_name = "SparseGroupTesting";

        vector<vector<size_t>> search_res;
        vector<vector<size_t>> double_search_res;
        vector<vector<size_t>> naive_res;

        void search_subtree(tree_node_t *data_node);
        void search_single_data_point(tree_node_t *data_node, tree_node_t *query_node);
        void search_single_query_point(tree_node_t *data_node, tree_node_t *query_node);
        void search_double_subtree(tree_node_t *data_node, tree_node_t *query_node);
    };
    
    string path_append(const string& p1, const string& p2);
    void recursive_mkdir(const char *dir);
    void check_file(ofstream &file);
    pair<vector<sparse_vec_t>, size_t> read_sparse_matrix(const string &file_name);
    tree_node_t sort_and_build_index_tree(vector<sparse_vec_t> &matrix);
    tree_node_t build_index_tree(vector<sparse_vec_t> &matrix, size_t *__indices, size_t __size);
    template <size_t N> array<tree_node_t*, 1 << N> get_nodes_at_level(tree_node_t *root);
};

GT::sparse_vec_t operator + (const GT::sparse_vec_t &one, const GT::sparse_vec_t &two) {
    GT::sparse_vec_t result;
    size_t i = 0, j = 0;
    while (i < one.values.size() && j < two.values.size()) {
        if (one.indices[i] == two.indices[j]) {
            result.values.push_back(one.values[i] + two.values[j]);
            result.indices.push_back(one.indices[i]);
            i++;
            j++;
        } else if (one.indices[i] < two.indices[j]) {
            result.values.push_back(one.values[i]);
            result.indices.push_back(one.indices[i]);
            i++;
        } else {
            result.values.push_back(two.values[j]);
            result.indices.push_back(two.indices[j]);
            j++;
        }
    }
    while (i < one.values.size()) {
        result.values.push_back(one.values[i]);
        result.indices.push_back(one.indices[i]);
        i++;
    }
    while (j < two.values.size()) {
        result.values.push_back(two.values[j]);
        result.indices.push_back(two.indices[j]);
        j++;
    }
    return result;
}

double operator * (GT::sparse_vec_t &one, GT::sparse_vec_t &two) {
    double result = 0;
    size_t i = 0, j = 0;
    while (i < one.values.size() && j < two.values.size()) {
        if (one.indices[i] == two.indices[j]) {
            result += one.values[i] * two.values[j];
            i++;
            j++;
        } else if (one.indices[i] < two.indices[j]) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}

bool operator < (const GT::sparse_vec_t &one, const GT::sparse_vec_t &two) {
    double avg_idx1 = 0.0;
    double avg_idx2 = 0.0;
    for (size_t i = 0; i < one.indices.size(); i++) {
        avg_idx1 += one.indices[i] * one.values[i] * one.values[i];
    }
    for (size_t i = 0; i < two.indices.size(); i++) {
        avg_idx2 += two.indices[i] * two.values[i] * two.values[i];
    }
    return avg_idx1 < avg_idx2;
}

GT::GroupTestingNN::GroupTestingNN(string &__file_directory, double __threshold) {
    string data_file = GT::path_append(__file_directory, "X.txt");
    string query_file = GT::path_append(__file_directory, "Q.txt");
    auto [__data_set, __dim_1] = GT::read_sparse_matrix(data_file);
    auto [__query_set, __dim_2] = GT::read_sparse_matrix(query_file);
    if (__dim_1 != __dim_2) {
        throw invalid_argument("Data and query set have different dimention");
    }
    this->threshold = __threshold;
    this->dimention = __dim_1;
    this->data_set = move(__data_set);
    this->query_set = move(__query_set);
    this->root = make_unique<GT::tree_node_t>(GT::sort_and_build_index_tree(this->data_set));

}

size_t GT::GroupTestingNN::get_dimention(void) {
    return this->dimention;
}

size_t GT::GroupTestingNN::get_data_set_size(void) {
    return this->data_set.size();
}

size_t GT::GroupTestingNN::get_query_set_size(void) {
    return this->query_set.size();
}

double GT::GroupTestingNN::get_threshold(void) {
    return this->threshold;
}

template <size_t N = 0> pair<double, size_t> GT::GroupTestingNN::search(void) {
    this->search_res.clear();
    this->search_res.resize(this->query_set.size(), vector<size_t>());
    pair<double, size_t> time_and_num_dots = make_pair(0.0, 0UL);
    array<GT::tree_node_t*, 1 << N> nodes = GT::get_nodes_at_level<N>(this->root.get());
    for (this->query_index = 0; this->query_index < this->query_set.size(); this->query_index++) {
        // cerr << "Query index: " << this->query_index << endl;
        this->num_dot_products = 0;
        auto start = high_resolution_clock::now();
        for (size_t i = 0; i < (1 << N); i++) {
            this->search_subtree(nodes[i]);
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        time_and_num_dots.first += duration.count() / 1.0e+3;
        time_and_num_dots.second += this->num_dot_products;
    }
    return time_and_num_dots;
}

template <size_t N = 0, size_t Q = 0> pair<double, size_t> GT::GroupTestingNN::double_search(void) {
    this->double_search_res.clear();
    this->double_search_res.resize(this->query_set.size(), vector<size_t>());
    pair<double, size_t> time_and_num_dots = make_pair(0.0, 0UL);
    this->num_dot_products = 0;
    array<GT::tree_node_t*, 1 << N> nodes = GT::get_nodes_at_level<N>(this->root.get());
    auto start = high_resolution_clock::now();
    unique_ptr<GT::tree_node_t> query_root = make_unique<GT::tree_node_t>(
            GT::sort_and_build_index_tree(this->query_set));
    array<GT::tree_node_t*, 1 << Q> query_nodes = GT::get_nodes_at_level<Q>(query_root.get());
    for (size_t i = 0; i < (1 << N); i++) {
        for (size_t j = 0; j < (1 << Q); j++) {
            this->search_double_subtree(nodes[i], query_nodes[j]);
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    time_and_num_dots.first += duration.count() / 1.0e+3;
    time_and_num_dots.second += this->num_dot_products;
    return time_and_num_dots;
}

pair<double, size_t> GT::GroupTestingNN::naive_search(void) {
    this->naive_res.clear();
    this->naive_res.resize(this->query_set.size(), vector<size_t>());
    pair<double, size_t> time_and_num_dots = make_pair(0.0, 0UL);
    for (this->query_index = 0; this->query_index < this->query_set.size(); this->query_index++) {
        // cerr << "Naive query index: " << this->query_index << endl;
        this->num_dot_products = 0;
        auto start = high_resolution_clock::now();
        for (size_t i = 0; i < this->data_set.size(); i++) {
            this->num_dot_products++;
            if (this->data_set[i] * this->query_set[this->query_index] >= this->threshold) {
                this->naive_res[this->query_index].push_back(i);
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        time_and_num_dots.first += duration.count() / 1.0e+3;
        time_and_num_dots.second += this->num_dot_products;
    }
    return time_and_num_dots;
}

pair<double, double> GT::GroupTestingNN::get_precision_and_recall(void) {
    double mean_precision = 0;
    double mean_recall = 0;
    for (unsigned int i = 0; i < this->naive_res.size(); i++) {
        unordered_set<size_t> truth_set(this->naive_res[i].begin(), this->naive_res[i].end());
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
    return make_pair(mean_precision, mean_recall);
}

pair<double, double> GT::GroupTestingNN::get_double_precision_and_recall(void) {
    double mean_precision = 0;
    double mean_recall = 0;
    for (unsigned int i = 0; i < this->naive_res.size(); i++) {
        unordered_set<size_t> truth_set(this->naive_res[i].begin(), this->naive_res[i].end());
        size_t true_positives = 0;
        for (const auto& pred : this->double_search_res[i]) {
            if (truth_set.find(pred) != truth_set.end()) {
                true_positives++;
            }
        }
        mean_precision += (this->double_search_res[i].size() == 0) ? 1.0 :
            (static_cast<double>(true_positives) / this->double_search_res[i].size());
        mean_recall += (this->naive_res[i].size() == 0) ? 1.0 :
            (static_cast<double>(true_positives) / this->naive_res[i].size());
    }
    mean_precision /= this->double_search_res.size();
    mean_recall /= this->naive_res.size();
    return make_pair(mean_precision, mean_recall);
}

void GT::GroupTestingNN::search_subtree(GT::tree_node_t *data_node) {
    this->num_dot_products++;
    if (data_node->sum_vec * this->query_set[this->query_index] < threshold) {
        return;
    }
    if (data_node->left == nullptr) {
        this->search_res[this->query_index].push_back(data_node->index);
        return;
    }
    this->search_subtree(data_node->left.get());
    this->search_subtree(data_node->right.get());
}

void GT::GroupTestingNN::search_single_data_point(GT::tree_node_t *data_node, GT::tree_node_t *query_node) {
    this->num_dot_products++;
    if (data_node->sum_vec * query_node->sum_vec < threshold) {
        return;
    }
    if (query_node->left == nullptr) {
        this->double_search_res[query_node->index].push_back(data_node->index);
        return;
    }
    this->search_single_data_point(data_node, query_node->left.get());
    this->search_single_data_point(data_node, query_node->right.get());
}

void GT::GroupTestingNN::search_single_query_point(GT::tree_node_t *data_node, GT::tree_node_t *query_node) {
    this->num_dot_products++;
    if (data_node->sum_vec * query_node->sum_vec < threshold) {
        return;
    }
    if (data_node->left == nullptr) {
        this->double_search_res[query_node->index].push_back(data_node->index);
        return;
    }
    this->search_single_query_point(data_node->left.get(), query_node);
    this->search_single_query_point(data_node->right.get(), query_node);
}

void GT::GroupTestingNN::search_double_subtree(GT::tree_node_t *data_node, GT::tree_node_t *query_node) {
    if (data_node->left == nullptr) {
        search_single_data_point(data_node, query_node);
        return;
    }
    if (query_node->left == nullptr) {
        search_single_query_point(data_node, query_node);
        return;
    }
    this->num_dot_products++;
    if (data_node->sum_vec * query_node->sum_vec < threshold) {
        return;
    }
    this->search_double_subtree(data_node->left.get(), query_node->left.get());
    this->search_double_subtree(data_node->left.get(), query_node->right.get());
    this->search_double_subtree(data_node->right.get(), query_node->left.get());
    this->search_double_subtree(data_node->right.get(), query_node->right.get());
}

string GT::path_append(const string& p1, const string& p2) {
    char sep = '/';
    std::string tmp = p1;

    #ifdef _WIN32
        sep = '\\';
    #endif

    if (p1[p1.length()] != sep) { 
        tmp += sep;
        return(tmp + p2);
    }
    else {
        return(p1 + p2);
    }
}

void GT::recursive_mkdir(const char *dir) {
    char tmp[256];
    char *p = NULL;
    unsigned int len;
    snprintf(tmp, sizeof(tmp),"%s",dir);
    len = strlen(tmp);
    if (tmp[len - 1] == '/')
        tmp[len - 1] = 0;
    for (p = tmp + 1; *p; p++)
        if (*p == '/') {
            *p = 0;
            mkdir(tmp, S_IRWXU);
            *p = '/';
        }
    mkdir(tmp, S_IRWXU);
}

void GT::check_file(ofstream &file) {
    if(file.fail()) {
        throw invalid_argument("Failed to open file");
    }
}

pair<vector<GT::sparse_vec_t>, size_t> GT::read_sparse_matrix(const string &file_name) {
    size_t num_vectors, dimention, num_non_zero;
    ifstream file(file_name);
    file >> num_vectors >> dimention >> num_non_zero;
    pair<vector<GT::sparse_vec_t>, size_t> result;
    result.second = dimention;
    result.first.resize(num_vectors);
    vector<double> __values(num_non_zero);
    vector<size_t> __indices(num_non_zero);
    vector<size_t> indptr(num_vectors);
    for (size_t i = 0; i < num_non_zero; i++) {
        file >> __values[i];
    }
    for (size_t i = 0; i < num_non_zero; i++) {
        file >> __indices[i];
    }
    for (size_t i = 0; i < num_vectors; i++) {
        file >> indptr[i];
    }
    size_t j = 0;
    for (size_t i = 0; i < num_non_zero; i++) {
        if (j <= num_vectors - 1 && i == indptr[j]) {
            j++;
        }
        result.first[j - 1].values.push_back(__values[i]);
        result.first[j - 1].indices.push_back(__indices[i]);
    }
    for (size_t i = 0; i < num_vectors; i++) {
        double norm = 0;
        for (size_t j = 0; j < result.first[i].values.size(); j++) {
            if (result.first[i].values[j] < 0) {
                throw invalid_argument("Negative value in the matrix");
            }
            norm += result.first[i].values[j] * result.first[i].values[j];
        }
        if (norm == 0) {
            continue;
            throw invalid_argument("Zero norm vector in the matrix");
        }
        norm = sqrt(norm);
        for (size_t j = 0; j < result.first[i].values.size(); j++) {
            result.first[i].values[j] /= norm;
        }
    }
    return result;
}

GT::tree_node_t GT::sort_and_build_index_tree(vector<GT::sparse_vec_t> &matrix) {
    vector<size_t> indices(matrix.size());
    iota(indices.begin(), indices.end(), 0);
    sort(indices.begin(), indices.end(), [&matrix](size_t i, size_t j) {
        return matrix[i] < matrix[j];
    });
    return GT::build_index_tree(matrix, indices.data(), matrix.size());
}

GT::tree_node_t GT::build_index_tree(vector<GT::sparse_vec_t> &matrix, size_t *__indices, size_t __size) {
    if (__size == 0) {
        throw invalid_argument("Size of the matrix is 0");
    }
    GT::tree_node_t node;
    if (__size == 1) {
        node.sum_vec = matrix[*__indices];
        node.index = *__indices;
        return node;
    }
    size_t mid = __size / 2;
    node.left = make_unique<GT::tree_node_t>(GT::build_index_tree(matrix, __indices, mid));
    node.right = make_unique<GT::tree_node_t>(GT::build_index_tree(matrix, __indices + mid, __size - mid));
    node.sum_vec = node.left->sum_vec + node.right->sum_vec;
    node.index = node.left->index;
    return node;
}

template <size_t N>
array<GT::tree_node_t*, 1 << N> GT::get_nodes_at_level(tree_node_t *root) {
    array<GT::tree_node_t*, 1 << N> nodes;
    if (N == 0) {
        nodes[0] = root;
        return nodes;
    }
    for (size_t i = 0; i < (1 << N); i++) {
        nodes[i] = root;
        for (size_t j = 0; j < N; j++) {
            if (nodes[i] == nullptr) {
                throw invalid_argument("Node is nullptr");
            }
            if (i & (1 << j)) {
                nodes[i] = nodes[i]->right.get();
            } else {
                nodes[i] = nodes[i]->left.get();
            }
        }
    }
    return nodes;
}
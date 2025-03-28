#include <immintrin.h>
#include <bits/stdc++.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <Eigen/Sparse>

using namespace std;
using namespace std::chrono;

namespace GT {
    typedef vector<pair<double, uint>> sparse_vec_t;
    typedef vector<sparse_vec_t> sparse_mat_t;

    template <size_t N = 0, size_t Q = 0>
    class GroupTestingNN {
    public:
        GroupTestingNN(string &__file_directory, double __threshold);
        virtual ~GroupTestingNN(void) = default;
        pair<double, size_t> search(void);
        pair<double, size_t> naive_search(void);
        size_t get_dimention(void);
        size_t get_query_set_size(void);
        size_t get_data_set_size(void);
        double get_threshold(void);
        pair<double, double> get_precision_and_recall(void);

    protected:
        sparse_mat_t data_set;
        sparse_mat_t query_set;
        sparse_mat_t data_index;
        sparse_mat_t query_index;
        size_t dimention;
        double threshold;
        size_t num_dot_products = 0;
        const string algo_name = "SparseGroupTesting";

        vector<vector<uint>> search_res;
        vector<vector<uint>> naive_res;
        random_device rd;
        mt19937 gen;
        uint left_data;
        uint right_data;
        uint left_query;
        uint right_query;
        void search_single_data(double dot_product);
        void search_single_query(double dot_product);
        void search_subspans(double dot_product);
    };
    
    string path_append(const string& p1, const string& p2);
    void recursive_mkdir(const char *dir);
    void check_file(ofstream &file);
    pair<sparse_mat_t, uint> read_sparse_matrix(const string &file_name);
    template <size_t N> sparse_mat_t build_index(sparse_mat_t &matrix);
};

GT::sparse_vec_t operator + (const GT::sparse_vec_t &one, const GT::sparse_vec_t &two) {
    GT::sparse_vec_t result;
    size_t i = 0, j = 0;
    while (i < one.size() && j < two.size()) {
        if (one[i].second == two[j].second) {
            result.push_back(make_pair(one[i].first + two[j].first, one[i].second));
            i++;
            j++;
        } else if (one[i].second < two[j].second) {
            result.push_back(one[i]);
            i++;
        } else {
            result.push_back(two[j]);
            j++;
        }
    }
    while (i < one.size()) {
        result.push_back(one[i]);
        i++;
    }
    while (j < two.size()) {
        result.push_back(two[j]);
        j++;
    }
    return result;
}

double operator * (const GT::sparse_vec_t &one, const GT::sparse_vec_t &two) {
    double result = 0;
    size_t i = 0, j = 0;
    while (i < one.size() && j < two.size()) {
        if (one[i].second == two[j].second) {
            result += one[i].first * two[j].first;
            i++;
            j++;
        } else if (one[i].second < two[j].second) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}

template <size_t N, size_t Q>
GT::GroupTestingNN<N, Q>::GroupTestingNN(string &__file_directory, double __threshold) {
    string data_file = GT::path_append(__file_directory, "X.txt");
    string query_file = GT::path_append(__file_directory, "Q.txt");
    auto [__data_set, __dim_1] = GT::read_sparse_matrix(data_file);
    auto [__query_set, __dim_2] = GT::read_sparse_matrix(query_file);
    if (__dim_1 != __dim_2) {
        throw invalid_argument("Data and query set have different dimention");
    } else {
        this->dimention = __dim_1;
        this->data_set = std::move(__data_set);
        this->query_set = std::move(__query_set);
    }
    this->threshold = __threshold;
    this->data_index = GT::build_index<N>(this->data_set);
    this->gen = mt19937(this->rd());
}

template <size_t N, size_t Q>
size_t GT::GroupTestingNN<N, Q>::get_dimention(void) {
    return this->dimention;
}

template <size_t N, size_t Q>
size_t GT::GroupTestingNN<N, Q>::get_data_set_size(void) {
    return this->data_set.size();
}

template <size_t N, size_t Q>
size_t GT::GroupTestingNN<N, Q>::get_query_set_size(void) {
    return this->query_set.size();
}

template <size_t N, size_t Q>
double GT::GroupTestingNN<N, Q>::get_threshold(void) {
    return this->threshold;
}

template <size_t N, size_t Q> 
pair<double, size_t> GT::GroupTestingNN<N, Q>::search(void) {
    this->search_res.clear();
    this->search_res.resize(static_cast<uint>(this->query_set.size()), vector<uint>());
    pair<double, size_t> time_and_num_dots = make_pair(0.0, 0UL);
    this->num_dot_products = (1 << N) * (1 << Q);
    auto start = high_resolution_clock::now();
    this->query_index = GT::build_index<Q>(this->query_set);
    const uint dstep = 1 << static_cast<uint>(ceil(log2(this->data_index.size())) - N);
    const uint qstep = 1 << static_cast<uint>(ceil(log2(this->query_index.size())) - Q);
    for (size_t i = 0; i < this->data_index.size(); i += dstep) {
        for (size_t j = 0; j < this->query_index.size(); j += qstep) {
            this->left_data = i;
            this->right_data = i + dstep;
            this->left_query = j;
            this->right_query = j + qstep;
            cerr << this->left_data << " " << this->right_data << " " << this->left_query << " " << this->right_query << endl;
            double dot_product = this->data_index[i] * this->query_index[j];
            if (dot_product >= this->threshold) {
                this->search_subspans(dot_product);
            }
        }
    }
    auto stop = high_resolution_clock::now();
    cerr << "Finished search with num_dot_products = " << this->num_dot_products << endl;
    auto duration = duration_cast<microseconds>(stop - start);
    time_and_num_dots.first += duration.count() / 1.0e+3;
    time_and_num_dots.second += this->num_dot_products;
    return time_and_num_dots;
}

template <size_t N, size_t Q>
pair<double, size_t> GT::GroupTestingNN<N, Q>::naive_search(void) {
    this->naive_res.clear();
    this->naive_res.resize(static_cast<uint>(this->query_set.size()), vector<uint>());
    pair<double, size_t> time_and_num_dots = make_pair(0.0, 0UL);
    for (uint j = 0; j < this->query_set.size(); j++) {
        // cerr << "Naive query index: " << this->query_index << endl;
        this->num_dot_products = 0;
        auto start = high_resolution_clock::now();
        for (size_t i = 0; i < this->data_set.size(); i++) {
            this->num_dot_products++;
            if (this->data_set[i] * this->query_set[j] >= this->threshold) {
                this->naive_res[j].push_back(i);
            }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        time_and_num_dots.first += duration.count() / 1.0e+3;
        time_and_num_dots.second += this->num_dot_products;
    }
    return time_and_num_dots;
}

template <size_t N, size_t Q>
pair<double, double> GT::GroupTestingNN<N, Q>::get_precision_and_recall(void) {
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

template <size_t N, size_t Q>
void GT::GroupTestingNN<N, Q>::search_single_data(double dot_product) {
    if (this->left_query + 1 == this->right_query) {
        if (dot_product >= threshold) {
            this->search_res[this->left_query].push_back(this->left_data);
        }
        return;
    }
    const uint mid_query = (this->left_query + this->right_query) / 2;
    uint lq = this->left_query;
    uint rq = this->right_query;
    double right_dot_product = this->data_set[this->left_data] * this->query_set[mid_query];
    dot_product -= right_dot_product;
    this->num_dot_products++;
    if (dot_product >= threshold) {
        this->right_query = mid_query;
        this->search_single_data(dot_product);
        this->right_query = 2 * mid_query - this->left_query;
    }
    if (right_dot_product >= threshold) {
        this->left_query = mid_query;
        this->search_single_data(right_dot_product);
        this->left_query = 2 * mid_query - this->right_query;
    }
    if (left_query == right_query) {
        throw runtime_error("Single query");         
    }
    if (lq != this->left_query || rq != this->right_query) {
        throw runtime_error("Indices not restored");
    }
}

template <size_t N, size_t Q>
void GT::GroupTestingNN<N, Q>::search_single_query(double dot_product) {
    if (this->left_data + 1 == this->right_data) {
        if (dot_product >= threshold) {
            this->search_res[this->left_query].push_back(this->left_data);
        }
        return;
    }
    const uint mid_data = (this->left_data + this->right_data) / 2;
    uint ld = this->left_data;
    uint rd = this->right_data;
    cerr << ld << " " << rd << " " << this->left_query << " " << this->right_query << endl;
    if (ld == 999424 && rd == 1015808) {
        cerr 
    }
    double right_dot_product = this->data_index[mid_data] * this->query_index[this->left_query];
    dot_product -= right_dot_product;
    this->num_dot_products++;
    if (dot_product >= threshold) {
        this->right_data = mid_data;
        this->search_single_query(dot_product);
        this->right_data = 2 * mid_data - this->left_data;
    }
    if (right_dot_product >= threshold) {
        this->left_data = mid_data;
        this->search_single_query(right_dot_product);
        this->left_data = 2 * mid_data - this->right_data;
    }
    if (left_data == right_data) {
        throw runtime_error("Single data");         
    }
    if (ld != this->left_data || rd != this->right_data) {
        throw runtime_error("Indices not restored");
    }
}

template <size_t N, size_t Q>
void GT::GroupTestingNN<N, Q>::search_subspans(double dot_product) {
    if (this->left_data + 1 == this->right_data) {
        return search_single_data(dot_product);
    }
    if (this->left_query + 1 == this->right_query) {
        return search_single_query(dot_product);
    }
    if (this->gen() % 3) {
        const uint mid_query = (this->left_query + this->right_query) / 2;
        uint lq = this->left_query;
        uint rq = this->right_query;
        double right_dot_product = this->data_index[this->left_data] * this->query_index[mid_query];
        dot_product -= right_dot_product;
        this->num_dot_products++;
        if (dot_product >= threshold) {
            this->right_query = mid_query;
            this->search_subspans(dot_product);
            this->right_query = 2 * mid_query - this->left_query;
        }
        if (right_dot_product >= threshold) {
            this->left_query = mid_query;
            this->search_subspans(right_dot_product);
            this->left_query = 2 * mid_query - this->right_query;
        }
        if (left_query == right_query) {
            throw runtime_error("Single query");         
        }
        if (lq != this->left_query || rq != this->right_query) {
            throw runtime_error("Indices not restored");
        }
    } 
    else {
        const uint mid_data = (this->left_data + this->right_data) / 2;
        uint ld = this->left_data;
        uint rd = this->right_data;
        double right_dot_product = this->data_index[mid_data] * this->query_index[this->left_query];
        dot_product -= right_dot_product;
        this->num_dot_products++;
        if (dot_product >= threshold) {
            this->right_data = mid_data;
            this->search_subspans(dot_product);
            this->right_data = 2 * mid_data - this->left_data;
        }
        if (right_dot_product >= threshold) {
            this->left_data = mid_data;
            this->search_subspans(right_dot_product);
            this->left_data = 2 * mid_data - this->right_data;
        }
        if (left_data == right_data) {
            throw runtime_error("Single data");         
        }
        if (ld != this->left_data || rd != this->right_data) {
            throw runtime_error("Indices not restored");
        }
    }
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

pair<GT::sparse_mat_t, uint> GT::read_sparse_matrix(const string &file_name) {
    size_t num_vectors, dimention, num_non_zero;
    ifstream file(file_name);
    file >> num_vectors >> dimention >> num_non_zero;
    vector<double> __values(num_non_zero);
    vector<uint> __indices(num_non_zero);
    vector<size_t> indptr(num_vectors);
    for (size_t i = 0; i < num_non_zero; i++) {
        file >> __values[i];
        if (__values[i] < 0) {
            throw invalid_argument("Negative value in the matrix");
        }
    }
    for (size_t i = 0; i < num_non_zero; i++) {
        file >> __indices[i];
    }
    for (uint i = 0; i < num_vectors; i++) {
        file >> indptr[i];
    }
    uint j = 0;
    pair<GT::sparse_mat_t, uint> result;
    result.first.resize(num_vectors);
    result.second = dimention;
    for (size_t i = 0; i < num_non_zero; i++) {
        if (j <= num_vectors - 1 && i == indptr[j]) {
            j++;
        }
        result.first[j - 1].push_back(make_pair(__values[i], __indices[i]));
        // cout << i << endl;
    }
    for (size_t i = 0; i < num_vectors; i++) {
        double norm = 0;
        for (const auto& elem : result.first[i]) {
            norm += elem.first * elem.first;
        }
        norm = sqrt(norm);
        if (norm == 0) {
            continue;
            throw invalid_argument("Zero norm vector in the matrix");
        }
        for (auto& elem : result.first[i]) {
            elem.first /= norm;
        }
    }
    return result;
}

template <size_t N>
GT::sparse_mat_t GT::build_index(sparse_mat_t &matrix) {
    const uint num_vectors = matrix.size();
    const uint log_size = ceil(log2(num_vectors));
    const uint upper_size = 1 << log_size;
    sparse_mat_t data_index(upper_size);
    for (uint i = 0; i < num_vectors; i++) {
        data_index[i] = matrix[i];
    }
    for (uint n = 1; n <= log_size - N; n++) {
        const uint step = 1 << n;
        for (uint i = 0; i < upper_size; i += step) {
            data_index[i] = data_index[i] + data_index[i + step / 2];
        }
    }
    return data_index;
}

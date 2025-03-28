#include "GTnn/SPARSE_GTNN_SUM_CLASSWISE_SMART.hpp"
#include <unistd.h>
#include <sys/wait.h>
using namespace std;

vector<vector<pair<double, size_t>>> all_queries(vector<GT::sparse_vec_t> &data_set,
        vector<GT::sparse_vec_t> &query_set, uint dimention) {
    vector<vector<pair<double, size_t>>> scores(query_set.size());
    cerr << "Start..." << endl;
    for (uint i = 0; i < query_set.size(); i++) {
        for (size_t j = 0; j < data_set.size(); j++) {
            double dot_product = data_set[j] * query_set[i];
            if (dot_product >= 0.5) {
                scores[i].push_back(make_pair(dot_product, j));
            }
        }
        cerr << "Sorting: ";
        sort(scores[i].begin(), scores[i].end(), [](pair<double, size_t> &a, pair<double, size_t> &b) {
            return a.first > b.first;
        });
        cerr << i << endl;
    }
    return scores;
}

vector<vector<pair<double, size_t>>> all_queries(vector<GT::sparse_vec_t> &data_set,
        vector<GT::sparse_vec_t> &query_set, vector<vector<size_t>> &search_res) {
    vector<vector<pair<double, size_t>>> scores(query_set.size());
    for (uint i = 0; i < query_set.size(); i++) {
        for (size_t j : search_res[i]) {
            double dot_product = data_set[j] * query_set[i];
            if (dot_product >= 0.5) {
                scores[i].push_back(make_pair(dot_product, j));
            }
        }
        sort(scores[i].begin(), scores[i].end(), [](pair<double, size_t> &a, pair<double, size_t> &b) {
            return a.first > b.first;
        });
        cerr << i << endl;
    }
    return scores;
}

void save_file(vector<vector<pair<double, size_t>>> &scores, string file_name) {
    ofstream file(file_name);
    file << scores.size() << endl;
    for (uint i = 0; i < scores.size(); i++) {
        file << scores[i].size() << endl;
        for (uint j = 0; j < scores[i].size(); j++) {
            file << scores[i][j].first << " " << scores[i][j].second << endl;
        }
    }
    return file.close();
}

vector<vector<pair<double, size_t>>> load_file(string file_name) {
    ifstream file(file_name);
    uint num_queries;
    file >> num_queries;
    vector<vector<pair<double, size_t>>> scores(num_queries);
    for (uint i = 0; i < num_queries; i++) {
        uint num_data_points;
        file >> num_data_points;
        scores[i].resize(num_data_points);
        for (uint j = 0; j < num_data_points; j++) {
            file >> scores[i][j].first >> scores[i][j].second;
        }
    }
    return scores;
}

double recall(array<pair<double, size_t>, 10> &scores, vector<pair<double, size_t>> &actual) {
    int val = actual.size() - 1;
    if (val == 0) {
        return 1.0; // our algo did not detect anything
    }
    while (val >= 0) {
        if (actual[val].second == scores[0].second) {
            break;
        }
        val--;
    }
    return 20.0 * (val + 1) / (100.0 + (val + 1) * (val + 1));
}

vector<double> compare_with_pysparnn(const string &pysparnn_out, vector<vector<pair<double, size_t>>> &actual,
        vector<GT::sparse_vec_t> &data_set, vector<GT::sparse_vec_t> &query_set) {
    ifstream file(pysparnn_out);
    const uint num_queries = query_set.size();
    const uint num_data_points = data_set.size();
    vector<double> precicion_recall;
    for (uint i = 0; i < num_queries; i++) {
        array<pair<double, size_t>, 10> scores;
        string line0, line1;
        for (uint j = 0; j < 10; j++) {
            file >> line0 >> line1;
            scores[j].first = stod(line0.substr(2, line0.size() - 3));
            scores[j].second = stoul(line1.substr(5, line1.size() - 7));
        }
        double pr = recall(scores, actual[i]);
        cerr << pr << endl;
        precicion_recall.push_back(pr);
    }       
    return precicion_recall;
}

int main(int argc, char *argv[]) {
    string dir = "sparse_dataset";
    GT::GroupTestingNN gt_nn(dir, 0.5);
    auto &data_set = gt_nn.data_set;
    auto &query_set = gt_nn.query_set;
    // gt_nn.double_search<12, 5>();
    // vector<vector<pair<double, size_t>>> actual = all_queries(data_set, query_set, gt_nn.double_search_res);
    // save_file(actual, "sparse_dataset/actual.txt");
    vector<vector<pair<double, size_t>>> actual = load_file("sparse_dataset/actual.txt");
    cerr << "Start" << endl;
    vector<double> precicion_recall = compare_with_pysparnn("sparse_dataset/pysparnn.txt", actual, data_set, query_set);
    /*for (uint i = 0; i < precicion_recall.size(); i++) {
        cout << precicion_recall[i] << endl;
    }*/
    return 0;
}
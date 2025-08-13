#include "GTnn/optimized_single.hpp"
#include "GTnn/optimized_double.hpp"
#include "GTnn/knns.hpp"
#include "GTnn/knns_double.hpp"
#include <iomanip>
#define K_VAL 10
#define THRESHOLD 0.8
#define NUM_RUNS 5

// Function to calculate mean and standard deviation
pair<double, double> calculate_stats(const vector<double>& values) {
    double mean = 0.0;
    for (double val : values) {
        mean += val;
    }
    mean /= values.size();
    
    double variance = 0.0;
    for (double val : values) {
        variance += (val - mean) * (val - mean);
    }
    variance /= values.size();
    double std_dev = sqrt(variance);
    
    return make_pair(mean, std_dev);
}

// Calculate precision and recall
pair<double, double> calculate_precision_recall(
    const vector<uint>& result, 
    const vector<uint>& ground_truth_threshold,
    const array<uint, K_VAL>& ground_truth_knns, 
    bool is_it_knns
) {
    set<uint> result_set(result.begin(), result.end());
    set<uint> truth_set;
    if (is_it_knns) {
        truth_set = set<uint>(
            ground_truth_knns.begin(), 
            ground_truth_knns.end()
        );
    }
    else {
        truth_set = set<uint>(
            ground_truth_threshold.begin(), 
            ground_truth_threshold.end()
        );
    }
    
    // Count intersection
    size_t intersection = 0;
    for (uint item : result_set) {
        if (truth_set.count(item)) {
            intersection++;
        }
    }
    
    double precision = result_set.empty() 
        ? 1.0 
        : (double)intersection / result_set.size();
    double recall = truth_set.empty() 
        ? 1.0 
        : (double)intersection / truth_set.size();
    return make_pair(precision, recall);
}

// Run naive search ONCE to get ground truth
map<long, pair<vector<uint>, array<uint, K_VAL>>> run_naive_search_once(
    GTnn::matrix_t& query_set, 
    GTnn::matrix_t& data_set, 
    double& naive_time
) {
    map<long, pair<vector<uint>, array<uint, K_VAL>>> ground_truth;
    
    cout << "Running naive search ONCE to get ground truth..." << endl;
    auto start = high_resolution_clock::now();
    
    for (Eigen::Index i = 0; i < query_set.rows(); i++) {
        GTnn::vector_t query = query_set.row(i);
        
        // Compute all dot products manually (naive search)
        vector<pair<double, uint>> scores;
        vector<uint> above_tr;
        for (Eigen::Index j = 0; j < data_set.rows(); j++) {
            double score = query.dot(data_set.row(j));
            scores.push_back(make_pair(score, j));
            // Use a small epsilon to handle floating point precision issues
            if (score >= THRESHOLD) {
                above_tr.push_back(j);
            }
        }
        
        // Sort by score (descending) and take top k
        sort(scores.begin(), scores.end(), greater<pair<double, uint>>());
        
        array<uint, K_VAL> top_k;
        for (size_t idx = 0; idx < K_VAL; idx++) {
            top_k[idx] = scores[idx].second;
        }
        sort(top_k.begin(), top_k.end());
        ground_truth[i] = {above_tr, top_k};
    }
    
    auto stop = high_resolution_clock::now();
    naive_time = duration_cast<microseconds>(stop - start).count() / 1.0e+3;
    naive_time /= query_set.rows(); // Average per query
    
    cout << "Naive search completed. Avg time per query: " << 
        naive_time << " ms" << endl;
    return ground_truth;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dataset_path>" << endl;
        return 1;
    }

    GTnn::matrix_t query_set;
    if (!GTnn::extract_matrix(GTnn::path_append(argv[1], "Q.txt"), query_set)) {
        cerr << "Error extracting query matrix." << endl;
        return 1;
    }
    
    cout << "Original query matrix dimensions: " << query_set.rows() << " x " << query_set.cols() << endl;

    cout << "=== COMPREHENSIVE BENCHMARKING ===" << endl;
    cout << "Dataset: " << argv[1] << endl;
    cout << "Number of queries: " << query_set.rows() << endl << endl;

    struct MethodResult {
        string name;
        double mean_time;
        double stddev_time;
        double mean_precision;
        double stddev_precision;
        double mean_recall;
        double stddev_recall;
        double speedup;
    };
    vector<MethodResult> all_results;
    map<string, map<long, vector<uint>>> all_final_results;

    // --- 1. OPTIMIZED SINGLE ---
    cout << "Starting OptimizedSingle tests..." << endl;
    {
        GTnn::OptimizedSingleNN gt(
            GTnn::path_append(argv[1], "X.txt"), 
            THRESHOLD
        );
        cout << "OptimizedSingle object created successfully." << endl;
        vector<double> search_times;
        map<long, vector<uint>> final_results;
        for (int run = 0; run < NUM_RUNS; run++) {
            cout << "OptimizedSingle run " << (run + 1) << "/" << NUM_RUNS << endl;
            double total_search_time = 0.0;
            for (Eigen::Index i = 0; i < query_set.rows(); i++) {
                GTnn::vector_t query = query_set.row(i);
                auto start = high_resolution_clock::now();
                auto [result, num_dots] = gt.search(query);
                auto stop = high_resolution_clock::now();
                double search_time = duration_cast<microseconds>(
                    stop - start
                ).count() / 1.0e+3;
                total_search_time += search_time;
                if (run == NUM_RUNS - 1) final_results[i] = result;
            }
            double avg_search_time = total_search_time / query_set.rows();
            search_times.push_back(avg_search_time);
        }
        all_final_results["OptimizedSingle"] = final_results;
        auto [mean_time, std_dev_time] = calculate_stats(search_times);
        all_results.push_back({
            "OptimizedSingle", mean_time, std_dev_time, 0, 0, 0, 0, 0
        });
        cout << "OptimizedSingle completed successfully." << endl;
    }

    // --- 2. OPTIMIZED DOUBLE ---
    cout << "Starting OptimizedDouble tests..." << endl;
    {
        GTnn::OptimizedDoubleNN gt(
            GTnn::path_append(argv[1], "X.txt"), 
            THRESHOLD
        );
        cout << "OptimizedDouble object created successfully." << endl;
        vector<double> search_times;
        map<long, vector<uint>> final_results;
        for (int run = 0; run < NUM_RUNS; run++) {
            cout << "OptimizedDouble run " << (run + 1) << "/" << NUM_RUNS << endl;
            auto start = high_resolution_clock::now();
            auto [result_set, num_dots] = gt.search(query_set);
            auto stop = high_resolution_clock::now();
            double total_search_time = duration_cast<microseconds>(
                stop - start
            ).count() / 1.0e+3;
            double avg_search_time = total_search_time / query_set.rows();
            search_times.push_back(avg_search_time);
            if (run == NUM_RUNS - 1) {
                for (
                    long query_id = 0; 
                    query_id < static_cast<long>(result_set.size()); 
                    query_id++
                ) {
                    final_results[query_id] = result_set[query_id];
                }
            }
        }
        all_final_results["OptimizedDouble"] = final_results;
        auto [mean_time, std_dev_time] = calculate_stats(search_times);
        all_results.push_back({
            "OptimizedDouble", mean_time, std_dev_time, 0, 0, 0, 0, 0
        });
        cout << "OptimizedDouble completed successfully." << endl;
    }

    // --- 3. KNNS ---
    cout << "Starting Knns tests..." << endl;
    {
        GTnn::KnnsNN gt(GTnn::path_append(argv[1], "X.txt"), K_VAL);
        cout << "Knns object created successfully." << endl;
        vector<double> search_times;
        map<long, vector<uint>> final_results;
        for (int run = 0; run < NUM_RUNS; run++) {
            cout << "Knns run " << (run + 1) << "/" << NUM_RUNS << endl;
            double total_search_time = 0.0;
            for (Eigen::Index i = 0; i < query_set.rows(); i++) {
                GTnn::vector_t query = query_set.row(i);
                auto start = high_resolution_clock::now();
                auto [result, num_dots] = gt.search(query);
                auto stop = high_resolution_clock::now();
                double search_time = duration_cast<microseconds>(
                    stop - start
                ).count() / 1.0e+3;
                total_search_time += search_time;
                if (run == NUM_RUNS - 1) final_results[i] = result;
            }
            double avg_search_time = total_search_time / query_set.rows();
            search_times.push_back(avg_search_time);
        }
        all_final_results["Knns"] = final_results;
        auto [mean_time, std_dev_time] = calculate_stats(search_times);
        all_results.push_back({"Knns", mean_time, std_dev_time, 0, 0, 0, 0, 0});
        cout << "Knns completed successfully." << endl;
    }

    // --- 4. KNNS DOUBLE ---
    cout << "Starting KnnsDouble tests..." << endl;
    {
        GTnn::KnnsDoubleNN gt(GTnn::path_append(argv[1], "X.txt"), K_VAL);
        cout << "KnnsDouble object created successfully." << endl;
        vector<double> search_times;
        map<long, vector<uint>> final_results;
        for (int run = 0; run < NUM_RUNS; run++) {
            cout << "KnnsDouble run " << (run + 1) << "/" << NUM_RUNS << endl;
            auto start = high_resolution_clock::now();
            auto [result_set, num_dots] = gt.search(query_set);
            auto stop = high_resolution_clock::now();
            double total_search_time = duration_cast<microseconds>(
                stop - start
            ).count() / 1.0e+3;
            double avg_search_time = total_search_time / query_set.rows();
            search_times.push_back(avg_search_time);
            if (run == NUM_RUNS - 1) {
                for (auto& [query_id, results] : result_set) {
                    final_results[query_id] = results;
                }
            }
        }
        all_final_results["KnnsDouble"] = final_results;
        auto [mean_time, std_dev_time] = calculate_stats(search_times);
        all_results.push_back({
            "KnnsDouble", mean_time, std_dev_time, 0, 0, 0, 0, 0
        });
        cout << "KnnsDouble completed successfully." << endl;
    }

    // --- NAIVE SEARCH (RUN ONCE) ---
    double naive_time = 0.0;
    GTnn::matrix_t data_set;
    GTnn::extract_matrix(GTnn::path_append(argv[1], "X.txt"), data_set);
    map<long, 
        pair<vector<uint>, array<uint, K_VAL>>
    > ground_truth = run_naive_search_once(
        query_set, data_set, naive_time
    );

    // --- PRECISION AND RECALL FOR ALL METHODS ---
    cout << "\n=== STARTING PRECISION/RECALL ANALYSIS ===" << endl;
    cout << "Number of methods in all_results: " << all_results.size() << endl;
    
    for (auto& method : all_results) {
        vector<double> precisions, recalls;
        auto& final_results = all_final_results[method.name];
        cout << "\n=== CHECKING " << method.name << " ===" << endl;
        
        int perfect_recall_count = 0;
        int total_queries = 0;
        
        for (Eigen::Index i = 0; i < query_set.rows(); i++) {
            auto [precision, recall] = calculate_precision_recall(
                final_results[i],
                ground_truth[i].first,
                ground_truth[i].second,
                static_cast<bool>(method.name[0] == 'K')
            );
            precisions.push_back(precision);
            recalls.push_back(recall);
            total_queries++;
            
            if (recall >= 0.9999) {
                perfect_recall_count++;
            }
            
            // Output queries where recall is not 1.0
            if (recall < 0.9999) {  // Use 0.9999 to handle floating point precision
                cout << "Query " << i << " - " << method.name << ": "
                     << "Precision=" << fixed << setprecision(6) << precision
                     << ", Recall=" << fixed << setprecision(6) << recall
                     << " | Result size=" << final_results[i].size();
                
                if (method.name[0] == 'K') {
                    cout << ", Ground truth size=" << ground_truth[i].second.size();
                } else {
                    cout << ", Ground truth size=" << ground_truth[i].first.size();
                }
                cout << endl;
            }
        }
        
        cout << "Perfect recall queries: " << perfect_recall_count 
             << " / " << total_queries << endl;
        
        tie(
            method.mean_precision, method.stddev_precision
        ) = calculate_stats(precisions);
        tie(
            method.mean_recall, method.stddev_recall
        ) = calculate_stats(recalls);
        
        cout << "Calculated mean recall: " << fixed << setprecision(6) 
             << method.mean_recall << endl;
        
        method.speedup = naive_time / method.mean_time;
    }

    // --- FINAL RESULTS TABLE ---
    cout << endl << "=== FINAL RESULTS SUMMARY ===" << endl;
    cout << setw(15) << "Method"
         << setw(15) << "Time (ms)"
         << setw(15) << "StdDev"
         << setw(15) << "Precision"
         << setw(15) << "StdDev"
         << setw(15) << "Recall"
         << setw(15) << "StdDev"
         << setw(15) << "Speedup" << endl;
    cout << string(120, '-') << endl;
    for (const auto& method : all_results) {
        cout << setw(15) << method.name
             << setw(15) << fixed << setprecision(3) << method.mean_time
             << setw(15) << fixed << setprecision(3) << method.stddev_time
             << setw(15) << fixed << setprecision(4) << method.mean_precision
             << setw(15) << fixed << setprecision(4) << method.stddev_precision
             << setw(15) << fixed << setprecision(4) << method.mean_recall
             << setw(15) << fixed << setprecision(4) << method.stddev_recall
             << setw(15) << fixed << setprecision(2) << method.speedup << endl;
    }

    cout << endl << "Naive search time: " << 
        naive_time << " ms per query" << endl
        << endl << "=== BENCHMARK COMPLETE ===" << endl;
    return 0;
}

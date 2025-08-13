#include "GTnn/optimized_single.hpp"

int main(int argc, char ** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <dataset_path>" << endl;
        return 1;
    }

    GTnn::OptimizedSingleNN gt(GTnn::path_append(argv[1], "X.txt"), 0.8);
    GTnn::matrix_t query_set;
    GTnn::matrix_t data_set;
    
    if (!GTnn::extract_matrix(GTnn::path_append(argv[1], "Q.txt"), query_set)) {
        cerr << "Error extracting query matrix." << endl;
        return 1;
    }
    
    if (!GTnn::extract_matrix(GTnn::path_append(argv[1], "X.txt"), data_set)) {
        cerr << "Error extracting data matrix." << endl;
        return 1;
    }

    cout << "Running comparison between optimized and naive search..." << endl;
    cout << "Number of queries: " << query_set.rows() << endl;
    cout << "Number of data vectors: " << data_set.rows() << endl;
    cout << "Threshold: 0.8" << endl;
    cout << "Note: Naive search will be slow for large datasets..." << endl << endl;
    
    double threshold = 0.8;
    size_t total_optimized_results = 0;
    size_t total_naive_results = 0;
    size_t total_intersections = 0;
    
    for (long i = 0; i < query_set.rows(); i++) {
        if (i % 10 == 0 || i < 5) {
            cout << "Processing query " << i << "..." << flush;
        }
        
        GTnn::vector_t query = query_set.row(i);
        
        // Run optimized search
        auto [optimized_result, num_dots] = gt.search(query);
        
        // Run naive search
        vector<uint> naive_result;
        for (long j = 0; j < data_set.rows(); j++) {
            double score = query.dot(data_set.row(j));
            if (score >= threshold) {
                naive_result.push_back(j);
            }
        }
        
        // Calculate intersection
        set<uint> optimized_set(optimized_result.begin(), optimized_result.end());
        set<uint> naive_set(naive_result.begin(), naive_result.end());
        set<uint> intersection;
        set_intersection(optimized_set.begin(), optimized_set.end(),
                        naive_set.begin(), naive_set.end(),
                        inserter(intersection, intersection.begin()));
        
        total_optimized_results += optimized_result.size();
        total_naive_results += naive_result.size();
        total_intersections += intersection.size();
        
        if (i % 10 == 0 || i < 5 || optimized_result.size() != naive_result.size() || intersection.size() != naive_result.size()) {
            cout << " Optimized=" << optimized_result.size() 
                 << ", Naive=" << naive_result.size()
                 << ", Intersection=" << intersection.size();
            
            if (optimized_result.size() != naive_result.size() || intersection.size() != naive_result.size()) {
                cout << " [MISMATCH]";
            }
            cout << endl;
        } else if (i % 10 == 0) {
            cout << " OK" << endl;
        }
    }
    
    cout << "\n=== SUMMARY ===" << endl;
    cout << "Total optimized results: " << total_optimized_results << endl;
    cout << "Total naive results: " << total_naive_results << endl;
    cout << "Total intersections: " << total_intersections << endl;
    
    double precision = (total_optimized_results == 0) ? 1.0 : (double)total_intersections / total_optimized_results;
    double recall = (total_naive_results == 0) ? 1.0 : (double)total_intersections / total_naive_results;
    
    cout << "Overall Precision: " << precision << endl;
    cout << "Overall Recall: " << recall << endl;
    
    return 0;
}

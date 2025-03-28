#ifdef PLAIN
#include "GTnn/SPARSE_GTNN_SUM.hpp"
#endif
#ifdef SMART
#include "GTnn/SPARSE_GTNN_SUM_SMART.hpp"
#endif
#ifdef CLASSWISE
#include "GTnn/SPARSE_GTNN_SUM_CLASSWISE.hpp"
#endif
#ifdef CLASSWISE_SMART
#include "GTnn/SPARSE_GTNN_SUM_CLASSWISE_SMART.hpp"
#endif
#ifdef OPTIMIZED
#include "GTnn/SPARSE_OPTIMIZED_GTNN.hpp"
#endif
#ifndef DDEPTH
#define DDEPTH 0
#endif
#ifndef QDEPTH
#define QDEPTH 0
#endif

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_set_dir>" << std::endl;
        return EXIT_FAILURE;
    }
    std::string data_set_dir(argv[1]);
    #ifdef OPTIMIZED
    GT::GroupTestingNN<DDEPTH, QDEPTH> gt_nn(data_set_dir, 0.5);
    #else
    GT::GroupTestingNN gt_nn(data_set_dir, 0.5);
    #endif
    array<pair<double, size_t>, 3> times_and_num_dot_products;
    array<pair<double, double>, 2> precisions_and_recalls;
    pair<double, size_t> temp1;
    pair<double, double> temp2;
    cout << "Starting searches" << endl;
    for (int __iter = 0; __iter < (argc >= 3 ? std::stoi(argv[2]) : 1); __iter++) {
        #ifdef OPTIMIZED
        temp1 = gt_nn.search();
        #else
        temp1 = gt_nn.search<DDEPTH>();
        #endif
        temp2 = gt_nn.get_precision_and_recall();
        times_and_num_dot_products[0].first += temp1.first;
        times_and_num_dot_products[0].second += temp1.second;
        precisions_and_recalls[0].first += temp2.first;
        precisions_and_recalls[0].second += temp2.second;
        
        #ifndef OPTIMIZED
        temp1 = gt_nn.double_search<DDEPTH, QDEPTH>();
        temp2 = gt_nn.get_double_precision_and_recall();
        times_and_num_dot_products[1].first += temp1.first;
        times_and_num_dot_products[1].second += temp1.second;
        precisions_and_recalls[1].first += temp2.first;
        precisions_and_recalls[1].second += temp2.second;
        #endif

        temp1 = gt_nn.naive_search();
        times_and_num_dot_products[2].first += temp1.first;
        times_and_num_dot_products[2].second += temp1.second;
    }
    for (int i = 0; i < 3; i++) {
        times_and_num_dot_products[i].first /= (argc >= 3 ? std::stoi(argv[2]) : 1);
        times_and_num_dot_products[i].second /= (argc >= 3 ? std::stoi(argv[2]) : 1);
    }
    for (int i = 0; i < 2; i++) {
        precisions_and_recalls[i].first /= (argc >= 3 ? std::stoi(argv[2]) : 1);
        precisions_and_recalls[i].second /= (argc >= 3 ? std::stoi(argv[2]) : 1);
    }
    std::cout << "Average search time: " << times_and_num_dot_products[0].first / gt_nn.get_query_set_size() << std::endl << "Net number of dot products: " << times_and_num_dot_products[0].second <<
    std::endl << "Average precision: " << precisions_and_recalls[0].first << 
    std::endl << "Average recall: " << precisions_and_recalls[0].second << 
    #ifndef OPTIMIZED
    std::endl << "Average double search time: " << times_and_num_dot_products[1].first / gt_nn.get_query_set_size() << std::endl << "Net number of dot products: " << times_and_num_dot_products[1].second << 
    std::endl << "Average double precision: " << precisions_and_recalls[1].first << 
    std::endl << "Average double recall: " << precisions_and_recalls[1].second << 
    #endif
    std::endl << "Average naive search time: " << times_and_num_dot_products[2].first / gt_nn.get_query_set_size() << std::endl << "Net number of dot products: " << times_and_num_dot_products[2].second << std::endl;
    return EXIT_SUCCESS;
}

#include "sparse_types.hpp"
#include <thread>
#include <chrono>

using namespace std;

#define INNER_NUM 8
#define NUM_LEVELS 4
#define OUTER_NUM (1 << NUM_LEVELS)
#define PAGE_SIZE (INNER_NUM * OUTER_NUM)


struct ThresholdIndexRandomizedElem {
    SparseVec cum_sum;
    size_t base;
    array<uint8_t, INNER_NUM> offset;
};
using ThresholdIndexRandomizedSingle = std::array<ThresholdIndexRandomizedElem, OUTER_NUM>;
using ThresholdIndexRandomizedIndex = std::vector<ThresholdIndexRandomizedSingle>;

ThresholdIndexRandomizedIndex build_threshold_index_randomized(SparseMat& matrix) {
    ThresholdIndexRandomizedIndex index;
    size_t pages = (matrix.size() + PAGE_SIZE - 1) / PAGE_SIZE;
    index.resize(pages);
    
    for (size_t page = 0; page < pages; page++) {
        for (uint outer = 0; outer < OUTER_NUM; outer++) {
            index[page][outer].base = page * PAGE_SIZE;
            index[page][outer].cum_sum.clear();
            
            for (uint inner = 0; inner < INNER_NUM; inner++) {
                size_t idx = page * PAGE_SIZE + outer * INNER_NUM + inner;
                if (idx < matrix.size()) {
                    index[page][outer].offset[inner] = inner;
                }
            }
        }
    }
    return index;
}

class ThresholdIndexRandomized {
public:
    ThresholdIndexRandomized(string file_name, double __threshold = 0.8);
    std::pair<std::vector<uint>, size_t> search(SparseVec &query);
    std::array<double, 3> verify_results(SparseVec &query, std::vector<uint> &src_result);

    // protected:
    void search_thread(SparseVec &query, std::vector<uint> &result, size_t &num_dots, uint thread_id);
    inline void individual_search(SparseVec &query, std::vector<uint> &result, size_t &num_dots, double dot, ThresholdIndexRandomizedElem &pool);
    SparseMat data_set;
    ThresholdIndexRandomizedIndex data_index;
    uint size;
    uint dimention;
    double threshold;
};

// All GTnn and legacy typedefs removed. All types and methods are PascalCase and Python/numpy compatible.

ThresholdIndexRandomized::ThresholdIndexRandomized(std::string file_name, double __threshold) {
    std::tie(this->data_set, this->dimention) = read_sparse_matrix(file_name);
    this->threshold = __threshold;
    this->size = (this->data_set.size() + PAGE_SIZE - 1) / PAGE_SIZE;
    this->data_index = build_threshold_index_randomized(this->data_set);
}

std::pair<std::vector<uint>, size_t> ThresholdIndexRandomized::search(SparseVec& query) {
    std::array<std::thread, NUM_THREADS - 1> threads;
    std::array<std::pair<std::vector<uint>, size_t>, NUM_THREADS - 1> ranges;
    std::pair<std::vector<uint>, size_t> result;
    
    for (uint t = 0; t < NUM_THREADS - 1; t++) {
        ranges[t].first = std::vector<uint>();
        ranges[t].second = 0;
        threads[t] = std::thread(&ThresholdIndexRandomized::search_thread, this, std::ref(query), 
                std::ref(ranges[t].first), std::ref(ranges[t].second), t + 1);
    }
    this->search_thread(query, result.first, result.second, 0);
    
    for (uint t = 0; t < NUM_THREADS - 1; t++) {
        threads[t].join();
        result.first.insert(result.first.end(), ranges[t].first.begin(), ranges[t].first.end());
        result.second += ranges[t].second;
    }
    std::sort(result.first.begin(), result.first.end());
    return result;
}

std::array<double, 3> ThresholdIndexRandomized::verify_results(SparseVec& query, std::vector<uint>& result) {
    std::vector<uint> true_result;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < data_set.size(); i++) {
        if (dot_product(query, data_set[i]) >= this->threshold) {
            true_result.push_back(i);
        }
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double time = duration.count() / 1.0e+3;
    
    std::set<uint> true_set(true_result.begin(), true_result.end());
    std::set<uint> res_set(result.begin(), result.end());
    std::set<uint> match_set;
    std::set_intersection(true_set.begin(), true_set.end(),
                        res_set.begin(), res_set.end(),
                        std::inserter(match_set, match_set.begin()));
    
    double precision = (res_set.size() == 0) ? 1.0 : (static_cast<double>(match_set.size()) / res_set.size());
    double recall = (true_set.size() == 0) ? 1.0 : (static_cast<double>(match_set.size()) / true_set.size());
    return {time, precision, recall};
}

void ThresholdIndexRandomized::search_thread(SparseVec& query, std::vector<uint>& result, 
        size_t& num_dots, uint thread_id) {
    for (uint base = thread_id; base < this->size; base += NUM_THREADS) {
        std::array<double, (1 + NUM_LEVELS)> dot_vals;
        std::array<std::pair<uint, uint>, (1 + NUM_LEVELS)> ranges;
        
        dot_vals[0] = dot_product(query, this->data_index[base][0].cum_sum);
        ranges[0] = {0, OUTER_NUM};
        uint pos = 1;
        
        while (pos > 0) {
            if (dot_vals[pos - 1] < this->threshold) {
                pos--;
                continue;
            }
            uint start = ranges[pos - 1].first;
            uint end = ranges[pos - 1].second;
            uint mid = (start + end) / 2;
            
            if (start + 1 == end) {
                this->individual_search(query, result, num_dots, dot_vals[pos - 1], this->data_index[base][mid]);
                pos--;
                continue;
            }
            num_dots++;
            dot_vals[pos] = dot_product(query, this->data_index[base][mid].cum_sum);
            ranges[pos] = {mid, end};
            dot_vals[pos - 1] -= dot_vals[pos];
            ranges[pos - 1] = {start, mid};
            pos++;
        }
    }
}

inline void ThresholdIndexRandomized::individual_search(SparseVec& query, std::vector<uint>& result, 
                                                         size_t& num_dots, double dot, ThresholdIndexRandomizedElem& pool) {
    double net_dot = 0;
    for (uint i = 1; i < INNER_NUM; i++) {
        num_dots++;
        double dot_i = dot_product(query, this->data_set[pool.base + pool.offset[i]]);
        if (dot_i >= this->threshold) {
            result.push_back(pool.base + pool.offset[i]);
        }
        net_dot += dot_i;
    }
    if (dot - net_dot >= this->threshold) {
        result.push_back(pool.base + pool.offset[0]);
    }
}


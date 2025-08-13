# Sparse Source Directory

This directory contains all the sparse vector similarity search implementations and related files.

## Files Overview

### Core Implementation Files
- `optimized_gtnn.cpp` - Main optimized GTnn implementation
- `benchmark_optimized.cpp` - Benchmark for optimized implementations
- `compare_naive_linscan.cpp` - Comparison between naive search and LINSCAN
- `sparse_GTnn.cpp` - Original sparse GTnn implementation

### Header Files
- `GTnn/` - Directory containing all header files
  - `header.hpp` - Core data structures and functions
  - `optimized_linscan.hpp` - LINSCAN-style implementation
  - `naive_vs_linscan.hpp` - Comparison implementations
  - `optimized_double.hpp` - Optimized double precision implementation
  - `SPARSE_OPTIMIZED_GTNN.hpp` - Optimized GTnn class

### Build Files
- `optimized.mak` - Optimized build configuration
- `compare.mak` - Build configuration for comparison tools

### Documentation
- `NAIVE_VS_LINSCAN_ANALYSIS.md` - Detailed analysis of naive vs LINSCAN approaches
- `PERFORMANCE_COMPARISON_SUMMARY.md` - Quick performance comparison summary

## Building

### To build the optimized version:
```bash
make -f optimized.mak
```

### To build the comparison tool:
```bash
make -f compare.mak
```

## Running

### Optimized GTnn:
```bash
./bin/optimized_gtnn ../data/dataset.bin ../data/queries.bin
```

### Comparison tool:
```bash
./bin/compare_naive_linscan ../data/dataset.bin ../data/queries.bin
```

## Performance

The implementations in this directory focus on achieving high performance for sparse vector similarity search:

- **Naive approach**: ~1.3 QPS (baseline)
- **LINSCAN approach**: ~95 QPS (target)
- **Optimized implementations**: 15-2000+ QPS (estimated)

## Key Optimizations

1. **Algorithm**: LINSCAN-style inverted index approach
2. **Data Structures**: Structure of Arrays (SoA) for better cache locality
3. **Parallelization**: OpenMP for multi-threaded processing
4. **Compiler**: Aggressive optimization flags (-O3, -march=native, -flto)
5. **Memory**: Reduced allocations and improved access patterns

## Usage

All paths in the build files and documentation assume you're running from the `sparse_src/` directory. Data files are expected to be in `../data/` relative to this directory.

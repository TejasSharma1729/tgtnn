# Master Makefile for sparse GTnn implementations
# This makefile provides convenient targets for all builds

.PHONY: all optimized compare naive-search clean-all help run-benchmark

# Default target
all: optimized compare naive-search

# Build optimized GTnn
optimized:
	@echo "Building optimized GTnn..."
	$(MAKE) -f optimized.mak

# Build comparison tool
compare:
	@echo "Building comparison tool..."
	$(MAKE) -f compare.mak

# Build naive search tool
naive-search:
	@echo "Building naive search tool..."
	g++ -std=c++17 -O3 -pthread -o naive_search_only naive_search_only.cpp -I.

# Run comparison with big-ann-benchmarks data
run-benchmark: compare
	@echo "Running comparison with big-ann-benchmarks data..."
	$(MAKE) -f compare.mak run-benchmark

# Clean all build artifacts
clean-all:
	@echo "Cleaning all build artifacts..."
	$(MAKE) -f optimized.mak clean
	$(MAKE) -f compare.mak clean
	rm -f naive_search_only

# Run benchmark (requires data files)
benchmark: optimized
	@echo "Running benchmark..."
	$(MAKE) -f optimized.mak benchmark

# Profile optimized version
profile: optimized
	@echo "Running profiling..."
	$(MAKE) -f optimized.mak profile

# Help target
help:
	@echo "Available targets:"
	@echo "  all           - Build all targets (optimized + compare + naive-search)"
	@echo "  optimized     - Build optimized GTnn implementation"
	@echo "  compare       - Build naive vs LINSCAN comparison tool"
	@echo "  naive-search  - Build naive search only tool"
	@echo "  run-benchmark - Run comparison with big-ann-benchmarks data"
	@echo "  clean-all     - Clean all build artifacts"
	@echo "  benchmark     - Run benchmark (requires big-ann-benchmarks data)"
	@echo "  profile       - Run profiling (requires big-ann-benchmarks data)"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Data paths:"
	@echo "  Data file:  ~/big-ann-benchmarks/data/sparse/base_1M.csr"
	@echo "  Query file: ~/big-ann-benchmarks/data/sparse/queries.dev.csr"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                    # Build all"
	@echo "  make run-benchmark      # Build and run comparison"
	@echo "  make benchmark          # Run optimized GTnn benchmark"
	@echo "  make naive-search       # Build naive search tool"
	@echo "  ./naive_search_only     # Run naive search on both 1M and full datasets"
	@echo "  make clean-all          # Clean everything"

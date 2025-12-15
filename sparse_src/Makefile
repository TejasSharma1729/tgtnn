# Master Makefile for sparse GTnn implementations
# This makefile provides convenient targets for all builds

.PHONY: all optimized compare naive-search python-libs headers clean-all help run-benchmark

# Configuration
SRC_DIR = src
LIB_DIR = lib
ASSETS_DIR = ../assets

# Source CPP files to compile (all 11 self-contained modules)
SOURCE_FILES = \
	$(SRC_DIR)/sparse_types.cpp \
	$(SRC_DIR)/sparse_csr_create.cpp \
	$(SRC_DIR)/sparse_csr_read.cpp \
	$(SRC_DIR)/knn_index_dataset.cpp \
	$(SRC_DIR)/knn_index_double_group.cpp \
	$(SRC_DIR)/threshold_index_dataset.cpp \
	$(SRC_DIR)/threshold_index_double_group.cpp \
	$(SRC_DIR)/threshold_index_randomized.cpp \
	$(SRC_DIR)/threshold_index_sparse_optimized.cpp \
	$(SRC_DIR)/threshold_linscan_naive.cpp \
	$(SRC_DIR)/threshold_linscan_optimized.cpp

# Individual .so modules (one per C++ file)
SO_TARGETS = \
	$(LIB_DIR)/sparse_types.so \
	$(LIB_DIR)/sparse_csr_create.so \
	$(LIB_DIR)/sparse_csr_read.so \
	$(LIB_DIR)/knn_index_dataset.so \
	$(LIB_DIR)/knn_index_double_group.so \
	$(LIB_DIR)/threshold_index_dataset.so \
	$(LIB_DIR)/threshold_index_double_group.so \
	$(LIB_DIR)/threshold_index_randomized.so \
	$(LIB_DIR)/threshold_index_sparse_optimized.so \
	$(LIB_DIR)/threshold_linscan_naive.so \
	$(LIB_DIR)/threshold_linscan_optimized.so

# Default target
all: optimized compare naive-search

# ============================================================================
# Header Compilation Targets (Verification & Building)
# ============================================================================

# Compile all modules to Python .so
headers: $(SO_TARGETS)
	@echo "✓ All 11 modules compiled to $(LIB_DIR)/"

# Create lib directory if it doesn't exist
$(LIB_DIR):
	@mkdir -p $@

# Generic rule: compile any module from src/*.cpp to lib/*.so
$(LIB_DIR)/%.so: $(SRC_DIR)/%.cpp | $(LIB_DIR)
	@echo "  Compiling $*..."
	@g++ -std=c++17 -Wno-shadow -shared -fPIC \
		$$(python3 -m pybind11 --includes) \
		-I$(ASSETS_DIR)/Eigen -I$(ASSETS_DIR) \
		$< -o $@ 2>&1 && echo "    ✓ $* compiled" || (echo "    ✗ $* failed"; exit 1)

# Clean compiled modules
clean-headers:
	@echo "Removing compiled .so modules from $(LIB_DIR)..."
	rm -f $(LIB_DIR)/*.so
	@echo "✓ Cleaned $(LIB_DIR)/"

# Target to verify headers syntax (using CPP source files)
check-headers:
	@echo "Checking CPP file syntax (without compilation)..."
	@for file in $(SOURCE_FILES); do \
		echo "  Checking $$file..."; \
		g++ -std=c++17 -I$(ASSETS_DIR)/Eigen -I$(ASSETS_DIR) -fsyntax-only $$file || exit 1; \
	done
	@echo "✓ All source files have valid syntax"

# ============================================================================
# Existing Targets
# ============================================================================

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
	@echo ""
	@echo "C++ Compilation:"
	@echo "  all           - Build all targets (optimized + compare + naive-search)"
	@echo "  optimized     - Build optimized GTnn implementation"
	@echo "  compare       - Build naive vs LINSCAN comparison tool"
	@echo "  naive-search  - Build naive search only tool"
	@echo ""
	@echo "Header Verification:"
	@echo "  headers       - Compile all modules to individual Python .so files"
	@echo "  check-headers - Syntax-check all headers (lightweight, no compilation)"
	@echo "  clean-headers - Remove compiled .so modules"
	@echo ""
	@echo "NOTE: Each .so module is independent and self-contained."
	@echo ""
	@echo "Benchmarking:"
	@echo "  run-benchmark - Run comparison with big-ann-benchmarks data"
	@echo "  benchmark     - Run benchmark (requires big-ann-benchmarks data)"
	@echo "  profile       - Run profiling (requires big-ann-benchmarks data)"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean-all     - Clean all build artifacts (C++, Python, and data)"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "Quick Start:"
	@echo "  make headers          # Compile all headers to .so modules"
	@echo "  make check-headers    # Quick syntax validation (faster)"
	@echo ""
	@echo "Data paths (for benchmarking):"
	@echo "  Data file:  ~/big-ann-benchmarks/data/sparse/base_1M.csr"
	@echo "  Query file: ~/big-ann-benchmarks/data/sparse/queries.dev.csr"

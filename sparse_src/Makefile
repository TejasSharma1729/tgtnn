
# Makefile for building sparse_gtnn Python extension
.PHONY: all clean help

PYBIND11_FLAGS := $(shell python3 -m pybind11 --includes)
PYTHON_LIB := $(shell python3-config --ldflags)
CXX := g++
CXXFLAGS := -O3 -Wall -Wno-unused-variable -shared -std=c++17 -fPIC

SRC = src/pybind11_module.cpp
TARGET = sparse_gtnn.so
ASSETS_DIR = ../assets

all: $(TARGET)

$(TARGET): $(SRC)
	@echo "Compiling sparse_gtnn Python extension..."
	$(CXX) $(CXXFLAGS) $(PYBIND11_FLAGS) -I$(ASSETS_DIR)/Eigen -I$(ASSETS_DIR) $^ -o $@ $(PYTHON_LIB) && echo "✓ Built $@" || (echo "✗ Build failed"; exit 1)

clean:
	@echo "Cleaning build artifacts..."
	rm -f sparse_gtnn.so
	@echo "✓ Cleaned."

help:
	@echo "Available targets:"
	@echo "  all    - Build sparse_gtnn Python extension (sparse_gtnn.so)"
	@echo "  clean  - Remove build artifacts"
	@echo "  help   - Show this help message"
	@echo ""
	@echo "Usage:"
	@echo "  python3 -c 'import sparse_gtnn'"
	@echo "  # or use in your Python scripts:"
	@echo "  import sparse_gtnn"

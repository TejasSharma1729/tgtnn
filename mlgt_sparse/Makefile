# Build mlgt_saffron Python extension for sparse vectors

PYTHON ?= python3
CXX ?= g++

# Compiler flags
CXXFLAGS := $(CXXFLAGS) -O3 -Wall -Wno-shadow -std=c++20 -fPIC -fopenmp
SHARED := -shared

# Pybind11 and Eigen configuration
# Assuming assets/Eigen is at the project root
INCLUDES := $(shell $(PYTHON) -m pybind11 --includes) -I../assets/ -Isrc/

# Python library paths
PYLIBDIR := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')
PYLIBS := $(shell $(PYTHON) -c 'import sysconfig; print("{} {}".format(sysconfig.get_config_var("LIBS") or "", sysconfig.get_config_var("SYSLIBS") or ""))')
LDFLAGS := -L$(PYLIBDIR) $(PYLIBS) -Wl,-rpath,$(PYLIBDIR)

TARGET := mlgt_sparse$(shell $(PYTHON)-config --extension-suffix)
SRC := src/mlgt_saffron.cpp
HDR := $(wildcard src/*.hpp)

.PHONY: all build clean help

all: build

build: $(TARGET)
	stubgen -m mlgt_sparse -o .

$(TARGET): $(SRC) $(HDR)
	@echo "Building mlgt_sparse extension for sparse vectors..."
	$(CXX) $(SHARED) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(LDFLAGS)
	@echo "✓ Built $(TARGET)"

test: build
	@echo "Testing mlgt_sparse extension..."
	ipython test.py

clean:
	@echo "Cleaning build artifacts..."
	rm -f *.so *.o *.pyc src/*.o *.pyi
	@echo "✓ Cleaned"

help:
	@echo "MLGT SAFFRON Sparse Extension Build"
	@echo "==================================="
	@echo "Targets:"
	@echo "  all   - Build extension (default)"
	@echo "  build - Build the shared library"
	@echo "  clean - Remove built artifacts"
	@echo "  help  - Show this help message"

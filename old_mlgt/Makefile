# Build mlgt_saffron Python extension and generate type stubs

PYTHON ?= python3
CXX ?= g++

# Get Python version info
PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; ext = sysconfig.get_config_var('EXT_SUFFIX'); print(ext.replace('.cpython-' + '$(PYTHON_VERSION)'.replace('.', ''), '').replace('.so', '') if ext else '')")

# Compiler flags
CXXFLAGS := $(CXXFLAGS) -O3 -Wall -Wno-shadow -std=c++17 -fPIC
SHARED := -shared

# Pybind11 configuration
INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

# Python library paths
PYLIBDIR := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')
PYLIBS := $(shell $(PYTHON) -c 'import sysconfig; print("{} {}".format(sysconfig.get_config_var("LIBS") or "", sysconfig.get_config_var("SYSLIBS") or ""))')
LDFLAGS := -L$(PYLIBDIR) $(PYLIBS) -Wl,-rpath,$(PYLIBDIR)

# Dependency headers
INCLUDES_DIR := -I.
TARGET := mlgt_saffron.so
SRC := mlgt_saffron.cpp
HDR := mlgt_saffron.hpp data_structures.hpp inverted_index.hpp saffron.hpp
PYSTUB := mlgt_saffron.pyi

.PHONY: all build stubs clean distclean help

all: build stubs

build: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	@echo "Building mlgt_saffron extension (Python $(PYTHON_VERSION))..."
	$(CXX) $(SHARED) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(INCLUDES_DIR) $(LDFLAGS)
	@echo "✓ Built $(TARGET)"

stubs: $(TARGET)
	@echo "Generating type stubs..."
	@if command -v stubgen > /dev/null; then \
		stubgen -m mlgt_saffron -o . 2>/dev/null && echo "✓ Generated $(PYSTUB)"; \
	else \
		echo "Warning: stubgen not found, skipping type stubs generation"; \
	fi

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET) $(PYSTUB) *.o *.pyc
	@echo "✓ Cleaned"

distclean: clean
	@echo "Cleaning all generated files..."
	rm -rf __pycache__ .pytype

help:
	@echo "MLGT SAFFRON Extension Build"
	@echo "============================="
	@echo "Targets:"
	@echo "  all       - Build extension and generate stubs (default)"
	@echo "  build     - Build the shared library only"
	@echo "  stubs     - Generate Python type stubs only"
	@echo "  clean     - Remove built artifacts"
	@echo "  distclean - Remove all generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON    - Python interpreter (default: python3)"
	@echo "  CXX       - C++ compiler (default: g++)"
	@echo "  CXXFLAGS  - Additional C++ compiler flags"

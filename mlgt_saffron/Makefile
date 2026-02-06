# Build mlgt_saffron Python extension and generate type stubs

PYTHON ?= python3
CXX ?= g++

# Get Python version info
PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; ext = sysconfig.get_config_var('EXT_SUFFIX'); print(ext.replace('.cpython-' + '$(PYTHON_VERSION)'.replace('.', ''), '').replace('.so', '') if ext else '')")

# Compiler flags
OPT ?= -O3
CXXFLAGS := $(CXXFLAGS) $(OPT) -Wall -Wno-shadow -std=c++20 -fPIC
SHARED := -shared

# Pybind11 configuration
INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)

# Python library paths
PYLIBDIR := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')
PYLIBS := $(shell $(PYTHON) -c 'import sysconfig; print("{} {}".format(sysconfig.get_config_var("LIBS") or "", sysconfig.get_config_var("SYSLIBS") or ""))')
LDFLAGS := -L$(PYLIBDIR) $(PYLIBS) -Wl,-rpath,$(PYLIBDIR)

# Dependency headers
INCLUDES_DIR := -Isrc
TARGET := mlgt_saffron.so
SRC := src/mlgt_saffron.cpp
HDR := src/headers.hpp src/saffron.hpp src/saffron_index.hpp
INIT_PY := __init__.py
PYSTUB := __init__.pyi

# AddressSanitizer support
LIBASAN := $(shell $(CXX) -print-file-name=libasan.so)

.PHONY: all build stubs clean distclean help asan

all: build stubs $(INIT_PY)

asan: CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
asan: LDFLAGS += -fsanitize=address
asan: OPT = -O1
asan: clean all

build: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	@echo "Building mlgt_saffron extension (Python $(PYTHON_VERSION))..."
	$(CXX) $(SHARED) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(INCLUDES_DIR) $(LDFLAGS)
	@echo "✓ Built $(TARGET)"

$(INIT_PY):
	@echo "from .mlgt_saffron import *" > $(INIT_PY)
	@echo "✓ Created $(INIT_PY)"

stubs: $(TARGET)
	@echo "Generating type stubs..."
	@if command -v stubgen > /dev/null; then \
		IS_ASAN=$$(ldd $(TARGET) | grep libasan > /dev/null && echo 1 || echo 0); \
		if [ "$$IS_ASAN" = "1" ]; then \
			LD_PRELOAD=$(LIBASAN) stubgen -m mlgt_saffron -o . 2>/dev/null && mv mlgt_saffron.pyi $(PYSTUB) && echo "✓ Generated $(PYSTUB) (ASan mode)"; \
		else \
			stubgen -m mlgt_saffron -o . 2>/dev/null && mv mlgt_saffron.pyi $(PYSTUB) && echo "✓ Generated $(PYSTUB)"; \
		fi; \
	else \
		echo "Warning: stubgen not found, skipping type stubs generation"; \
	fi

tests: build stubs
	@echo "Running tests..."
	$(PYTHON) test.py -q 100
	@echo "✓ All tests passed"

test_asan: asan
	@echo "Running tests with AddressSanitizer..."
	LD_PRELOAD=$(LIBASAN) ASAN_OPTIONS=detect_leaks=1 $(PYTHON) test.py -q 100
	@echo "✓ All tests passed with ASan"

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(TARGET) $(PYSTUB) $(INIT_PY) *.o *.pyc
	@echo "✓ Cleaned"

distclean: clean
	@echo "Cleaning all generated files..."
	rm -rf __pycache__ .pytype

help:
	@echo "MLGT SAFFRON Extension Build"
	@echo "============================="
	@echo "Targets:"
	@echo "  all       - Build extension and generate stubs (default, no ASan)"
	@echo "  asan      - Build extension with AddressSanitizer enabled"
	@echo "  build     - Build the shared library only"
	@echo "  stubs     - Generate Python type stubs only"
	@echo "  tests     - Build and run basic tests"
	@echo "  test_asan - Build and run tests with AddressSanitizer"
	@echo "  clean     - Remove built artifacts"
	@echo "  distclean - Remove all generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  PYTHON    - Python interpreter (default: python3)"
	@echo "  CXX       - C++ compiler (default: g++)"
	@echo "  CXXFLAGS  - Additional C++ compiler flags"
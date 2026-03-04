# Build mlgt_saffron Python extension and generate type stubs

PYTHON  ?= python3
CXX     ?= g++
TARGET  := mlgt_saffron.so
PYSTUB  := mlgt_saffron.pyi
INIT_PY := __init__.py

# Get Python configuration
PY_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PY_LDFLAGS  := $(shell $(PYTHON) -c 'import sysconfig; print("-L" + (sysconfig.get_config_var("LIBDIR") or "") + " " + (sysconfig.get_config_var("LIBS") or "") + " " + (sysconfig.get_config_var("SYSLIBS") or ""))') -lgomp

# Compiler flags
OPT      ?= -O3
CXXFLAGS := $(OPT) -Wall -Wno-shadow -std=c++20 -fPIC -fopenmp $(CXXFLAGS)
LDFLAGS  := $(PY_LDFLAGS) -shared -Wl,-rpath,$(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')

# Dependency headers
INCLUDES_DIR := -Isrc -I../assets
SRC := src/mlgt_saffron.cpp
HDR := $(wildcard src/*.hpp)

# AddressSanitizer support
LIBASAN := $(shell $(CXX) -print-file-name=libasan.so)

.PHONY: all build stubs clean asan tests help

all: build stubs $(INIT_PY)

asan: CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
asan: LDFLAGS  += -fsanitize=address
asan: OPT = -O1
asan: clean all

build: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	@echo "Building $(TARGET)..."
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(PY_INCLUDES) $(INCLUDES_DIR) $(LDFLAGS)
	@echo "✓ Built $(TARGET)"

$(INIT_PY):
	@echo "from .mlgt_saffron import *" > $@
	@echo "✓ Created $@"

stubs: $(TARGET)
	@echo "Generating type stubs..."
	@if command -v stubgen > /dev/null; then \
		IS_ASAN=$$(ldd $(TARGET) | grep libasan > /dev/null && echo 1 || echo 0); \
		if [ "$$IS_ASAN" = "1" ]; then \
			LD_PRELOAD=$(LIBASAN) stubgen -m mlgt_saffron -o . 2>/dev/null; \
		else \
			stubgen -m mlgt_saffron -o . 2>/dev/null; \
		fi; \
		mv mlgt_saffron.pyi $(PYSTUB) 2>/dev/null || true; \
		echo "✓ Generated $(PYSTUB)"; \
	else \
		echo "Warning: stubgen not found, skipping type stubs"; \
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
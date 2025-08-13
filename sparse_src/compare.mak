# Makefile for Naive vs LINSCAN comparison

CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -Wall -Wextra
CXXFLAGS += -fopenmp -pthread -DNDEBUG
CXXFLAGS += -I.
LDFLAGS = -fopenmp -pthread

SRCDIR = .
OBJDIR = obj
BINDIR = bin

# Source files
COMPARE_SOURCES = $(SRCDIR)/compare_naive_linscan.cpp
COMPARE_OBJECTS = $(COMPARE_SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
COMPARE_TARGET = $(BINDIR)/compare_naive_linscan

.PHONY: all clean compare

all: $(COMPARE_TARGET)

$(COMPARE_TARGET): $(COMPARE_OBJECTS) | $(BINDIR)
	$(CXX) $(COMPARE_OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Build and run comparison
compare: $(COMPARE_TARGET)
	@echo "Built comparison tool: $(COMPARE_TARGET)"
	@echo "Usage: $(COMPARE_TARGET) <data_file> <query_file>"
	@echo ""
	@echo "Example:"
	@echo "  $(COMPARE_TARGET) ~/big-ann-benchmarks/data/sparse/base_1M.csr ~/big-ann-benchmarks/data/sparse/queries.dev.csr"

# Run with big-ann-benchmarks data
run-benchmark: $(COMPARE_TARGET)
	@echo "Running comparison with big-ann-benchmarks data..."
	./$(COMPARE_TARGET) ~/big-ann-benchmarks/data/sparse/base_1M.csr ~/big-ann-benchmarks/data/sparse/queries.dev.csr

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build all targets"
	@echo "  compare      - Build comparison tool"
	@echo "  run-benchmark - Run comparison with big-ann-benchmarks data"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help"
	@echo ""
	@echo "Data paths:"
	@echo "  Data file:  ~/big-ann-benchmarks/data/sparse/base_1M.csr"
	@echo "  Query file: ~/big-ann-benchmarks/data/sparse/queries.dev.csr"

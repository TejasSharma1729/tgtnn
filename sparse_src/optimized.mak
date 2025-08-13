# Optimized Makefile for GTnn
CXX = g++
CXXFLAGS = -O3 -march=native -mtune=native -flto -DNDEBUG
CXXFLAGS += -ffast-math -funroll-loops -fprefetch-loop-arrays
CXXFLAGS += -std=c++17 -Wall -Wextra
CXXFLAGS += -I/usr/include/eigen3
CXXFLAGS += -fopenmp

# For maximum performance
CXXFLAGS += -DEIGEN_NO_DEBUG -DEIGEN_NO_STATIC_ASSERT
CXXFLAGS += -mavx2 -mfma -msse4.2

# Link-time optimization
LDFLAGS = -flto -fopenmp

SRCDIR = .
OBJDIR = obj
BINDIR = bin

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)
TARGET = $(BINDIR)/optimized_gtnn

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Benchmark target
benchmark: $(TARGET)
	./$(TARGET) ~/big-ann-benchmarks/data/sparse/base_1M.csr ~/big-ann-benchmarks/data/sparse/queries.dev.csr

# Profile target
profile: CXXFLAGS += -pg
profile: $(TARGET)
	./$(TARGET) ~/big-ann-benchmarks/data/sparse/base_1M.csr ~/big-ann-benchmarks/data/sparse/queries.dev.csr
	gprof $(TARGET) gmon.out > analysis.txt

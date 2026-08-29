CXX      = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC -fopenmp
INCLUDES = $(shell python3 -m pybind11 --includes) \
           -I. \
           -I../k_btnn/src \
           -I../assets/Eigen \
           -I../assets
LIB_SUFFIX = $(shell python3-config --extension-suffix)
TARGET = gtnn_ablation.so
SRCS   = pybind11_ablation_module.cpp

all: $(TARGET) stub check

$(TARGET): $(SRCS) knns_dataset_reordered.hpp header.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET)

stub: $(TARGET)
	stubgen -m gtnn_ablation -o .

check: $(TARGET)
	python3 -c "import sys; sys.path.insert(0, '.'); import gtnn_ablation; print('gtnn_ablation loaded:', dir(gtnn_ablation))"

clean:
	rm -f $(TARGET) gtnn_ablation.pyi *.so

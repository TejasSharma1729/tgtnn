CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC -fopenmp
# Add paths manually if Python/environment settings are missing
INCLUDES = $(shell python3 -m pybind11 --includes) -I./src \
           -I../assets/SHNSW/hnswlib \
           -I../assets/Eigen \
           -I../assets
LIB_SUFFIX = $(shell python3-config --extension-suffix)
TARGET = gtnn.so
SRCS = src/pybind11_module.cpp

all: $(TARGET) stub check

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRCS) -o $(TARGET)

stub: $(TARGET)
	stubgen -m gtnn -o .

check: $(TARGET)
	python3 -c "import gtnn; print('Module loaded successfully')"

clean:
	rm -f $(TARGET) gtnn.pyi *.so
	rm -rf out/

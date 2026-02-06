CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC -fopenmp
INCLUDES = $(shell python3 -m pybind11 --includes) -I./src -I$(shell python3 -c "import eigenpip; print(eigenpip.get_include())")
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

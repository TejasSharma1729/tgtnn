#include <iostream>
#include <vector>

int main() {
    std::vector<int> vec;
    vec.push_back(1);
    
    // This should cause an error - undefined variable
    undefinedVariable = 5;
    
    // This should cause an error - wrong function call
    vec.unknown_method();
    
    return 0;
}

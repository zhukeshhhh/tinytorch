#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>

struct matrix
{
    float* _values;
    std::size_t _rows;
    std::size_t _cols;

    matrix(
        std::size_t rows,
        std::size_t cols
    ) : _rows{rows},
        _cols{cols}
    {
        _values = (float*)malloc(sizeof(float) * rows * cols);
    }
};

int main() {

    matrix m = matrix(3, 4);
    
    std::cout << m._rows;

    return 0;
}


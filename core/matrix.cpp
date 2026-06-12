#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>

class Matrix {
public:
    float* _values;
    std::size_t _rows;
    std::size_t _cols;
    std::vector<std::size_t> _mshape;

    Matrix(
        std::size_t rows,
        std::size_t cols
    ) : _rows{rows},
        _cols{cols}
    {
        _mshape = {rows, cols};
        _values = new float[rows * cols]{};
    }

    Matrix(std::size_t size)
        : _rows{1}, _cols{size}
    {
        _values = new float[size]; // no values
    }

    Matrix(const Matrix& other)
        : _rows{other._rows}, _cols{other._cols}
    {
        _values = new float[_rows * _cols]; // no values
        std::copy(other._values, other._values + (_cols * _rows), _values);
    }

    std::size_t getSize() {
        return _rows * _cols;
    }

    std::vector<std::size_t> getShape() {
        return _mshape;
    }
};


int main() {

    Matrix m(3, 5);

    for (int i = 0; i < m._rows; i++) {
        for (int j = 0; j < m._cols; j++) {
            std::cout << m._values[i * m._cols + j];
        }
        std::cout << std::endl;
    }

    std::cout << m._mshape[0] << " " << m._mshape[1] << std::endl;

    return 0;
}
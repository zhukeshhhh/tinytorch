#include "matrix_cpu.hpp"

MatrixCpu::MatrixCpu(std::size_t rows, std::size_t cols)
    : _rows{rows}, _cols{cols}
{
    _values = new float[rows * cols]{};
}

MatrixCpu::MatrixCpu(float value, std::size_t rows, std::size_t cols)
    : _rows{rows}, _cols{cols}
{
    _values[0] = value;
}

MatrixCpu::~MatrixCpu() {
    delete[] _values;
}

std::size_t MatrixCpu::rows() const { return _rows; }
std::size_t MatrixCpu::cols() const { return _cols; }
std::size_t MatrixCpu::size() const { return _rows * _cols; }

float* MatrixCpu::values() const { return _values; }

Matrix* MatrixCpu::add(const Matrix& other) const {
    MatrixCpu* result = new MatrixCpu(_rows, _cols);
    for (std::size_t i = 0; i < size(); i++) {
        result->_values[i] = _values[i] + other.values()[i];
    }
    return result;
}
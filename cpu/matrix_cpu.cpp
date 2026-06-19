#include "matrix_cpu.hpp"
#include <random>

MatrixCpu::MatrixCpu(std::size_t rows, std::size_t cols)
    : _rows{rows}, _cols{cols}
{
    _values = new float[rows * cols]{};
}

MatrixCpu::MatrixCpu(float fillValue, std::size_t rows, std::size_t cols)
    : _rows{rows}, _cols{cols}
{
    _values = new float[rows * cols];

    for (std::size_t i = 0; i < rows * cols; i++) {
        _values[i] = fillValue;
    }
}

MatrixCpu::~MatrixCpu() {
    delete[] _values;
}

Matrix* MatrixCpu::add(const Matrix& other) const {
    MatrixCpu* result = new MatrixCpu(_rows, _cols);
    for (std::size_t i = 0; i < size(); i++) {
        result->_values[i] = _values[i] + other.values()[i];
    }
    return result;
}

Matrix* MatrixCpu::matmul(const Matrix& other) const {
    MatrixCpu* result = new MatrixCpu(1, 1);
    return result;
}

Matrix* MatrixCpu::relu() {
    MatrixCpu* result = new MatrixCpu(1, 1);
    return result;
}

Matrix* MatrixCpu::randn() {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < rows() * cols(); i++) {
        _values[i] = dist(gen);
    }
    return this;
}


std::size_t MatrixCpu::rows() const { return _rows; }
std::size_t MatrixCpu::cols() const { return _cols; }
std::size_t MatrixCpu::size() const { return _rows * _cols; }
float* MatrixCpu::values() const { return _values; }
float* MatrixCpu::at(std::size_t index) { return &_values[index]; }
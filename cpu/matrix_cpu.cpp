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
    for (std::size_t i = 0; i < _rows * _cols; i++) {
        result->_values[i] = _values[i] + other.values()[i];
    }

    return result;
}

Matrix* MatrixCpu::matmul(const Matrix& other) const {
    if (_cols != other.rows())
        throw std::runtime_error("matmul: dimensions do not match\n");

    MatrixCpu* result = new MatrixCpu(_rows, other.cols());

    std::size_t N = _rows;
    std::size_t M = other.cols();
    std::size_t K = _cols;
    
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            auto a = _values[i * K + k];
            for (std::size_t j = 0; j < M; ++j) {
                result->_values[i * M + j] +=
                    a * other.values()[k * M + j];
            }
        }
    }

    return (Matrix*)result;
}

Matrix* MatrixCpu::relu() {
    for (std::size_t i = 0; i < _rows * _cols; i++) {
        _values[i] = (_values[i] > 0) ? _values[i] : 0.0f;
    }
    return (Matrix*)this;
}

Matrix* MatrixCpu::randn() {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (std::size_t i = 0; i < _rows * _cols; i++) {
        _values[i] = dist(gen);
    }
    return (Matrix*)this;
}

Matrix* MatrixCpu::transpose() {
    MatrixCpu* result = new MatrixCpu(_cols, _rows);

    for (std::size_t i = 0; i < _rows; i++) {
        for (std::size_t j = 0; j < _cols; j++) {
            result->_values[j * _rows + i] = this->_values[i * _cols + j];
        }
    }

    return (Matrix*)result;
}

std::size_t MatrixCpu::rows() const { return _rows; }
std::size_t MatrixCpu::cols() const { return _cols; }
std::size_t MatrixCpu::size() const { return _rows * _cols; }
float* MatrixCpu::values() const { return _values; }
float* MatrixCpu::at(std::size_t index) { return &_values[index]; }
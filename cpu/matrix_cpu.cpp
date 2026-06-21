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

MatrixCpu::MatrixCpu(const Matrix& other)
    : _rows{other.rows()}, _cols{other.cols()}
{
    _values = new float[_rows * _cols];

    for (std::size_t i = 0; i < _rows * _cols; i++) {
        _values[i] = other.values()[i];
    }
}

MatrixCpu::~MatrixCpu() {
    delete[] _values;
}

Matrix* MatrixCpu::add(const Matrix& other) const {

    // same-sized matices
    if (_rows == other.rows() && _cols == other.cols()) {
        auto* result = new MatrixCpu(_rows, _cols);

        for (std::size_t i = 0; i < _rows * _cols; i++) {
            result->_values[i] = _values[i] + other.values()[i];
        }

        return result;
    }
    
    // row vector + col vector
    if (_rows == 1 && other.cols() == 1 && _cols == other.rows()) {
        auto* result = new MatrixCpu(_cols, _cols);

        for (std::size_t i = 0; i < _cols; i++) {
            for (std::size_t j = 0; j < _cols; j++) {
                result->_values[i * _cols + j] += _values[j] + other.values()[i];
            }
        }

        return result;
    }

    // col vector + row vector
    if (_cols == 1 && other.rows() == 1 && _rows == other.cols()) {
        auto* result = new MatrixCpu(_rows, _rows);

        for (std::size_t i = 0; i < _rows; i++) {
            for (std::size_t j = 0; j < _rows; j++) {
                result->_values[i * _rows + j] += _values[i] + other.values()[j];
            }
        }

        return result;
    }

    // matrix + row vector (equal cols)
    if (_cols == other.cols() && other.rows() == 1) {
        auto* result = new MatrixCpu(_rows, _cols);

        for (std::size_t i = 0; i < _rows; i++) {
            for (std::size_t j = 0; j < _cols; j++) {
                result->_values[i * _cols + j] = _values[i * _cols + j] + other.values()[j];
            }
        }

        return result;
    }

    // row vector + matrix (equal cols)
    if (_rows == 1 && _cols == other.cols()) {
        auto* result = new MatrixCpu(other.rows(), other.cols());

        for (std::size_t i = 0; i < other.rows(); i++) {
            for (std::size_t j = 0; j < _cols; j++) {
                result->_values[i * _cols + j] = _values[j] + other.values()[i * _cols + j];
            }
        }

        return result;
    }

    // matrix + col vector (equal rows)
    if (_rows == other.rows() && other.cols() == 1) {
        auto* result = new MatrixCpu(_rows, _cols);

        for (std::size_t i = 0; i < _rows; i++) {
            for (std::size_t j = 0; j < _cols; j++) {
                result->_values[i * _cols + j] = _values[i * _cols + j] + other.values()[i];
            }
        }

        return result;
    }

    // col vector + matrix (equal rows)
    if (_cols == 1 && _rows == other.rows()) {
        auto* result = new MatrixCpu(other.rows(), other.cols());

        for (std::size_t j = 0; j < other.cols(); j++) {
            for (std::size_t i = 0; i < _rows; i++)
            {
                result->_values[i * other.cols() + j] = _values[i] + other.values()[i * other.cols() + j];
            }
        }

        return result;
    }

    // matrix + scalar
    if (size() != 1 && other.size() == 1) {
        auto* result = new MatrixCpu(_rows, _cols);
        
        float addValue = other.values()[0];

        for (std::size_t i = 0; i < size(); i++) {
            result->_values[i] = _values[i] + addValue;
        }

        return result;
    }

    // scalar + matrix
    if (size() == 1 && other.size() != 1) {
        auto result = new MatrixCpu(other.rows(), other.cols());
        float addValue = _values[0];

        for (std::size_t i = 0; i < other.size(); i++) {
            result->_values[i] = other.values()[i] + addValue;
        }

        return result;
    }

    throw std::runtime_error("Matrix* add: dimensions do not match. Broadcasting failed\n");
}

Matrix* MatrixCpu::matmul(const Matrix& other) const {
    if (_cols != other.rows())
        throw std::runtime_error("Matrix* matmul: dimensions do not match\n");

    auto* result = new MatrixCpu(_rows, other.cols());

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
    auto* result = new MatrixCpu(_rows, _cols);
    
    for (std::size_t i = 0; i < _rows * _cols; i++) {
        result->_values[i] = (_values[i] > 0) ? _values[i] : 0.0f;
    }
        
    return result;
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

Matrix* MatrixCpu::relu_backward(const Matrix& upstream_grad) const {
    auto* result = new MatrixCpu(_rows, _cols);
    for (std::size_t i = 0; i < size(); i++) {
        result->_values[i] = (_values[i] > 0) ? upstream_grad.values()[i] : 0.0f;
    }

    return result;
}

std::size_t MatrixCpu::rows() const { return _rows; }
std::size_t MatrixCpu::cols() const { return _cols; }
std::size_t MatrixCpu::size() const { return _rows * _cols; }
float* MatrixCpu::values() const { return _values; }
float* MatrixCpu::at(std::size_t index) { return &_values[index]; }
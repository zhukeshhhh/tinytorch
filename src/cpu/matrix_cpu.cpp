#include <random>
#include <iostream>
#include "tinytorch/cpu/matrix_cpu.hpp"

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
    _values = new float[numel()];

    for (std::size_t i = 0; i < numel(); i++) {
        _values[i] = other.values()[i];
    }
}

MatrixCpu::~MatrixCpu() {
    delete[] _values;
}

Matrix* MatrixCpu::add(const Matrix& other) const {
    if (_rows != other.rows() && _rows != 1 && other.rows() != 1)
        throw std::runtime_error("MatrixCpu::add : incompatible row dimensions\n");
    if (_cols != other.cols() && _cols != 1 && other.cols() != 1)
        throw std::runtime_error("MatrixCpu::add : incompatible col dimensions\n");

    std::size_t out_rows = std::max(_rows, other.rows());
    std::size_t out_cols = std::max(_cols, other.cols());

    auto* result = new MatrixCpu(out_rows, out_cols);

    for (std::size_t i = 0; i < out_rows; i++) {
        for (std::size_t j = 0; j < out_cols; j++) {
            std::size_t a_idx = (i % _rows) * _cols + (j % _cols);
            std::size_t b_idx = (i % other.rows()) * other.cols() + (j % other.cols());
            result->_values[i * out_cols + j] = _values[a_idx] + other.values()[b_idx];
        }
    }

    return result;
}

Matrix* MatrixCpu::matmul(const Matrix& other) const {

    if (other.numel() == 1) {
        return matsmul(other);
    }

    if (numel() == 1) {
        return smatmul(other);
    }

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

Matrix* MatrixCpu::mul(const Matrix& other) const {
    if (_rows != other.rows() || _cols != other.cols())
        throw std::runtime_error("Matrix::mul() : matrix sizes do not match\n");

    auto* result = new MatrixCpu(_rows, _cols);

    for (std::size_t i = 0; i < numel(); i++) {
        result->_values[i] = _values[i] * other.values()[i];
    }

    return result;
}

Matrix* MatrixCpu::relu() const {
    auto* result = new MatrixCpu(_rows, _cols);
    
    for (std::size_t i = 0; i < numel(); i++) {
        result->_values[i] = (_values[i] > 0) ? _values[i] : 0.0f;
    }
        
    return result;
}

Matrix* MatrixCpu::relu_backward(const Matrix& upstream_grad) const {
    auto* result = new MatrixCpu(_rows, _cols);
    for (std::size_t i = 0; i < numel(); i++) {
        result->_values[i] = (_values[i] > 0) ? upstream_grad.values()[i] : 0.0f;
    }

    return result;
}

Matrix* MatrixCpu::softmax() const {
    auto* result = new MatrixCpu(_rows, _cols);

    for (std::size_t i = 0; i < _rows; i++) {
        float max_val = _values[0];
        for (std::size_t j = 0; j < _cols; j++) {
            float v = _values[i * _cols + j];
            if (v > max_val) max_val = v;
        }

        float sum_exp = 0.0f;
        for (std::size_t j = 0; j < _cols; j++) {
            float e = std::exp(_values[i * _cols + j] - max_val);
            result->_values[i * _cols + j] = e;
            sum_exp += e;
        }

        for (std::size_t j = 0; j < _cols; j++) {
            result->_values[i * _cols + j] /= sum_exp;
        }
    }

    return result;
}

Matrix* MatrixCpu::softmax_backward(const Matrix& upstream_grad) const {
    const auto& g = static_cast<const MatrixCpu&>(upstream_grad);
    auto* grad = new MatrixCpu(_rows, _cols);

    for (std::size_t i = 0; i < _rows; i++) {
        float dot = 0.0f;
        for (std::size_t j = 0; j < _cols; j++)
            dot += _values[i * _cols + j] * g._values[i * _cols + j];

        for (std::size_t j = 0; j < _cols; j++)
            grad->_values[i * _cols + j] =
                _values[i * _cols + j] * (g._values[i * _cols + j] - dot);
    }
    return grad;
}

Matrix& MatrixCpu::randn() {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (std::size_t i = 0; i < numel(); i++) {
        _values[i] = dist(gen);
    }

    return *this;
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

Matrix* MatrixCpu::matsmul(const Matrix& other) const {
    float value = other.values()[0];

    auto* result = new MatrixCpu(_rows, _cols);

    for (std::size_t i = 0; i < numel(); i++) {
        result->_values[i] = _values[i] * value;
    }

    return result;
}

Matrix* MatrixCpu::smatmul(const Matrix& other) const {
    float value = _values[0];

    auto* result = new MatrixCpu(other.rows(), other.cols());

    for (std::size_t i = 0; i < other.numel(); i++) {
        result->_values[i] = other.values()[i] * value;
    }

    return result;
}

Matrix* MatrixCpu::exp() const {
    auto* result = new MatrixCpu(_rows, _cols);

    for (std::size_t i = 0; i < numel(); i++) {
        result->_values[i] = std::exp(_values[i]);
    }

    return result;
}

Matrix* MatrixCpu::exp_backward(const Matrix& upstream_grad, const Matrix& exp_result) const {
    auto* result = new MatrixCpu(_rows, _cols);
    for (std::size_t i = 0; i < numel(); i++) {
        result->_values[i] = upstream_grad.values()[i] * exp_result.values()[i]; 
    }

    return result;
}

Matrix* MatrixCpu::log() const {
    auto* result = new MatrixCpu(_rows, _cols);

    for (std::size_t i = 0; i < numel(); i++) {
        if (_values[i] <= 0) throw std::runtime_error("MatrixCpu::log() : log operation possible only for positive values\n");
        result->_values[i] = std::log(_values[i]);
    }

    return result;
}

Matrix* MatrixCpu::log_backward(const Matrix& upstream_grad) const {
    auto* result = new MatrixCpu(_rows, _cols);

    for (std::size_t i = 0; i < numel(); i++) {
        result->_values[i] = upstream_grad.values()[i] / _values[i];
    }

    return result;
}

float& MatrixCpu::sum() const {
    float sum = 0.0f;
    for (std::size_t i = 0; i < numel(); i++) {
        sum += _values[i];
    }
    return sum;
}

float& MatrixCpu::mean() const {
    float sum = 0.0f;
    for (std::size_t i = 0; i < numel(); i++) {
        sum += _values[i];
    }
    float mean = sum / numel();
    return mean;
}

void MatrixCpu::repr() const {
    for (std::size_t i = 0; i < numel(); i++) {
        if (i % cols() == 0) std::cout << "[";
        std::cout << _values[i];
        if ((i + 1) % cols() == 0) std::cout << "]\n";
        else std::cout << ", ";
    }
}

float* MatrixCpu::values() const { return _values; }
std::size_t MatrixCpu::rows() const { return _rows; }
std::size_t MatrixCpu::cols() const { return _cols; }
std::size_t MatrixCpu::numel() const { return _rows * _cols; }
float* MatrixCpu::at(std::size_t index) { return &_values[index]; }

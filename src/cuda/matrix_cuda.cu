#include <cuda_runtime.h>
#include <cstdint>
#include "tinytorch/cuda/matrix_cuda.cuh"
#include "matrix_cuda.cuh"

__global__ void kernelFillMatrix(float* data, const float fillValue, std::size_t n) {
    std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) data[i] = fillValue;
}

__global__ void kernelAddMatrices(const float* a, const float* b, float* result, std::size_t n) {
    std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < n) result[i] = a[i] + b[i];
}

MatrixCuda::MatrixCuda(std::size_t rows, std::size_t cols)
    : _rows{rows}, _cols{cols}
{
    std::size_t bytes = sizeof(float) * rows * cols;

    cudaError_t error = cudaMalloc(&_values, bytes);
    if (error != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc() : ") + cudaGetErrorString(error));

    error = cudaMemset(_values, 0, bytes);
    if (error != cudaSuccess) throw std::runtime_error(std::string("cudaMemset() : ") + cudaGetErrorString(error));
}

MatrixCuda::MatrixCuda(float fillValue, std::size_t rows, std::size_t cols)
    : _rows{rows}, _cols{cols}
{
    std::size_t n = rows * cols;
    std::size_t bytes = sizeof(float) * n;

    cudaError_t error = cudaMalloc(&_values, bytes);
    if (error != cudaSuccess) throw cudaGetErrorName(error);

    uint64_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;
    kernelFillMatrix<<<blocks, threads>>>(_values, fillValue, n);
}

MatrixCuda::MatrixCuda(const Matrix& other)
    : _rows{other.rows()}, _cols{other.cols()}
{
    std::size_t bytes = sizeof(float) * _rows * _cols;

    if (other.device() == Device::CUDA) {
        cudaError_t error = cudaMemcpy(_values, other.values(), bytes, cudaMemcpyDeviceToDevice);
        if (error != cudaSuccess) throw cudaGetErrorName(error);
    }

    else {
        cudaMemcpy(_values, other.values(), bytes, cudaMemcpyHostToDevice);
    }
}

MatrixCuda::~MatrixCuda() {
    cudaFree(_values);
}

Matrix* MatrixCuda::add(const Matrix& other) const {
    // same-sized matices
    if (_rows == other.rows() && _cols == other.cols()) {
        auto* result = new MatrixCuda(_rows, _cols);

        // for (std::size_t i = 0; i < _rows * _cols; i++) {
        //     result->_values[i] = _values[i] + other.values()[i];
        // }

        return result;
    }
    
    // row vector + col vector
    if (_rows == 1 && other.cols() == 1 && _cols == other.rows()) {
        auto* result = new MatrixCuda(_cols, _cols);

        for (std::size_t i = 0; i < _cols; i++) {
            for (std::size_t j = 0; j < _cols; j++) {
                result->_values[i * _cols + j] += _values[j] + other.values()[i];
            }
        }

        return result;
    }

    // col vector + row vector
    if (_cols == 1 && other.rows() == 1 && _rows == other.cols()) {
        auto* result = new MatrixCuda(_rows, _rows);

        for (std::size_t i = 0; i < _rows; i++) {
            for (std::size_t j = 0; j < _rows; j++) {
                result->_values[i * _rows + j] += _values[i] + other.values()[j];
            }
        }

        return result;
    }

    // matrix + row vector (equal cols)
    if (_cols == other.cols() && other.rows() == 1) {
        auto* result = new MatrixCuda(_rows, _cols);

        for (std::size_t i = 0; i < _rows; i++) {
            for (std::size_t j = 0; j < _cols; j++) {
                result->_values[i * _cols + j] = _values[i * _cols + j] + other.values()[j];
            }
        }

        return result;
    }

    // row vector + matrix (equal cols)
    if (_rows == 1 && _cols == other.cols()) {
        auto* result = new MatrixCuda(other.rows(), other.cols());

        for (std::size_t i = 0; i < other.rows(); i++) {
            for (std::size_t j = 0; j < _cols; j++) {
                result->_values[i * _cols + j] = _values[j] + other.values()[i * _cols + j];
            }
        }

        return result;
    }

    // matrix + col vector (equal rows)
    if (_rows == other.rows() && other.cols() == 1) {
        auto* result = new MatrixCuda(_rows, _cols);

        for (std::size_t i = 0; i < _rows; i++) {
            for (std::size_t j = 0; j < _cols; j++) {
                result->_values[i * _cols + j] = _values[i * _cols + j] + other.values()[i];
            }
        }

        return result;
    }

    // col vector + matrix (equal rows)
    if (_cols == 1 && _rows == other.rows()) {
        auto* result = new MatrixCuda(other.rows(), other.cols());

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
        auto* result = new MatrixCuda(_rows, _cols);
        
        float addValue = other.values()[0];

        for (std::size_t i = 0; i < size(); i++) {
            result->_values[i] = _values[i] + addValue;
        }

        return result;
    }

    // scalar + matrix
    if (size() == 1 && other.size() != 1) {
        auto result = new MatrixCuda(other.rows(), other.cols());
        float addValue = _values[0];

        for (std::size_t i = 0; i < other.size(); i++) {
            result->_values[i] = other.values()[i] + addValue;
        }

        return result;
    }

    throw std::runtime_error("Matrix* add: dimensions do not match. Broadcasting failed\n");
}

Matrix* MatrixCuda::matmul(const Matrix& other) const {

}

Matrix* MatrixCuda::relu() {

}

Matrix* randn() {

}

float* MatrixCuda::values() const { return _values; }
float* MatrixCuda::at(std::size_t index) { &_values[index]; }
std::size_t MatrixCuda::rows() const { return _rows; }
std::size_t MatrixCuda::cols() const { return _cols; }
std::size_t MatrixCuda::size() const { return _rows * _cols; }
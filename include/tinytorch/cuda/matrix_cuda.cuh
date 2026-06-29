#pragma once

#include <stdexcept>
#include <random>
#include "../core/matrix.hpp"

#ifdef __CUDACC__
__global__ void kernelFillMatrix(float* data, float fillValue, std::size_t n);
__global__ void kernelBroadcastAdd(
    float* const a, std::size_t a_rows, std::size_t a_cols,
    float* const b, std::size_t b_rows, std::size_t b_cols,
    float* result, std::size_t out_rows, std::size_t out_cols);
__global__ void kernelMatmul(const float* a, const float* b, float* result, std::size_t N, std::size_t M, std::size_t K);
__global__ void kernelRelu(const float* in, float* out, std::size_t n);
__global__ void kernelReluBackward(const float* input, const float* upstream, float* output, std::size_t n);
__global__ void kernelMatSmul(const float* mat, float value, float* result, std::size_t n);
#endif


class MatrixCuda : public Matrix {
private:
    float* _values;
    std::size_t _rows;
    std::size_t _cols;

public:
    MatrixCuda(std::size_t rows, std::size_t cols);
    MatrixCuda(float fillValue, std::size_t rows, std::size_t cols);
    MatrixCuda(const Matrix& other);
    ~MatrixCuda();

    Matrix* add(const Matrix& other) const override;
    Matrix* matmul(const Matrix& other) const override;
    Matrix* relu() const override;
    Matrix& randn() override;
    Matrix* transpose() override;
    Matrix* relu_backward(const Matrix& upstream_grad) const override;
    Matrix* matsmul(const Matrix& other) const override;
    Matrix* smatmul(const Matrix& other) const override;
    Matrix* mul(const Matrix& other) const override;


    float* values() const override;
    float* at(std::size_t index) override;
    std::size_t rows() const override;
    std::size_t cols() const override;
    std::size_t numel() const override;
    Device device() const override { return Device::CUDA; }
    void repr() const override;
};
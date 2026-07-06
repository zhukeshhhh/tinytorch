#pragma once

#include <stdexcept>
#include <random>
#include "../core/matrix.hpp"

#ifdef __CUDACC__
__global__ void fill_mat_kernel(float* data, float fillValue, std::size_t n);

__global__ void add_broadcast_kernel(
    float* const a, std::size_t a_rows, std::size_t a_cols,
    float* const b, std::size_t b_rows, std::size_t b_cols,
    float* result, std::size_t out_rows, std::size_t out_cols);

__global__ void matmul_kernel(const float* a, const float* b, float* result, std::size_t N, std::size_t M, std::size_t K);
__global__ void relu_kernel(const float* in, float* out, std::size_t n);
__global__ void relu_backward_kernel(const float* input, const float* upstream, float* output, std::size_t n);
__global__ void transpose_kernel(const float* input, float* output, std::size_t rows, std::size_t cols);
__global__ void mat_scalar_mul_kernel(const float* mat, float value, float* result, std::size_t n);
__global__ void mul_kernel(const float* a, const float* b, float* result, std::size_t n);

__global__ void softmax_forward_kernel(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int cols);

__global__ void softmax_backward_kernel(
    const float* __restrict__ s,
    const float* __restrict__ g,
    float*       __restrict__ dx, 
    int cols);

__global__ void exp_kernel(const float* in, float* out, std::size_t n);
__global__ void exp_backward_kernel(const float* upstream_grad, const float* exp_result, float* out, std::size_t n);
__global__ void log_kernel(const float* in, float* out, std::size_t n);
__global__ void log_backward_kernel(const float* upstream_grad, const float* self, float* out, std::size_t n);
__global__ void sum_kernel(const float* input, float* out, std::size_t n);
__global__ void sdg_step_kernel(float* a, float* b, std::size_t n, float learning_rate, float batch_size);
__global__ void reduce_to_kernel(
    const float* in, std::size_t in_rows, std::size_t in_cols,
    float* out, std::size_t out_rows, std::size_t out_cols);
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
    MatrixCuda(const std::vector<float> vector_1d);
    MatrixCuda(const std::vector<std::vector<float>> vector_2d);
    ~MatrixCuda();

    Matrix* add(const Matrix& other) const override;
    Matrix* matmul(const Matrix& other) const override;
    Matrix* relu() const override;
    Matrix& randn(unsigned long long seed) override;
    Matrix* transpose() override;
    Matrix* relu_backward(const Matrix& upstream_grad) const override;
    Matrix* matsmul(const Matrix& other) const override;
    Matrix* smatmul(const Matrix& other) const override;
    Matrix* mul(const Matrix& other) const override;
    Matrix* softmax() const override;
    Matrix* softmax_backward(const Matrix& other) const override;
    Matrix* exp() const override;
    Matrix* exp_backward(const Matrix& upstream_grad, const Matrix& exp_result) const override;
    Matrix* log() const override;
    Matrix* log_backward(const Matrix& upstream_grad) const override;

    float sum() const override;
    float mean() const override;

    void sdg_step(float& learning_rate, float& batch_size, Matrix* grad) override;
    std::vector<std::size_t> argmax() const override;

    float scalar_value() const override;
    std::vector<float> to_host_vector() const override;

    Matrix* reduce_to(std::size_t target_rows, std::size_t target_cols) const override;

    float* values() const override;
    float* at(std::size_t index) override;
    std::size_t rows() const override;
    std::size_t cols() const override;
    std::size_t numel() const override;
    Device device() const override { return Device::CUDA; }
    void repr() const override;
};
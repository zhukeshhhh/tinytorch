#pragma once

#include "../core/matrix.hpp"
#include "../core/device.hpp"

class MatrixCpu : public Matrix {
private:
    float* _values;
    std::size_t _rows;
    std::size_t _cols;

public:
    MatrixCpu(std::size_t rows, std::size_t cols);
    MatrixCpu(float fillValue, std::size_t rows, std::size_t cols);
    MatrixCpu(const Matrix& other);
    MatrixCpu(const std::vector<float> vector_1d);
    MatrixCpu(const std::vector<std::vector<float>> vector_2d);
    ~MatrixCpu();

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
    Matrix* softmax_backward(const Matrix& upstream_grad) const override;
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
    Device device() const override { return Device::CPU; }
    void repr() const override;
};
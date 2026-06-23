#pragma once

#include <stdexcept>
#include "../core/matrix.hpp"


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
    Matrix* relu() override;
    Matrix* randn() override;
    Matrix* transpose() override;
    Matrix* relu_backward(const Matrix& upstream_grad) const override;


    float* values() const override;
    float* at(std::size_t index) override;
    std::size_t rows() const override;
    std::size_t cols() const override;
    std::size_t size() const override;
    Device device() const override { return Device::CUDA; }

    void kernelFillMatrix(float* data, float fillValue, std::size_t n);
};
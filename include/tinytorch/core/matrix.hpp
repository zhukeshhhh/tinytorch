#pragma once

#include "device.hpp"
#include <cstdint>
#include <cstddef>
#include <vector>

class Matrix {
public:
    virtual ~Matrix() = default;

    virtual Matrix* add(const Matrix& other) const = 0;
    virtual Matrix* matmul(const Matrix& other) const = 0;
    virtual Matrix* relu() const = 0;
    virtual Matrix& randn() = 0;
    virtual Matrix* transpose() = 0;
    virtual Matrix* relu_backward(const Matrix& upstream_grad) const = 0;
    virtual Matrix* matsmul(const Matrix& other) const = 0;
    virtual Matrix* smatmul(const Matrix& other) const = 0;
    virtual Matrix* mul(const Matrix& other) const = 0;
    virtual Matrix* softmax() const = 0;
    virtual Matrix* softmax_backward(const Matrix& upstream_grad) const = 0;
    virtual Matrix* exp() const = 0;
    virtual Matrix* exp_backward(const Matrix& upstream_grad, const Matrix& exp_result) const = 0;
    virtual Matrix* log() const = 0;
    virtual Matrix* log_backward(const Matrix& upstream_grad) const = 0;

    virtual float sum() const = 0;
    virtual float mean() const = 0;

    virtual void sdg_step(float& learning_rate, float& batch_size, Matrix* grad) = 0;

    virtual float scalar_value() const = 0;

    virtual float* values() const = 0;
    virtual float* at(std::size_t index) = 0;
    virtual std::size_t rows() const = 0;
    virtual std::size_t cols() const = 0;
    virtual std::size_t numel() const = 0;
    virtual Device device() const = 0;
    virtual void repr() const = 0;
};
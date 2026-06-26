#pragma once

#include "device.hpp"
#include <cstdint>

class Matrix {
public:
    virtual ~Matrix() = default;

    virtual Matrix* add(const Matrix& other) const = 0;
    virtual Matrix* matmul(const Matrix& other) const = 0;
    virtual Matrix* relu() const = 0;
    virtual Matrix& randn() = 0;
    virtual Matrix* transpose() = 0;
    virtual Matrix* relu_backward(const Matrix& upstream_grad) const = 0;


    virtual float* values() const = 0;
    virtual float* at(std::size_t index) = 0;
    virtual std::size_t rows() const = 0;
    virtual std::size_t cols() const = 0;
    virtual std::size_t numel() const = 0;
    virtual Device device() const = 0;
    virtual void repr() const = 0;
};
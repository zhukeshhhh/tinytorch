#pragma once

#include <cstdint>

class Matrix {
public:
    virtual ~Matrix() = default;

    virtual Matrix* add(const Matrix& other) const = 0;
    virtual Matrix* matmul(const Matrix& other) const = 0;
    virtual Matrix* relu() = 0;

    virtual float* values() const = 0;
    virtual std::size_t rows() const = 0;
    virtual std::size_t cols() const = 0;
    virtual std::size_t size() const = 0;
};
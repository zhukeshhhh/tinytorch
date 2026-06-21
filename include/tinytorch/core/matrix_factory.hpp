#pragma once

#include <cstdint>
#include "matrix.hpp"

class MatrixFactory {
public:
    static Matrix* create(std::size_t rows, std::size_t cols, Device device);
    static Matrix* create(float fillValue, std::size_t rows, std::size_t cols, Device device);
    static Matrix* copy(const Matrix& src, Device device);
};
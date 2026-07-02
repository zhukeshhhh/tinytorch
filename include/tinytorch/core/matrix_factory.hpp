#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include "matrix.hpp"

class MatrixFactory {
public:
    static Matrix* create(std::size_t rows, std::size_t cols, Device device);
    static Matrix* create(float fillValue, std::size_t rows, std::size_t cols, Device device);
    static Matrix* create(std::vector<float> vector_1d, Device device);
    static Matrix* create(std::vector<std::vector<float>> vector_2d, Device device);

    static Matrix* copy(const Matrix& src, Device device);
};
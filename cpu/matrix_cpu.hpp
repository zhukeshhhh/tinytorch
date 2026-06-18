#pragma once

#include "../core/matrix.hpp"

class MatrixCpu : public Matrix {
private:
    float* _values;
    std::size_t _rows;
    std::size_t _cols;

public:
    MatrixCpu(std::size_t rows, std::size_t cols);
    MatrixCpu(float fillValue, std::size_t rows, std::size_t cols);
    MatrixCpu(const Matrix& other);
    ~MatrixCpu();

    Matrix* add(const Matrix& other) const override;
    Matrix* matmul(const Matrix& other) const override;
    Matrix* relu() override;

    float* values() const override;
    std::size_t rows() const override;
    std::size_t cols() const override;
    std::size_t size() const override;
};
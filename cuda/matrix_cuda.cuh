#include "../core/matrix.hpp"


class MatrixCuda : public Matrix {
private:
    float* _values;
    std::size_t _rows;
    std::size_t _cols;

public:
    MatrixCuda(std::size_t rows, std::size_t cols);
    MatrixCuda(float value, std::size_t rows, std::size_t cols);
    MatrixCuda(const Matrix& other);
    ~MatrixCuda();

    Matrix* add(const Matrix& other) const override;
    Matrix* matmul(const Matrix& other) const override;
    Matrix* relu() override;

    float* values() const override;
    std::size_t rows() const override;
    std::size_t cols() const override;
    std::size_t size() const override;
};
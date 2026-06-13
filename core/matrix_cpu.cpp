#include <memory>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <vector>
#include "matrix.cpp"

class MatrixCpu : public Matrix {
public:
    float* values;
    std::size_t rows;
    std::size_t cols;
    std::vector<std::size_t> mshape;

    MatrixCpu(
        std::size_t rows,
        std::size_t cols
    ) : rows{rows},
        cols{cols}
    {
        mshape = {rows, cols};
        values = new float[rows * cols]{};
    }

    MatrixCpu(std::size_t size)
        : rows{1}, cols{size}
    {
        values = new float[size]{}; // zeros
    }

    MatrixCpu(const MatrixCpu& other)
        : rows{other.rows}, cols{other.cols}
    {
        values = new float[rows * cols]; // no values
        std::copy(other.values, other.values + (cols * rows), values);
    }

    std::size_t getSize() {
        return rows * cols;
    }

    MatrixCpu::add(MatrixCpu& other) {
        
    }
};

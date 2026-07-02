#include "tinytorch/core/matrix_factory.hpp"
#include "tinytorch/core/device.hpp"
#include "tinytorch/core/matrix.hpp"
#include "tinytorch/cpu/matrix_cpu.hpp"
#include "tinytorch/cuda/matrix_cuda.cuh"

Matrix* MatrixFactory::create(std::size_t rows, std::size_t cols, Device device) {
    switch(device) {
        case Device::CPU: return new MatrixCpu(rows, cols);

        #ifdef USE_CUDA
        case Device::CUDA: return new MatrixCuda(rows, cols);
        #endif

        default: throw std::runtime_error("MatrixFactory::create(rows, cols, device)\n");
    }
}

Matrix* MatrixFactory::create(float fillvalue, std::size_t rows, std::size_t cols, Device device) {
    switch(device) {
        case Device::CPU: return new MatrixCpu(fillvalue, rows, cols);

        #ifdef USE_CUDA
        case Device::CUDA: return new MatrixCuda(fillvalue, rows, cols);
        #endif

        default: throw std::runtime_error("MatrixFactory::create(fillValue, rows, cols, device)\n");
    }
}

Matrix* MatrixFactory::create(std::vector<float> vector_1d, Device device) {
    switch(device) {
        case Device::CPU: return new MatrixCpu(vector_1d);

        #ifdef USE_CUDA
        case Device::CUDA: return new MatrixCuda(vector_1d);
        #endif

        default: throw std::runtime_error("MatrixFactory::create(std::vector<float> data)\n");
    }
}

Matrix* MatrixFactory::create(std::vector<std::vector<float>> vector_2d, Device device) {
    switch(device) {
        case Device::CPU: return new MatrixCpu(vector_2d);

        #ifdef USE_CUDA
        case Device::CUDA: return new MatrixCuda(vector_2d);
        #endif

        default: throw std::runtime_error("MatrixFactory::create(std::vector<float> data)\n");
    }
}

Matrix* MatrixFactory::copy(const Matrix& src, Device device) {
    switch(device) {
        case Device::CPU: return new MatrixCpu(src);

        #ifdef USE_CUDA
        case Device::CUDA: return new MatrixCuda(src);
        #endif

        default: throw std::runtime_error("MatrixFactory::copy(src, device)\n");
    }
}
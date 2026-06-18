#include "device.hpp"
#include "../cpu/matrix_cpu.hpp"
#include "../cuda/matrix_cuda.cuh"


Matrix* MatrixFactory::create(std::size_t rows, std::size_t cols, Device device) {
    switch(device) {
        case Device::CPU: return new MatrixCpu(rows, cols);
        case Device::CUDA: return new MatrixCuda(rows, cols);
    }
}
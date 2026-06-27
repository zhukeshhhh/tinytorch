#include <cuda_runtime.h>
#include <curand.h>
#include <cstdint>
#include <iostream>
#include "tinytorch/cuda/matrix_cuda.cuh"

#define TILE_SIZE 32

#define CUDA_CALL(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        printf("CUDA error %d at %s:%d %s\n", err, __FILE__, __LINE__, cudaGetErrorString(err)); \
        abort(); \
    } \
} while(0)

#define CURAND_CALL(x) do { \
    curandStatus_t err = (x); \
    if (err != CURAND_STATUS_SUCCESS) { \
        printf("CURAND error %d at %s:%d\n", err, __FILE__, __LINE__); \
        abort(); \
    } \
} while(0)

__global__ void kernelFillMatrix(float* data, float fillValue, std::size_t n) {
    std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) data[i] = fillValue;
}

__global__ void kernelBroadcastAdd(
    float* const a, std::size_t a_rows, std::size_t a_cols,
    float* const b, std::size_t b_rows, std::size_t b_cols,
    float* result, std::size_t out_rows, std::size_t out_cols
)
{
    std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= out_rows * out_cols) return;

    std::size_t i = idx / out_cols;
    std::size_t j = idx % out_cols;

    std::size_t a_idx = (i % a_rows) * a_cols + (j % a_cols);
    std::size_t b_idx = (i % b_rows) * b_cols + (j % b_cols);

    result[idx] = a[a_idx] + b[b_idx];
}

__global__ void kernelMatmul(const float* a, const float* b, float* result, std::size_t N, std::size_t M, std::size_t K) {
    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE + 1];

    std::size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    std::size_t col = blockDim.x * blockIdx.x + threadIdx.x;

    std::size_t local_row = threadIdx.y;
    std::size_t local_col = threadIdx.x;

    float partial_sum = 0;

    for (uint64_t phase = 0; phase < (K + TILE_SIZE - 1) / TILE_SIZE; phase++) {
        if (row < N && phase * TILE_SIZE + local_col < K) {
            shared_a[local_row][local_col] = a[row * K + phase * TILE_SIZE + local_col];
        }
        else {
            shared_a[local_row][local_col] = 0.0f;
        }

        if (col < M && phase * TILE_SIZE + local_row < K) {
            shared_b[local_row][local_col] = b[(phase * TILE_SIZE + local_row) * M + col];
        }
        else {
            shared_b[local_row][local_col] = 0.0f;
        }
        __syncthreads();

        for (std::size_t k = 0; k < TILE_SIZE; k++) {
            partial_sum += shared_a[local_row][k] * shared_b[k][local_col];
        }
        __syncthreads();
    }
    
    if (row < N && col < M) {
        result[row * M + col] = partial_sum;
    }
}

__global__ void kernelRelu(const float* input, float* output, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) output[i] = input[i] > 0 ? input[i] : 0.0f;
    else return;
}

__global__ void kernelReluBackward(const float* input, const float* upstream, float* output, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) output[i] = input[i] > 0.0f ? upstream[i] : 0.0f;
    else return;    
}

__global__ void kernelTranspose(const float* input, float* output, std::size_t rows, std::size_t cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    std::size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;
    std::size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (col < cols && row < rows)
        tile[threadIdx.y][threadIdx.x] = input[row * cols + col];

    __syncthreads();

    col = blockIdx.y * TILE_SIZE + threadIdx.x;
    row = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (col < rows && row < cols)
        output[row * rows + col] = tile[threadIdx.x][threadIdx.y];
}

MatrixCuda::MatrixCuda(std::size_t rows, std::size_t cols)
    : _rows{rows}, _cols{cols}
{
    std::size_t bytes = sizeof(float) * rows * cols;

    CUDA_CALL(cudaMalloc(&_values, bytes));

    CUDA_CALL(cudaMemset(_values, 0, bytes));
}

MatrixCuda::MatrixCuda(float fillValue, std::size_t rows, std::size_t cols)
    : _rows{rows}, _cols{cols}
{
    std::size_t n = rows * cols;
    std::size_t bytes = sizeof(float) * n;

    CUDA_CALL(cudaMalloc(&_values, bytes));

    uint64_t THREADS = 256;
    uint64_t BLOCKS = (n + THREADS - 1) / THREADS;
    kernelFillMatrix<<<BLOCKS, THREADS>>>(_values, fillValue, n);
    CUDA_CALL(cudaDeviceSynchronize());
}

MatrixCuda::MatrixCuda(const Matrix& other)
    : _rows{other.rows()}, _cols{other.cols()}
{
    std::size_t bytes = sizeof(float) * numel();
    
    CUDA_CALL(cudaMalloc(&_values, bytes));

    if (other.device() == Device::CUDA) {
        CUDA_CALL(cudaMemcpy(_values, other.values(), bytes, cudaMemcpyDeviceToDevice));
    }

    else {
        CUDA_CALL(cudaMemcpy(_values, other.values(), bytes, cudaMemcpyHostToDevice));
    }
}

MatrixCuda::~MatrixCuda() {
    CUDA_CALL(cudaFree(_values));
}

Matrix* MatrixCuda::add(const Matrix& other) const {
    std::size_t out_rows = std::max(_rows, other.rows());
    std::size_t out_cols = std::max(_cols, other.cols());

    if (_rows != other.rows() && _rows != 1 && other.rows() != 1)
        throw std::runtime_error("MatrixCuda::add() : incompatible rows\n");

    if (_cols != other.cols() && _cols != 1 && other.cols() != 1)
        throw std::runtime_error("MatrixCuda::add() : incompatible cols\n");

    auto* result = new MatrixCuda(out_rows, out_cols);

    std::size_t n = out_rows * out_cols;

    uint64_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;

    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    kernelBroadcastAdd<<<BLOCKS, THREADS>>>(
        _values, _rows, _cols,
        other.values(), other.rows(), other.cols(),
        result->_values, out_rows, out_cols
    );

    CUDA_CALL(cudaDeviceSynchronize());

    return result;
}

Matrix* MatrixCuda::matmul(const Matrix& other) const {
    if (_cols != other.rows())
        throw std::runtime_error("Matrix* matmul: dimensions do not match\n");

    std::size_t N = _rows;
    std::size_t M = other.cols();
    std::size_t K = _cols;

    auto* result = new MatrixCuda(N, M);

    dim3 THREADS(TILE_SIZE, TILE_SIZE);
    dim3 BLOCKS((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    kernelMatmul<<<BLOCKS, THREADS>>>(_values, other.values(), result->_values, N, M, K);

    CUDA_CALL(cudaDeviceSynchronize());

    return result;
}

Matrix* MatrixCuda::relu() const {
    auto* result = new MatrixCuda(_rows, _cols);

    std::size_t n = numel();
    uint64_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;

    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    kernelRelu<<<BLOCKS, THREADS>>>(_values, result->_values, numel());
    CUDA_CALL(cudaDeviceSynchronize());

    return result;
}

Matrix* MatrixCuda::relu_backward(const Matrix& upstream_grad) const {
    auto* result = new MatrixCuda(_rows, _cols);
    std::size_t n = numel();
    uint64_t threads = 256;
    uint64_t blocks  = (n + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    kernelReluBackward<<<BLOCKS, THREADS>>>(_values, upstream_grad.values(), result->_values, n);
    CUDA_CALL(cudaDeviceSynchronize());
    return result;
}

Matrix& MatrixCuda::randn() {
    curandGenerator_t gen;

    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    CURAND_CALL(curandSetStream(gen, 0));

    std::size_t n = numel();
    std::size_t n_even = n + (n % 2);

    float* buf = _values;
    if (n_even != n)
        CUDA_CALL(cudaMalloc(&buf, n_even * sizeof(float)));

    CURAND_CALL(curandGenerateNormal(gen, buf, n_even, 0.0f, 1.0f));

    if (n_even != n) {
        CUDA_CALL(cudaMemcpy(_values, buf, n * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CALL(cudaFree(buf));
    }

    CURAND_CALL(curandDestroyGenerator(gen));
    return *this;
}

Matrix* MatrixCuda::transpose() {
    auto* result = new MatrixCuda(_cols, _rows);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (_cols + TILE_SIZE - 1) / TILE_SIZE,
        (_rows + TILE_SIZE - 1) / TILE_SIZE
    );

    kernelTranspose<<<blocks, threads>>>(_values, result->_values, _rows, _cols);
    CUDA_CALL(cudaDeviceSynchronize());

    return result;
}

float* MatrixCuda::values() const { return _values; }
float* MatrixCuda::at(std::size_t index) { return &_values[index]; }
std::size_t MatrixCuda::rows() const { return _rows; }
std::size_t MatrixCuda::cols() const { return _cols; }
std::size_t MatrixCuda::numel() const { return _rows * _cols; }

void MatrixCuda::repr() const {
    float* host = new float[numel()];
    std::size_t bytes = sizeof(float) * numel();
    
    CUDA_CALL(cudaMemcpy(host, _values, bytes, cudaMemcpyDeviceToHost));
    
    for (std::size_t i = 0; i < numel(); i++) {
        if (i % cols() == 0) std::cout << "[";
        std::cout << host[i];
        if ((i + 1) % cols() == 0) std::cout << "]\n";
        else std::cout << ", ";
    }

    delete[] host;
}
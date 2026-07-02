#include <cuda_runtime.h>
#include <curand.h>
#include <cstdint>
#include <iostream>
#include "tinytorch/cuda/matrix_cuda.cuh"

#define TILE_SIZE 32
#define COARSE_FACTOR 32

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

__global__ void fill_mat_kernel(float* data, float fillValue, std::size_t n) {
    std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) data[i] = fillValue;
}

__global__ void add_broadcast_kernel(
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

__global__ void matmul_kernel(const float* a, const float* b, float* result, std::size_t N, std::size_t M, std::size_t K) {
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

__global__ void relu_kernel(const float* input, float* output, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) output[i] = input[i] > 0 ? input[i] : 0.0f;
}

__global__ void relu_backward_kernel(const float* input, const float* upstream, float* output, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) output[i] = input[i] > 0.0f ? upstream[i] : 0.0f;
}

__global__ void transpose_kernel(const float* input, float* output, std::size_t rows, std::size_t cols) {
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

__global__ void mat_scalar_mul_kernel(const float* mat, float value, float* result, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) result[i] = mat[i] * value;
}

__global__ void mul_kernel(const float* a, const float* b, float* result, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) result[i] = a[i] * b[i];
}

__global__ void softmax_forward_kernel(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int cols)
{
    extern __shared__ float shmem[];

    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    const float* row_in  = in  + row * cols;
    float*       row_out = out + row * cols;

    float thread_max = -INFINITY;
    for (int j = tid; j < cols; j += bdim)
        thread_max = fmaxf(thread_max, row_in[j]);

    shmem[tid] = thread_max;
    __syncthreads();
    for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            shmem[tid] = fmaxf(shmem[tid], shmem[tid + stride]);
        __syncthreads();
    }
    const float row_max = shmem[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int j = tid; j < cols; j += bdim) {
        const float val = expf(row_in[j] - row_max);
        row_out[j]   = val;
        thread_sum  += val;
    }

    shmem[tid] = thread_sum;
    __syncthreads();
    for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }
    const float row_sum = shmem[0];
    __syncthreads();

    for (int j = tid; j < cols; j += bdim)
        row_out[j] /= row_sum;
}


__global__ void softmax_backward_kernel(
    const float* __restrict__ s,
    const float* __restrict__ g,
    float*       __restrict__ dx, 
    int cols)
{
    extern __shared__ float shmem[];

    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int bdim = blockDim.x;

    const float* s_row  = s  + row * cols;
    const float* g_row  = g  + row * cols;
    float*       dx_row = dx + row * cols;

    float thread_dot = 0.0f;
    for (int j = tid; j < cols; j += bdim)
        thread_dot += s_row[j] * g_row[j];

    shmem[tid] = thread_dot;
    __syncthreads();
    for (int stride = bdim >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            shmem[tid] += shmem[tid + stride];
        __syncthreads();
    }
    const float dot = shmem[0];
    __syncthreads();

    for (int j = tid; j < cols; j += bdim)
        dx_row[j] = s_row[j] * (g_row[j] - dot);
}


__global__ void exp_kernel(const float* in, float* out, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = std::exp(in[i]);
}

__global__ void exp_backward_kernel(const float* upstream_grad, const float* exp_result, float* out, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = upstream_grad[i] * exp_result[i];
}

__global__ void log_kernel(const float* in, float* out, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = std::log(in[i]);
}

__global__ void log_backward_kernel(const float* upstream_grad, const float* self, float* out, std::size_t n) {
    std::size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) out[i] = upstream_grad[i] / self[i];
}


__global__ void sum_kernel(const float* input, float* out, std::size_t n) {
    extern __shared__ float input_s[];

    uint64_t segment_start = blockIdx.x * COARSE_FACTOR * blockDim.x;
    uint64_t index = segment_start + threadIdx.x;

    float sum = 0.0f;
    for (std::size_t i = 0; i < COARSE_FACTOR; ++i) {
        if (index + i * blockDim.x < n)
            sum += input[index + i * blockDim.x];
    }

    input_s[threadIdx.x] = sum;

    for (std::size_t stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride)
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
    }

    if (threadIdx.x == 0) {
        atomicAdd(out, input_s[0]);
    }
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
    fill_mat_kernel<<<BLOCKS, THREADS>>>(_values, fillValue, n);
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

    add_broadcast_kernel<<<BLOCKS, THREADS>>>(
        _values, _rows, _cols,
        other.values(), other.rows(), other.cols(),
        result->_values, out_rows, out_cols
    );

    CUDA_CALL(cudaDeviceSynchronize());
    std::cout << "CUDA::add() just finished!\n";
    return result;
}

Matrix* MatrixCuda::matmul(const Matrix& other) const {

    if (other.numel() == 1) {
        return matsmul(other);
    }

    if (numel() == 1) {
        return smatmul(other);
    }

    if (_cols != other.rows())
        throw std::runtime_error("Matrix* matmul: dimensions do not match\n");

    std::size_t N = _rows;
    std::size_t M = other.cols();
    std::size_t K = _cols;

    auto* result = new MatrixCuda(N, M);

    dim3 THREADS(TILE_SIZE, TILE_SIZE);
    dim3 BLOCKS((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<BLOCKS, THREADS>>>(_values, other.values(), result->_values, N, M, K);

    CUDA_CALL(cudaDeviceSynchronize());

    return result;
}

Matrix* MatrixCuda::mul(const Matrix& other) const {
    auto* result = new MatrixCuda(_rows, _cols);
    
    std::size_t n = numel();
    uint64_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;

    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    mul_kernel<<<BLOCKS, THREADS>>>(_values, other.values(), result->_values, n);
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

    relu_kernel<<<BLOCKS, THREADS>>>(_values, result->_values, numel());
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
    relu_backward_kernel<<<BLOCKS, THREADS>>>(_values, upstream_grad.values(), result->_values, n);
    CUDA_CALL(cudaDeviceSynchronize());
    return result;
}

static int softmax_block_dim(std::size_t n) {
    std::size_t bdim = 1;
    while (bdim < n) bdim <<= 1;
    return bdim < 1024 ? bdim : 1024;
}

Matrix* MatrixCuda::softmax() const {
    auto* result = new MatrixCuda(_rows, _cols);

    const std::size_t bdim  = softmax_block_dim(static_cast<std::size_t>(_cols));
    const size_t sh = bdim * sizeof(float);

    softmax_forward_kernel<<<_rows, bdim, sh>>>(_values, result->_values, _cols);
    CUDA_CALL(cudaDeviceSynchronize());

    return result;
}

Matrix* MatrixCuda::softmax_backward(const Matrix& upstream_grad) const {
    const auto& g = static_cast<const MatrixCuda&>(upstream_grad);
    auto* result = new MatrixCuda(_rows, _cols);

    const int bdim  = softmax_block_dim(static_cast<int>(_cols));
    const size_t sh = bdim * sizeof(float);

    softmax_backward_kernel<<<_rows, bdim, sh>>>(
        _values, g._values, result->_values, _cols);
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

    transpose_kernel<<<blocks, threads>>>(_values, result->_values, _rows, _cols);
    CUDA_CALL(cudaDeviceSynchronize());

    return result;
}

Matrix* MatrixCuda::matsmul(const Matrix& other) const {
    auto* result = new MatrixCuda(_rows, _cols);
    float* value = new float;
    CUDA_CALL(cudaMemcpy(value, other.values(), sizeof(float), cudaMemcpyDeviceToHost));

    std::size_t n = numel();
    uint64_t threads = 256;
    uint64_t blocks  = (n + threads - 1) / threads;

    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    mat_scalar_mul_kernel<<<BLOCKS, THREADS>>>(values(), *value, result->_values, n);
    CUDA_CALL(cudaDeviceSynchronize());
    delete value;
    return result;
}

Matrix* MatrixCuda::smatmul(const Matrix& other) const {
    auto* result = new MatrixCuda(other.rows(), other.cols());
    float* value = new float;
    CUDA_CALL(cudaMemcpy(value, _values, sizeof(float), cudaMemcpyDeviceToHost));

    std::size_t n = other.numel();
    uint64_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;

    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    mat_scalar_mul_kernel<<<BLOCKS, THREADS>>>(other.values(), *value, result->_values, n);
    CUDA_CALL(cudaDeviceSynchronize());
    delete value;
    return result;
}


Matrix* MatrixCuda::exp() const {
    auto* result = new MatrixCuda(_rows, _cols);

    std::size_t n = numel();
    uint64_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;

    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    exp_kernel<<<BLOCKS, THREADS>>>(_values, result->_values, numel());
    CUDA_CALL(cudaDeviceSynchronize());

    return result;
}

Matrix* MatrixCuda::exp_backward(const Matrix& upstream_grad, const Matrix& exp_result) const {
    auto* result = new MatrixCuda(_rows, _cols);
    std::size_t n = numel();
    uint64_t threads = 256;
    uint64_t blocks  = (n + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    exp_backward_kernel<<<BLOCKS, THREADS>>>(upstream_grad.values(), exp_result.values(), result->_values, n);
    CUDA_CALL(cudaDeviceSynchronize());
    return result;
}


Matrix* MatrixCuda::log() const {
    auto* result = new MatrixCuda(_rows, _cols);

    std::size_t n = numel();
    uint64_t threads = 256;
    uint64_t blocks = (n + threads - 1) / threads;

    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    log_kernel<<<BLOCKS, THREADS>>>(_values, result->_values, numel());
    CUDA_CALL(cudaDeviceSynchronize());

    return result;
}


Matrix* MatrixCuda::log_backward(const Matrix& upstream_grad) const {
    auto* result = new MatrixCuda(_rows, _cols);
    std::size_t n = numel();
    uint64_t threads = 256;
    uint64_t blocks  = (n + threads - 1) / threads;
    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);
    log_backward_kernel<<<BLOCKS, THREADS>>>(upstream_grad.values(), _values, result->_values, n);
    CUDA_CALL(cudaDeviceSynchronize());
    return result;
}


float& MatrixCuda::sum() const {
    float* sum;
    CUDA_CALL(cudaMalloc(&sum, sizeof(float)));
    uint64_t threads = 256;
    uint64_t blocks = (numel() + threads - 1) / threads;

    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    sum_kernel<<<BLOCKS, THREADS>>>(_values, sum, numel());
    CUDA_CALL(cudaDeviceSynchronize());

    return *sum;
}

float& MatrixCuda::mean() const {
    float* sum;
    CUDA_CALL(cudaMalloc(&sum, sizeof(float)));
    uint64_t threads = 256;
    uint64_t blocks = (numel() + threads - 1) / threads;

    dim3 THREADS(threads);
    dim3 BLOCKS(blocks);

    sum_kernel<<<BLOCKS, THREADS>>>(_values, sum, numel());
    CUDA_CALL(cudaDeviceSynchronize());

    float* h_sum;
    CUDA_CALL(cudaMemcpy(h_sum, sum, sizeof(float), cudaMemcpyDeviceToHost));
    float mean = *h_sum / numel();

    return mean;
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
# tinytorch — local usage tutorial

This guide walks you through building tinytorch on your machine, running the demo, and writing your own programs with the current API.

---

## Prerequisites

Install a C++20 toolchain and CMake before you begin.

| Platform | Suggested setup |
|----------|-----------------|
| **Linux (Fedora/RHEL)** | `sudo dnf install gcc-c++ cmake` |
| **Linux (Debian/Ubuntu)** | `sudo apt install build-essential cmake` |
| **macOS** | Xcode Command Line Tools + `brew install cmake` |
| **Windows** | Visual Studio 2019+ with C++ workload + [CMake](https://cmake.org/download/) |

Verify your versions:

```bash
g++ --version    # need GCC for C++20
cmake --version  # need 3.20+
```

For the CUDA backend you additionally need the CUDA Toolkit (12.x or 13.x) and a host compiler that nvcc supports:

```bash
nvcc --version
g++ --version
```

---

## 1. Clone and build

### CPU-only (default)

```bash
git clone https://github.com/zhukeshhhh/tinytorch.git
cd tinytorch
mkdir build && cd build
cmake ..
make
```

This generates the executable at `build/tinytorch` (relative to the project root).

### With CUDA backend

Pass `-DUSE_CUDA=ON` and tell CMake which host compiler to hand to nvcc:

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=path/to/compatible/c++/compiler
make
```

When CUDA initialises correctly, the cmake output will include a line like:

```
-- The CUDA compiler identification is NVIDIA 13.2.78 with host compiler GNU 15.2.1
```

If you only see C and CXX lines, CUDA was not initialised — check your CUDA Toolkit installation.

### Rebuild after changes

```bash
cd build    # skip if you are already there
make
```

`make` only recompiles files that changed. A full clean rebuild is only needed when switching compilers or CMake flags:

```bash
cd ..
rm -rf build
mkdir build && cd build
cmake .. [flags]
make
```

---

## 2. Run the demo

```bash
./build/tinytorch        # from project root
# or
cd build && ./tinytorch  # from inside build/
```

---

## 3. Tensor API reference

All factory methods return `std::shared_ptr<Tensor>`.

### 3.1 Creating tensors

```cpp
// 1×1 scalar
auto s = Tensor::scalar(3.14f, Device::CPU, false, "s");

// M×N zero matrix
auto z = Tensor::zeros(3, 4, Device::CPU, false, "z");

// M×N filled with a constant
auto f = Tensor::full(0.5f, 2, 3, Device::CPU, false, "f");

// 1×N row vector, standard-normal random values
auto r1 = Tensor::randn(5, Device::CPU, false, "r1", 0);

// M×N matrix, standard-normal random values
auto r2 = Tensor::randn(3, 4, Device::CPU, false, "r2", 0);
```

The last three arguments to every factory method are always:

```cpp
Tensor::method(..., Device device, bool requires_grad, std::string label)
```

Set `requires_grad = true` when the tensor should participate in the autograd graph.

To create a tensor on the GPU, pass `Device::CUDA` — the library selects `MatrixCuda` internally:

```cpp
auto g = Tensor::randn(4, 4, Device::CUDA, true, "g", 0);
```

### 3.2 Operations

Binary operators are defined on the pointed-to `Tensor` object, so dereference the left-hand shared pointer with `*`:

```cpp
auto a = Tensor::full(1.0f, 3, 3, Device::CPU, false, "a");
auto b = Tensor::full(2.0f, 3, 1, Device::CPU, false, "b");

auto c = (*a) + b;   // broadcasting add  → 3×3
auto d = (*a) * b;   // matrix multiply   → 3×1
auto e = a->relu();  // ReLU              → 3×3  (unary, no dereference needed)
```

Both operands of `+` and `*` must be on the same device.

### 3.3 Inspecting tensors

```cpp
t->rows();       // number of rows
t->cols();       // number of columns
t->numel();      // total number of elements (rows * cols)
t->device();     // Device::CPU or Device::CUDA
t->label();      // debug name
t->set_label("new_name");

t->represent();  // pretty-print values, shape, and parents to stdout
t->get_grad();    // print accumulated gradient (requires backward, see section 5)
t->parents();    // std::vector<shared_ptr<Tensor>> — inputs in the graph
```

---

## 4. Broadcasting rules

`operator+` delegates to `MatrixCpu::add` or `MatrixCuda::add`, both of which use the same modulo-based algorithm. For each dimension independently: sizes must match, or one of them must be 1.

| Left | Right | Result shape |
|------|-------|--------------|
| M×N | M×N | M×N — element-wise |
| M×N | 1×N | M×N — add row vector to each row |
| M×N | M×1 | M×N — add column vector to each column |
| 1×N | M×N | M×N |
| M×1 | M×N | M×N |
| 1×N | M×1 | M×N — outer sum |
| M×N | 1×1 | M×N — add scalar |
| 1×1 | M×N | M×N |

Incompatible shapes throw `std::runtime_error`.

Matrix multiplication (`operator*`) requires inner dimensions to match: `(M×K) * (K×N) = M×N`. Mismatched inner dimensions throw `std::runtime_error`.

---

## 5. Autograd

### How the graph is built

When `requires_grad` is true on at least one input, each operation records:

- **`_parents`** — the input tensors that fed into this result
- **`_gradfn`** — a closure that knows how to push gradients back to those inputs

```cpp
auto x = Tensor::full(2.0f, 2, 2, Device::CPU, true, "x");
auto w = Tensor::full(0.5f, 2, 2, Device::CPU, true, "w");

auto y = (*x) * w;   y->set_label("y");
auto z = y->relu();  z->set_label("z");

// inspect the graph
y->represent();   // shows parents: x, w
z->represent();   // shows parents: y
```

### Supported backward rules

| Operation | Backward formula |
|-----------|-----------------|
| `a + b` | gradient passes through unchanged to both inputs |
| `a * b` (matmul) | `grad_a = upstream @ b.T`, `grad_b = a.T @ upstream` |
| `relu` | `grad_input[i] = upstream[i] if input[i] > 0 else 0` |

### `backward()` — coming soon

A public `backward()` method that performs a reverse topological traversal and fires each `_gradfn` is the next milestone. The planned algorithm:

1. Seed the output gradient with a matrix of ones
2. Build topological order via post-order DFS from the output node
3. Walk the order in reverse, calling `_gradfn()` at each node

Until then, `requires_grad = true` is useful for building and inspecting the graph structure via `parents()`.

---

## 6. Adding your own program

### Option A — replace main.cpp

Edit `main.cpp` in the project root, then rebuild:

```bash
cd build && make && ./tinytorch
```

### Option B — add a second executable

In `CMakeLists.txt`, list shared sources once and add a new target:

```cmake
set(SOURCES
    src/core/matrix_factory.cpp
    src/core/tensor.cpp
    src/cpu/matrix_cpu.cpp
)

add_executable(tinytorch   main.cpp       ${SOURCES})
add_executable(my_program  my_program.cpp ${SOURCES})

target_include_directories(tinytorch  PRIVATE include/)
target_include_directories(my_program PRIVATE include/)
```

Then reconfigure and build:

```bash
cd build
cmake ..
make
./my_program
```

---

## 7. CUDA backend notes

### What is implemented

All Matrix operations have CUDA kernel equivalents:

| Operation | Kernel | Notes |
|-----------|--------|-------|
| `fill` | `fill_mat_kernel` | used by the fill-value constructor |
| `add` | `add_broadcast_kernel` | same modulo-based broadcasting as CPU |
| `matmul` | `matmul_kernel` | tiled shared-memory, 32×32 tiles |
| `relu` | `relu_kernel` | element-wise, one thread per element |
| `relu_backward` | `relu_backward_kernel` | reads pre-activation input for mask |
| `transpose` | `transpose_kernel` | shared-memory tiling with +1 padding to avoid bank conflicts |
| `randn` | cuRAND | generates directly into device memory |

### Device memory

`MatrixCuda::_values` is device memory allocated with `cudaMalloc`. Never read or write it directly from CPU code — use `represent()` to print, or `cudaMemcpy` explicitly if you need host access.

### Cross-device operations

Mixing `Device::CPU` and `Device::CUDA` tensors in a single operation is not yet supported and will throw a `Device mismatch` error at the `Tensor` layer.

---

## 8. Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `cmake: command not found` | CMake not installed | Install CMake 3.20+ |
| `make: No targets specified` | Ran `make` outside `build/` | `cd build` first |
| `No such file or directory: ./tinytorch` | Wrong working directory | Use `./build/tinytorch` from the project root |
| C++20 errors | Compiler too old | Upgrade to GCC 10+ / Clang 12+ |
| CUDA compiler not found | CMake didn't init CUDA | Check `enable_language(CUDA)` is in CMakeLists.txt and CUDA Toolkit is installed |
| nvcc host compiler error | GCC version too new for CUDA | Pass `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13` |
| `Device mismatch` at runtime | Mixed CPU/CUDA tensors | Use the same `Device` for all tensors in one expression |
| `dimensions do not match` on `+` | Incompatible broadcast shapes | Check `rows()` / `cols()` via `represent()` |
| `dimensions do not match` on `*` | Inner dims differ | For `(M×K) * (K×N)`, ensure `K` matches |
| `Double indexing only works on 2D Tensors` | Used `(i,j)` on a vector | Use flat index `(*t)(i)` for 1×N or M×1 tensors |
| Link error after adding a file | New `.cpp`/`.cu` not in CMake | Add it to `SOURCES` in `CMakeLists.txt` |
| Stale IDE errors after rename | clangd cache out of date | Run `clangd: Restart language server` in the command palette; always trust the compiler over the IDE |
| `CURAND error 105` | Odd element count passed to cuRAND | `curandGenerateNormal` requires even count — handled internally by rounding up |

---

## 9. Next steps

- Read `src/core/tensor.cpp` — this is where the computation graph is built and all `gradfn` closures are defined
- Read `src/cpu/matrix_cpu.cpp` — the broadcasting algorithm and tiled matmul on CPU
- Read `src/cuda/matrix_cuda.cu` — the CUDA kernel implementations and cuRAND usage
- Browse `include/tinytorch/core/` — the public header surface for Tensor and Matrix
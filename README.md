<p align="center">
  <img src="assets/tinytorch.png" alt="tinytorch logo" width="400">
</p>

A minimal C++ tensor library for learning how deep learning frameworks work under the hood. tinytorch provides 2D tensors, basic linear-algebra operations, NumPy-style broadcasting, and a computation-graph foundation for automatic differentiation — with both CPU and CUDA backends.

Inspired by [PyTorch](https://pytorch.org/), but intentionally small and readable. The entire implementation fits in a handful of files with no third-party dependencies beyond the standard library and the optional CUDA toolkit.

## Features

- **2D tensors** with row-major flat storage
- **CPU and CUDA backends** selected at construction time via `Device::CPU` / `Device::CUDA`
- **Broadcasting addition** — same rules as NumPy/PyTorch: any dimension of size 1 stretches to match the other
- **Matrix multiplication** with a tiled shared-memory CUDA kernel
- **ReLU** activation with a correct backward mask
- **Computation graph** — operations record their inputs and register `gradfn` closures for future backpropagation
- **Autograd-ready** — `accumulateGrad` and parent tracking are in place; a public `backward()` traversal is the next milestone
- **Factory API** that mirrors PyTorch: `zeros`, `full`, `scalar`, `randn`

## Requirements

| | CPU build | CUDA build |
|---|---|---|
| Compiler | should support C++20 | same, host compiler must be GCC ≤ 15 for CUDA 13 |
| CMake | 3.20+ | 3.20+ |
| CUDA Toolkit | not required | 12.x or 13.x |
| cuRAND | not required | bundled with CUDA Toolkit |

No other third-party dependencies.

## Quick start

### CPU-only (default)

```bash
git clone https://github.com/zhukeshhhh/tinytorch.git
cd tinytorch
mkdir build && cd build
cmake ..
make
./tinytorch
```

### With CUDA backend

```bash
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=path/to/compatible/c++/compiler
make
./tinytorch
```

You do not need a `-DCMAKE_CUDA_HOST_COMPILER` flag if your default nvcc host compiler is compatible.

To rebuild after editing source files, run `make` from inside `build/`. A full clean rebuild (`rm -rf build/`) is only needed when changing compilers or CMake flags.

## Architecture

tinytorch is split into three layers, each with a single responsibility:

```
  main.cpp / user code
        │
        ▼
    Tensor          user-facing API — shared_ptr lifetime, computation graph,
        │           requires_grad, gradfn closures, accumulateGrad
        ▼
    Matrix          abstract interface — add, matmul, relu, transpose,
        │           relu_backward, randn, repr, values, rows, cols, numel
        │
   ┌────┴────┐
   ▼         ▼
MatrixCpu  MatrixCuda
(std C++)  (CUDA kernels)
```

`MatrixFactory` is the only place that knows which concrete backend to instantiate. Everything above it is backend-agnostic.

## Example

```cpp
#include "tinytorch/core/tensor.hpp"

int main() {
    // 3×3 matrix of ones + 3×1 column vector of twos → broadcasts to 3×3
    auto a = Tensor::full(1.0f, 3, 3, Device::CPU, false, "a");
    auto b = Tensor::full(2.0f, 3, 1, Device::CPU, false, "b");

    auto c = (*a) + b;   c->setLabel("c");   // broadcasting add
    auto d = (*a) * b;   d->setLabel("d");   // matmul (3×3 * 3×1 = 3×1)

    c->represent();
    d->represent();

    // gradient tracking
    auto x = Tensor::randn(4, 4, Device::CPU, true, "x");
    auto w = Tensor::randn(4, 4, Device::CPU, true, "w");
    auto y = ((*x) * w)->relu();
    y->setLabel("y");
    y->represent();   // prints parents: x, w
}
```

Tensors are `std::shared_ptr<Tensor>`. Binary operators are defined on the pointed-to object, so dereference with `*` on the left-hand side: `(*a) + b`.

## Broadcasting rules

Addition supports all standard NumPy-style broadcasting patterns for 2D tensors:

| Left | Right | Result |
|------|-------|--------|
| M×N | M×N | M×N |
| M×N | 1×N | M×N |
| M×N | M×1 | M×N |
| 1×N | M×1 | M×N (outer sum) |
| M×N | 1×1 | M×N |
| 1×1 | M×N | M×N |

For each dimension independently: sizes must match, or one of them must be 1. Incompatible shapes throw `std::runtime_error`.

## Project structure

```
tinytorch/
├── include/
│   └── tinytorch/
│       ├── core/
│       │   ├── tensor.hpp
│       │   ├── matrix.hpp
│       │   ├── matrix_factory.hpp
│       │   └── device.hpp
│       ├── cpu/
│       │   └── matrix_cpu.hpp
│       └── cuda/
│           └── matrix_cuda.cuh
├── src/
│   ├── core/
│   │   ├── tensor.cpp
│   │   └── matrix_factory.cpp
│   ├── cpu/
│   │   └── matrix_cpu.cpp
│   └── cuda/
│       └── matrix_cuda.cu
├── docs/
│   └── TUTORIAL.md
├── main.cpp
└── CMakeLists.txt
```

## Roadmap

- [x] CPU backend — add, matmul, relu, transpose, randn, broadcasting
- [x] CUDA backend — all of the above with tiled shared-memory kernels
- [x] Computation graph — parent tracking and gradfn closures
- [x] `backward()` — topological traversal to fire gradfn closures
- [x] More activations — sigmoid, tanh, softmax with backward passes
- [ ] `operator-` and neg() with broadcasting
- [ ] Cross-device tensor copy (`Device::CPU` ↔ `Device::CUDA`)

## License

[MIT](LICENSE) — Copyright (c) 2026 Nurzhan Zhukesh
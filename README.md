# tinytorch

A minimal C++ tensor library for learning how deep learning frameworks work under the hood. tinytorch provides 2D tensors, basic linear-algebra operations, broadcasting, and a computation-graph foundation for automatic differentiation. CPU backend works and a CUDA backend will be available soon.

Inspired by [PyTorch](https://pytorch.org/), but intentionally small and readable.

## Features

| Area | Status |
|------|--------|
| 2D tensors (`rows × cols`) | ✅ |
| Factory methods (`scalar`, `zeros`, `full`, `randn`) | ✅ |
| Element-wise addition with broadcasting | ✅ |
| Matrix multiplication | ✅ |
| ReLU activation | ✅ |
| Autograd graph (`requires_grad`, gradient accumulation hooks) | 🚧 partial |
| CPU backend (`MatrixCpu`) | ✅ |
| CUDA backend (`MatrixCuda`) | 🚧 partial |
| Installable library / package manager integration | ❌ not yet |

## Requirements

- **C++20** compiler (GCC 10+, Clang 12+, or MSVC 2019+)
- **CMake 3.20+**
- **CUDA Toolkit** — not required today; the active build is CPU-only

No third-party dependencies beyond the standard library.

## Quick start

```bash
git clone https://github.com/zhukeshhhh/tinytorch.git
cd tinytorch

mkdir build
cd build
cmake ..
make

./tinytorch
```

To rebuild after changing source files, run `make` again from the `build/` directory.

The demo program creates tensors, runs addition and matrix multiplication, and prints the computation graph.

For a step-by-step walkthrough — building, writing your first program, and understanding the API — see **[docs/TUTORIAL.md](docs/TUTORIAL.md)**.

## Project layout

```
tinytorch/
├── CMakeLists.txt          # Build configuration
├── main.cpp                # Demo / entry point
├── core/
│   ├── device.hpp          # Device enum (CPU, CUDA)
│   ├── matrix.hpp          # Abstract Matrix interface
│   ├── matrix_factory.hpp  # Backend factory
│   ├── matrix_factory.cpp
│   └── tensor.cpp          # Tensor class and public API
├── cpu/
│   ├── matrix_cpu.hpp      # CPU matrix implementation
│   └── matrix_cpu.cpp
└── cuda/
    └── matrix_cuda.cuh     # CUDA backend (stub, not wired in)
```

## Architecture

tinytorch is split into three layers:

1. **Tensor** — user-facing API. Holds data, tracks whether gradients are needed, and records parent tensors for the computation graph.
2. **Matrix** — abstract interface for storage and math (`add`, `matmul`, `relu`, `transpose`, etc.).
3. **Backend** — concrete implementations selected by `Device` through `MatrixFactory`.

```
  main.cpp
      │
      ▼
  Tensor  ──►  Matrix (interface)
                  │
          ┌───────┴───────┐
          ▼               ▼
     MatrixCpu      MatrixCuda (planned)
```

Operations return new tensors and, when `requires_grad` is enabled, register backward hooks that propagate gradients through the graph.

## Example

```cpp
#include "core/tensor.cpp"

int main() {
    auto a = Tensor::full(1.0f, 3, 3, Device::CPU, false, "a");
    auto b = Tensor::full(2.0f, 3, 1, Device::CPU, false, "b");

    auto c = (*a) + b;  c->setLabel("c");
    auto d = (*a) * b;  d->setLabel("d");

    c->represent();
    d->represent();
}
```

Tensors are `std::shared_ptr<Tensor>`. Binary operators are defined on the pointed-to object, so dereference with `*` when combining shared pointers: `(*a) + b`.

## Current limitations

- **Header-style inclusion** — `tensor.cpp` is included directly rather than linked as a separate library.
- **2D only** — tensors are matrices; there is no N-dimensional shape support yet.
- **No public `backward()`** — autograd hooks exist internally, but reverse-mode propagation is not exposed as a user API yet.
- **CPU-only runtime** — `Device::CUDA` is declared but not active in `MatrixFactory`.
- **No tests or CI** — validation is manual via the demo executable.

These are expected for an early-stage learning project and are natural next steps for contributors.

## Roadmap (informal)

- Expose a `backward()` method and complete the autograd loop
- Wire up and implement the CUDA backend
- Refactor into a linkable library with a clean public header
- Add N-dimensional tensor support
- Add more ops (loss functions, optimizers, etc.)

## License

[MIT](LICENSE) — Copyright (c) 2026 Nurzhan Zhukesh

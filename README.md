<p align="center">
  <img src="assets/tinytorch.png" alt="tinytorch logo" width="400">
</p>

A minimal C++ tensor library for learning how deep learning frameworks work under the hood. tinytorch provides 2D tensors, basic linear-algebra operations, broadcasting, and a computation-graph foundation for automatic differentiation. CPU backend works and a CUDA backend will be available soon.

Inspired by [PyTorch](https://pytorch.org/), but intentionally small and readable.

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
#include "tinytorch/core/tensor.hpp"

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



## License

[MIT](LICENSE) — Copyright (c) 2026 Nurzhan Zhukesh

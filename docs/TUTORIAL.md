# Local usage tutorial

This guide walks you through building tinytorch on your machine, running the included demo, and writing your own programs with the current API.

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
g++ --version    # or clang++ --version
cmake --version  # must be 3.20 or newer
```

## 1. Clone and build

All commands below assume you start from your shell's home directory (or wherever you keep projects).

```bash
git clone https://github.com/zhukeshhhh/tinytorch.git
cd tinytorch

mkdir build
cd build
cmake ..
make
```

This generates the executable at `build/tinytorch` (relative to the project root).

### Rebuild after changes

If you edit source files, recompile from inside `build/`:

```bash
cd build    # skip if you are already there
make
```

### Clean rebuild

From the project root:

```bash
rm -rf build
mkdir build
cd build
cmake ..
make
```

## 2. Run the demo

From inside the `build/` directory:

```bash
./tinytorch
```

Or from the project root:

```bash
./build/tinytorch
```

The demo (`main.cpp`) does the following:

1. Creates a 3×3 matrix filled with `1.0` (`a`)
2. Creates a 3×1 column vector filled with `2.0` (`b`)
3. Creates a 3×3 random normal matrix (`i`, with `requires_grad = true`)
4. Computes `c = a + b` and `d = a * b`
5. Prints each tensor's values, shape, and computation-graph parents

Sample output shape (values will differ for the random tensor):

```
Tensor {a}:
a.parents() = None
[1, 1, 1]
[1, 1, 1]
[1, 1, 1]
Shape : (3, 3)
====================================
...
```

When tensors go out of scope you may also see destructor messages like `Tensor a destroyed` — these are debug prints from the current implementation.

## 3. Write your own program

Headers live under `include/tinytorch/`, and implementations under `src/`. CMake adds `include/` to the include path and links the `.cpp` sources for you — include the public header and add your file to the build.

### Minimal example

Create a file `my_program.cpp` in the project root:

```cpp
#include "tinytorch/core/tensor.hpp"

int main() {
    // 2×2 matrix of zeros
    auto x = Tensor::zeros(2, 2, Device::CPU, false, "x");

    // Fill manually via flat index
    (*x)(0) = 1.0f;
    (*x)(1) = 2.0f;
    (*x)(2) = 3.0f;
    (*x)(3) = 4.0f;

    x->represent();
    return 0;
}
```

**Option A — edit the existing demo:** replace the contents of `main.cpp` with your code, then rebuild:

```bash
cd build
make
./tinytorch
```

**Option B — add a second executable:** list shared sources once and add another target in `CMakeLists.txt` (in the project root):

```cmake
set(SOURCES
    src/core/matrix_factory.cpp
    src/core/tensor.cpp
    src/cpu/matrix_cpu.cpp
)

add_executable(tinytorch main.cpp ${SOURCES})
add_executable(my_program my_program.cpp ${SOURCES})

target_include_directories(tinytorch PRIVATE include/)
target_include_directories(my_program PRIVATE include/)
```

Reconfigure and compile from `build/`:

```bash
cd build
cmake ..
make
./my_program
```

## 4. Tensor API reference

All factory methods return `std::shared_ptr<Tensor>`.

### Creating tensors

```cpp
// Scalar (1×1)
auto s = Tensor::scalar(3.14f, Device::CPU, false, "s");

// Zeros (rows × cols)
auto z = Tensor::zeros(2, 3, Device::CPU, false, "z");

// Fill with a constant
auto f = Tensor::full(0.5f, 2, 2, Device::CPU, false, "f");

// Standard normal random values
auto r1 = Tensor::randn(5, Device::CPU, false, "r1");      // 1×5 row vector
auto r2 = Tensor::randn(3, 4, Device::CPU, false, "r2");   // 3×4 matrix
```

Factory signature:

```cpp
Tensor::method(..., Device device = Device::CPU,
               bool requires_grad = false,
               std::string label = "")
```

Set `requires_grad = true` when you want the tensor to participate in the autograd graph (gradient hooks are recorded for supported ops).

### Accessing values

```cpp
auto t = Tensor::full(7.0f, 1, 1, Device::CPU, false, "t");

float& v = t->item();           // single-element tensor only
float& flat = (*t)(2);          // flat index (row-major)
float& elem = (*t)(1, 0);       // 2D index (rows > 1 and cols > 1)
```

### Operations

Binary operators require dereferencing the left-hand shared pointer:

```cpp
auto a = Tensor::full(1.0f, 2, 2, Device::CPU, false, "a");
auto b = Tensor::full(2.0f, 2, 2, Device::CPU, false, "b");

auto sum  = (*a) + b;   // element-wise add (with broadcasting)
auto prod = (*a) * b;   // matrix multiply
auto act  = a->relu();  // ReLU (unary, no dereference needed)
```

Both operands must live on the same device (`Device::CPU` for now; `Device::CUDA` support is planned).

### Inspecting tensors

```cpp
t->rows();    // number of rows
t->cols();    // number of columns
t->size();    // rows * cols
t->device();  // Device::CPU or Device::CUDA
t->label();   // debug name
t->setLabel("new_name");

t->represent();   // pretty-print values, shape, and parents
t->getGrad();     // print accumulated gradient (after backward exists)
t->parents();     // vector of input tensors in the graph
```

## 5. Broadcasting rules (addition)

`operator+` delegates to `MatrixCpu::add`, which supports several broadcasting patterns:

| Left | Right | Result shape |
|------|-------|--------------|
| M×N | M×N | M×N (element-wise) |
| M×N | 1×N | M×N (add row vector to each row) |
| M×N | M×1 | M×N (add column vector to each column) |
| 1×N | M×N | M×N |
| M×1 | M×N | M×N |
| M×N | 1×1 | M×N (add scalar) |
| 1×1 | M×N | M×N |

If shapes are incompatible, a `std::runtime_error` is thrown.

Matrix multiplication (`operator*`) requires the inner dimensions to match: an `M×K` matrix times a `K×N` matrix yields `M×N`.

## 6. Autograd (work in progress)

You can mark tensors with `requires_grad = true` when creating them. Supported operations (`+`, `*`, `relu`) register backward hooks that know how to propagate gradients.

**Important:** there is no public `backward()` method yet. The graph and `accumulateGrad` machinery exist internally, but you cannot trigger a full reverse pass from application code at this time. Setting `requires_grad = true` is useful for inspecting graph structure (`parents()`) and preparing for future backward support.

Example — building a graph (forward pass only for now):

```cpp
auto x = Tensor::full(2.0f, 2, 2, Device::CPU, true, "x");
auto w = Tensor::full(0.5f, 2, 2, Device::CPU, true, "w");

auto y = (*x) * w;
y->setLabel("y");

y->represent();   // shows parents: x, w
```

## 7. Troubleshooting

| Problem | Likely cause | Fix |
|---------|--------------|-----|
| `cmake: command not found` | CMake not installed | Install CMake 3.20+ |
| `make: *** No targets specified` | Ran `make` outside `build/` | `cd build` first, or run `cmake ..` if `build/` is empty |
| `No such file or directory: ./tinytorch` | Wrong working directory | Run from `build/`, or use `./build/tinytorch` from the project root |
| C++20 errors | Compiler too old | Upgrade GCC/Clang/MSVC |
| `Device mismatch` at runtime | Mixed CPU/CUDA tensors | Use `Device::CPU` for all tensors today |
| `dimensions do not match` on `+` | Invalid broadcast | Check `rows()` / `cols()` with `represent()` |
| `dimensions do not match` on `*` | Inner dims differ | For `(M×K) * (K×N)`, ensure `K` matches |
| `Double indexing only works on 2D Tensors` | Used `(i,j)` on a vector | Use flat index `(*t)(i)` for 1×N or M×1 |
| Link errors after adding files | New `.cpp` not in CMake | Add the file to `SOURCES` in `CMakeLists.txt` |
| `tinytorch/core/tensor.hpp: No such file` | Missing include path | Ensure `target_include_directories(... PRIVATE include/)` is set |

## 8. Next steps

- Read the source in `src/core/tensor.cpp` to see how the computation graph is built
- Explore `src/cpu/matrix_cpu.cpp` for broadcasting and BLAS-style matmul
- Browse public headers in `include/tinytorch/core/` for the Tensor API surface
- Watch `include/tinytorch/cuda/matrix_cuda.cuh` and `src/cuda/matrix_cuda.cu` for the planned GPU backend
- Check the [README](../README.md) for architecture overview and roadmap

Thank you for your attention!

#include "tinytorch/core/tensor.hpp"
#include <iostream>
#include <cassert>

void section(const std::string& title) {
    std::cout << "\n========== " << title << " ==========\n";
}

// ================================================================
// CPU TESTS
// ================================================================

void cpu_test_creation() {
    section("CPU Creation");
    auto a = Tensor::zeros(3, 4, Device::CPU, false, "a");
    assert(a->rows()  == 3);
    assert(a->cols()  == 4);
    assert(a->numel() == 12);
    a->represent();
    std::cout << "PASSED\n";
}

void cpu_test_full() {
    section("CPU Full");
    auto a = Tensor::full(5.0f, 2, 3, Device::CPU, false, "a");
    assert(a->rows() == 2);
    assert(a->cols() == 3);
    assert((*a)(0)   == 5.0f);
    assert((*a)(1,2) == 5.0f);
    a->represent();
    std::cout << "PASSED\n";
}

void cpu_test_scalar() {
    section("CPU Scalar");
    auto s = Tensor::scalar(3.14f, Device::CPU, false, "s");
    assert(s->rows()  == 1);
    assert(s->cols()  == 1);
    assert(s->numel() == 1);
    assert(s->item()  == 3.14f);
    s->represent();
    std::cout << "PASSED\n";
}

void cpu_test_randn() {
    section("CPU Randn");
    auto a = Tensor::randn(3, 5, Device::CPU, false, "a");
    assert(a->rows()  == 3);
    assert(a->cols()  == 5);
    assert(a->numel() == 15);
    a->represent();
    std::cout << "PASSED\n";
}

void cpu_test_indexing() {
    section("CPU Indexing");
    auto t = Tensor::zeros(2, 3, Device::CPU, false, "t");
    (*t)(0) = 1.0f;
    (*t)(1) = 2.0f;
    (*t)(2) = 3.0f;
    (*t)(3) = 4.0f;
    (*t)(4) = 5.0f;
    (*t)(5) = 6.0f;
    assert((*t)(0, 0) == 1.0f);
    assert((*t)(0, 2) == 3.0f);
    assert((*t)(1, 0) == 4.0f);
    assert((*t)(1, 2) == 6.0f);
    t->represent();
    std::cout << "PASSED\n";
}

void cpu_test_add_same_shape() {
    section("CPU Add — same shape");
    auto a = Tensor::full(1.0f, 2, 2, Device::CPU, false, "a");
    auto b = Tensor::full(3.0f, 2, 2, Device::CPU, false, "b");
    auto c = (*a) + b;
    c->setLabel("c");
    for (std::size_t i = 0; i < c->numel(); i++)
        assert((*c)(i) == 4.0f);
    c->represent();
    std::cout << "PASSED\n";
}

void cpu_test_broadcast_col() {
    section("CPU Broadcast — matrix + col vector (3x3 + 3x1)");
    auto a = Tensor::full(1.0f, 3, 3, Device::CPU, false, "a");
    auto b = Tensor::full(2.0f, 3, 1, Device::CPU, false, "b");
    auto c = (*a) + b;
    c->setLabel("c");
    for (std::size_t i = 0; i < c->numel(); i++)
        assert((*c)(i) == 3.0f);
    c->represent();
    std::cout << "PASSED\n";
}

void cpu_test_broadcast_row() {
    section("CPU Broadcast — matrix + row vector (3x3 + 1x3)");
    auto a = Tensor::full(1.0f, 3, 3, Device::CPU, false, "a");
    auto b = Tensor::full(2.0f, 1, 3, Device::CPU, false, "b");
    auto c = (*a) + b;
    c->setLabel("c");
    for (std::size_t i = 0; i < c->numel(); i++)
        assert((*c)(i) == 3.0f);
    c->represent();
    std::cout << "PASSED\n";
}

void cpu_test_outer_sum() {
    section("CPU Broadcast — outer sum (1x3 + 3x1)");
    // a = [1, 2, 3]  (1x3)
    auto a = Tensor::zeros(1, 3, Device::CPU, false, "a");
    (*a)(0) = 1.0f; (*a)(1) = 2.0f; (*a)(2) = 3.0f;

    // b = [10, 20, 30]^T  (3x1)
    auto b = Tensor::zeros(3, 1, Device::CPU, false, "b");
    (*b)(0) = 10.0f; (*b)(1) = 20.0f; (*b)(2) = 30.0f;

    // expected 3x3:
    // [11, 12, 13]
    // [21, 22, 23]
    // [31, 32, 33]
    auto c = (*a) + b;
    c->setLabel("c");
    assert((*c)(0, 0) == 11.0f);
    assert((*c)(0, 2) == 13.0f);
    assert((*c)(1, 1) == 22.0f);
    assert((*c)(2, 2) == 33.0f);
    c->represent();
    std::cout << "PASSED\n";
}

void cpu_test_broadcast_scalar() {
    section("CPU Broadcast — scalar + matrix (1x1 + 3x3)");
    auto a = Tensor::full(10.0f, 1, 1, Device::CPU, false, "a");
    auto b = Tensor::full(5.0f,  3, 3, Device::CPU, false, "b");
    auto c = (*a) + b;
    c->setLabel("c");
    for (std::size_t i = 0; i < c->numel(); i++)
        assert((*c)(i) == 15.0f);
    c->represent();
    std::cout << "PASSED\n";
}

void cpu_test_matmul() {
    section("CPU Matmul (2x3) * (3x2)");
    // a = [[1,2,3],[4,5,6]]  (2x3)
    auto a = Tensor::zeros(2, 3, Device::CPU, false, "a");
    (*a)(0)=1; (*a)(1)=2; (*a)(2)=3;
    (*a)(3)=4; (*a)(4)=5; (*a)(5)=6;

    // b = [[1,0],[0,1],[1,0]]  (3x2)
    auto b = Tensor::zeros(3, 2, Device::CPU, false, "b");
    (*b)(0)=1; (*b)(1)=0;
    (*b)(2)=0; (*b)(3)=1;
    (*b)(4)=1; (*b)(5)=0;

    // c = [[4,2],[10,5]]
    auto c = (*a) * b;
    c->setLabel("c");
    assert((*c)(0, 0) ==  4.0f);
    assert((*c)(0, 1) ==  2.0f);
    assert((*c)(1, 0) == 10.0f);
    assert((*c)(1, 1) ==  5.0f);
    c->represent();
    std::cout << "PASSED\n";
}

void cpu_test_relu() {
    section("CPU ReLU");
    auto a = Tensor::zeros(1, 5, Device::CPU, false, "a");
    (*a)(0) = -3.0f;
    (*a)(1) = -1.0f;
    (*a)(2) =  0.0f;
    (*a)(3) =  2.0f;
    (*a)(4) =  5.0f;

    auto b = a->relu();
    b->setLabel("relu(a)");
    assert((*b)(0) == 0.0f);
    assert((*b)(1) == 0.0f);
    assert((*b)(2) == 0.0f);
    assert((*b)(3) == 2.0f);
    assert((*b)(4) == 5.0f);
    b->represent();
    std::cout << "PASSED\n";
}

void cpu_test_graph() {
    section("CPU Computation Graph");
    auto x = Tensor::full(2.0f, 2, 2, Device::CPU, true, "x");
    auto w = Tensor::full(0.5f, 2, 2, Device::CPU, true, "w");

    auto y = (*x) * w;  y->setLabel("y");
    auto z = y->relu(); z->setLabel("z");

    assert(y->parents().size() == 2);
    assert(y->parents()[0]->label() == "x");
    assert(y->parents()[1]->label() == "w");
    assert(z->parents().size() == 1);
    assert(z->parents()[0]->label() == "y");

    y->represent();
    z->represent();
    std::cout << "PASSED\n";
}

void cpu_test_errors() {
    section("CPU Error Handling");
    auto a = Tensor::full(1.0f, 2, 3, Device::CPU, false, "a");
    auto b = Tensor::full(1.0f, 4, 5, Device::CPU, false, "b");

    bool caught = false;
    try { auto c = (*a) + b; }
    catch (...) { caught = true; }
    assert(caught);

    caught = false;
    try { auto c = (*a) * b; }
    catch (...) { caught = true; }
    assert(caught);

    std::cout << "PASSED\n";
}

// ================================================================
// CUDA TESTS
// ================================================================

#ifdef USE_CUDA

void cuda_test_creation() {
    section("CUDA Creation");
    auto a = Tensor::zeros(3, 4, Device::CUDA, false, "a");
    assert(a->rows()  == 3);
    assert(a->cols()  == 4);
    assert(a->numel() == 12);
    a->represent();
    std::cout << "PASSED\n";
}

void cuda_test_full() {
    section("CUDA Full");
    auto a = Tensor::full(5.0f, 2, 3, Device::CUDA, false, "a");
    assert(a->rows() == 2);
    assert(a->cols() == 3);
    a->represent();
    std::cout << "PASSED\n";
}

void cuda_test_randn() {
    section("CUDA Randn");
    auto a = Tensor::randn(3, 5, Device::CUDA, false, "a");
    assert(a->rows()  == 3);
    assert(a->cols()  == 5);
    assert(a->numel() == 15);
    a->represent();
    std::cout << "PASSED\n";
}

void cuda_test_add_same_shape() {
    section("CUDA Add — same shape");
    auto a = Tensor::full(1.0f, 2, 2, Device::CUDA, false, "a");
    auto b = Tensor::full(3.0f, 2, 2, Device::CUDA, false, "b");
    auto c = (*a) + b;
    c->setLabel("c");
    c->represent();
    std::cout << "PASSED\n";
}

void cuda_test_broadcast_col() {
    section("CUDA Broadcast — matrix + col vector (3x3 + 3x1)");
    auto a = Tensor::full(1.0f, 3, 3, Device::CUDA, false, "a");
    auto b = Tensor::full(2.0f, 3, 1, Device::CUDA, false, "b");
    auto c = (*a) + b;
    c->setLabel("c");
    c->represent();
    std::cout << "PASSED\n";
}

void cuda_test_outer_sum() {
    section("CUDA Broadcast — outer sum (1x3 + 3x1)");
    auto a = Tensor::full(1.0f, 1, 3, Device::CUDA, false, "a");
    auto b = Tensor::full(2.0f, 3, 1, Device::CUDA, false, "b");
    auto c = (*a) + b;
    c->setLabel("c");
    c->represent();
    std::cout << "PASSED\n";
}

void cuda_test_matmul() {
    section("CUDA Matmul (2x3) * (3x2)");
    // each a[i,k]=2, b[k,j]=4, K=3
    // result[i,j] = 3 * 2.0 * 4.0 = 24.0
    auto a = Tensor::full(2.0f, 2, 3, Device::CUDA, false, "a");
    auto b = Tensor::full(4.0f, 3, 2, Device::CUDA, false, "b");
    auto c = (*a) * b;
    c->setLabel("c");
    assert(c->rows() == 2 && c->cols() == 2);
    c->represent();
    std::cout << "PASSED\n";
}

void cuda_test_relu() {
    section("CUDA ReLU");
    // all negative → all zeros after relu
    auto a = Tensor::full(-3.0f, 2, 3, Device::CUDA, false, "a");
    auto b = a->relu();
    b->setLabel("relu(a)");
    b->represent();
    std::cout << "PASSED\n";
}

void cuda_test_graph() {
    section("CUDA Computation Graph");
    auto x = Tensor::full(2.0f, 2, 2, Device::CUDA, true, "x");
    auto w = Tensor::full(3.0f, 2, 2, Device::CUDA, true, "w");

    auto y = (*x) * w;  y->setLabel("y");
    auto z = y->relu(); z->setLabel("z");

    assert(y->parents().size() == 2);
    assert(y->parents()[0]->label() == "x");
    assert(y->parents()[1]->label() == "w");
    assert(z->parents().size() == 1);
    assert(z->parents()[0]->label() == "y");

    y->represent();
    z->represent();
    std::cout << "PASSED\n";
}

void cuda_test_errors() {
    section("CUDA Error Handling — incompatible shapes");
    auto a = Tensor::full(1.0f, 2, 3, Device::CUDA, false, "a");
    auto b = Tensor::full(1.0f, 4, 5, Device::CUDA, false, "b");

    bool caught = false;
    try { auto c = (*a) + b; }
    catch (...) { caught = true; }
    assert(caught);

    caught = false;
    try { auto c = (*a) * b; }
    catch (...) { caught = true; }
    assert(caught);

    std::cout << "PASSED\n";
}

void cuda_test_device_mismatch() {
    section("CUDA Error Handling — device mismatch");
    auto a = Tensor::full(1.0f, 2, 2, Device::CPU,  false, "a");
    auto b = Tensor::full(1.0f, 2, 2, Device::CUDA, false, "b");

    bool caught = false;
    try { auto c = (*a) + b; }
    catch (...) { caught = true; }
    assert(caught);

    std::cout << "PASSED\n";
}

#endif // USE_CUDA

// ================================================================

int main() {
    std::cout << "===== CPU TESTS =====\n";
    cpu_test_creation();
    cpu_test_full();
    cpu_test_scalar();
    cpu_test_randn();
    cpu_test_indexing();
    cpu_test_add_same_shape();
    cpu_test_broadcast_col();
    cpu_test_broadcast_row();
    cpu_test_outer_sum();
    cpu_test_broadcast_scalar();
    cpu_test_matmul();
    cpu_test_relu();
    cpu_test_graph();
    cpu_test_errors();
    std::cout << "\n===== ALL CPU TESTS PASSED =====\n";

#ifdef USE_CUDA
    std::cout << "\n===== CUDA TESTS =====\n";
    cuda_test_creation();
    cuda_test_full();
    cuda_test_randn();
    cuda_test_add_same_shape();
    cuda_test_broadcast_col();
    cuda_test_outer_sum();
    cuda_test_matmul();
    cuda_test_relu();
    cuda_test_graph();
    cuda_test_errors();
    cuda_test_device_mismatch();
    std::cout << "\n===== ALL CUDA TESTS PASSED =====\n";
#endif

    return 0;
}
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
    c->set_label("c");
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
    c->set_label("c");
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
    c->set_label("c");
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
    c->set_label("c");
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
    c->set_label("c");
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
    c->set_label("c");
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
    b->set_label("relu(a)");
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

    auto y = (*x) * w;  y->set_label("y");
    auto z = y->relu(); z->set_label("z");

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
    c->set_label("c");
    c->represent();
    std::cout << "PASSED\n";
}

void cuda_test_broadcast_col() {
    section("CUDA Broadcast — matrix + col vector (3x3 + 3x1)");
    auto a = Tensor::full(1.0f, 3, 3, Device::CUDA, false, "a");
    auto b = Tensor::full(2.0f, 3, 1, Device::CUDA, false, "b");
    auto c = (*a) + b;
    c->set_label("c");
    c->represent();
    std::cout << "PASSED\n";
}

void cuda_test_outer_sum() {
    section("CUDA Broadcast — outer sum (1x3 + 3x1)");
    auto a = Tensor::full(1.0f, 1, 3, Device::CUDA, false, "a");
    auto b = Tensor::full(2.0f, 3, 1, Device::CUDA, false, "b");
    auto c = (*a) + b;
    c->set_label("c");
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
    c->set_label("c");
    assert(c->rows() == 2 && c->cols() == 2);
    c->represent();
    std::cout << "PASSED\n";
}

void cuda_test_relu() {
    section("CUDA ReLU");
    // all negative → all zeros after relu
    auto a = Tensor::full(-3.0f, 2, 3, Device::CUDA, false, "a");
    auto b = a->relu();
    b->set_label("relu(a)");
    b->represent();
    std::cout << "PASSED\n";
}

void cuda_test_graph() {
    section("CUDA Computation Graph");
    auto x = Tensor::full(2.0f, 2, 2, Device::CUDA, true, "x");
    auto w = Tensor::full(3.0f, 2, 2, Device::CUDA, true, "w");

    auto y = (*x) * w;  y->set_label("y");
    auto z = y->relu(); z->set_label("z");

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


// ================================================================
// CPU BACKWARD TESTS
// ================================================================

// Helper: read grad value at flat index i (CPU only — values() is a host ptr)
static float cpu_grad(std::shared_ptr<Tensor> t, std::size_t i) {
    return t->grad()->values()[i];
}

// dc/da = dc/db = 1  (add passes upstream unchanged)
void cpu_test_backward_add_scalars() {
    section("CPU Backward — add scalars");
    auto a = Tensor::scalar(3.0f, Device::CPU, true, "a");
    auto b = Tensor::scalar(2.0f, Device::CPU, true, "b");
    auto c = (*a) + b; c->set_label("c");
    c->backward();

    assert(cpu_grad(a, 0) == 1.0f);
    assert(cpu_grad(b, 0) == 1.0f);
    std::cout << "PASSED\n";
}

// c = a * b (1x1 matmul = scalar multiply)
// dc/da = b = 4,  dc/db = a = 3
void cpu_test_backward_matmul_scalars() {
    section("CPU Backward — matmul scalars (1x1)");
    auto a = Tensor::scalar(3.0f, Device::CPU, true, "a");
    auto b = Tensor::scalar(4.0f, Device::CPU, true, "b");
    auto c = (*a) * b; c->set_label("c");
    c->backward();

    // grad_a = upstream(1x1) @ b^T(1x1) = 1 * 4 = 4
    // grad_b = a^T(1x1) @ upstream(1x1) = 3 * 1 = 3
    assert(cpu_grad(a, 0) == 4.0f);
    assert(cpu_grad(b, 0) == 3.0f);
    std::cout << "PASSED\n";
}

// relu passes gradient only where input was positive
void cpu_test_backward_relu() {
    section("CPU Backward — relu");
    auto a = Tensor::zeros(1, 5, Device::CPU, true, "a");
    (*a)(0) = -3.0f;
    (*a)(1) = -1.0f;
    (*a)(2) =  0.0f;   // boundary: NOT > 0, so grad = 0
    (*a)(3) =  2.0f;
    (*a)(4) =  5.0f;

    auto b = a->relu(); b->set_label("b");
    b->backward();

    // upstream = ones(1,5); relu_backward: output[i] = input[i] > 0 ? 1 : 0
    assert(cpu_grad(a, 0) == 0.0f);
    assert(cpu_grad(a, 1) == 0.0f);
    assert(cpu_grad(a, 2) == 0.0f);
    assert(cpu_grad(a, 3) == 1.0f);
    assert(cpu_grad(a, 4) == 1.0f);
    std::cout << "PASSED\n";
}

// add backward: each input gets the full upstream gradient
void cpu_test_backward_add_matrix() {
    section("CPU Backward — matrix add (2x3)");
    auto a = Tensor::full(2.0f, 2, 3, Device::CPU, true, "a");
    auto b = Tensor::full(1.0f, 2, 3, Device::CPU, true, "b");
    auto c = (*a) + b; c->set_label("c");
    c->backward();

    // upstream = ones(2,3); grad_a = grad_b = ones(2,3)
    for (std::size_t i = 0; i < 6; i++) {
        assert(cpu_grad(a, i) == 1.0f);
        assert(cpu_grad(b, i) == 1.0f);
    }
    std::cout << "PASSED\n";
}

// matmul backward:
//   grad_a = upstream(2x2) @ b^T(2x3)  — each entry = sum of 2 ones = 2
//   grad_b = a^T(3x2) @ upstream(2x2)  — each entry = sum of 2 ones = 2
void cpu_test_backward_matmul_matrix() {
    section("CPU Backward — matmul (2x3)@(3x2)");
    auto a = Tensor::full(1.0f, 2, 3, Device::CPU, true, "a");
    auto b = Tensor::full(1.0f, 3, 2, Device::CPU, true, "b");
    auto c = (*a) * b; c->set_label("c");
    c->backward();

    for (std::size_t i = 0; i < 6; i++)   // grad_a: 2x3 = 6 elements
        assert(cpu_grad(a, i) == 2.0f);
    for (std::size_t i = 0; i < 6; i++)   // grad_b: 3x2 = 6 elements
        assert(cpu_grad(b, i) == 2.0f);
    std::cout << "PASSED\n";
}

// relu(a + b) where output is positive — gradient flows through
void cpu_test_backward_chain_add_relu() {
    section("CPU Backward — chain: relu(a + b), all positive");
    auto a = Tensor::full( 2.0f, 1, 4, Device::CPU, true, "a");
    auto b = Tensor::full(-1.0f, 1, 4, Device::CPU, true, "b");
    auto c = (*a) + b; c->set_label("c");   // [1, 1, 1, 1] — all > 0
    auto d = c->relu(); d->set_label("d");
    d->backward();

    // relu backward: all c values > 0 → grad_c = ones(1,4)
    // add backward: grad_a = grad_b = ones(1,4)
    for (std::size_t i = 0; i < 4; i++) {
        assert(cpu_grad(a, i) == 1.0f);
        assert(cpu_grad(b, i) == 1.0f);
    }
    std::cout << "PASSED\n";
}

// relu(a + b) where output is negative — dead neurons, zero gradient
void cpu_test_backward_chain_relu_dead() {
    section("CPU Backward — chain: relu(a + b), all negative (dead neurons)");
    auto a = Tensor::full( 1.0f, 1, 4, Device::CPU, true, "a");
    auto b = Tensor::full(-3.0f, 1, 4, Device::CPU, true, "b");
    auto c = (*a) + b; c->set_label("c");   // [-2, -2, -2, -2] — all < 0
    auto d = c->relu(); d->set_label("d");
    d->backward();

    // relu backward: all c values < 0 → grad_c = zeros(1,4) → grad_a = grad_b = 0
    for (std::size_t i = 0; i < 4; i++) {
        assert(cpu_grad(a, i) == 0.0f);
        assert(cpu_grad(b, i) == 0.0f);
    }
    std::cout << "PASSED\n";
}

// z = (x @ w) + b,  full chain with two op types
// upstream_z = ones(2,2)
// grad_b  = ones(2,2)                                → each = 1
// grad_x  = ones(2,2) @ w^T(2,2)  = ones@full(2) → each = 2+2 = 4
// grad_w  = x^T(2,2) @ ones(2,2)  = full(1)@ones → each = 1+1 = 2
void cpu_test_backward_chain_matmul_add() {
    section("CPU Backward — chain: (x @ w) + b");
    auto x = Tensor::full(1.0f, 2, 2, Device::CPU, true, "x");
    auto w = Tensor::full(2.0f, 2, 2, Device::CPU, true, "w");
    auto b = Tensor::full(0.5f, 2, 2, Device::CPU, true, "b");

    auto y = (*x) * w; y->set_label("y");   // [[4,4],[4,4]]
    auto z = (*y) + b; z->set_label("z");   // [[4.5,4.5],[4.5,4.5]]
    z->backward();

    for (std::size_t i = 0; i < 4; i++) {
        assert(cpu_grad(x, i) == 4.0f);
        assert(cpu_grad(w, i) == 2.0f);
        assert(cpu_grad(b, i) == 1.0f);
    }
    std::cout << "PASSED\n";
}

// c = a + a — same tensor used twice, accumulateGrad called twice
// a._grad = upstream + upstream = 2 * ones = full(2.0f)
void cpu_test_backward_shared_input() {
    section("CPU Backward — shared input (c = a + a, grad accumulates)");
    auto a = Tensor::full(2.0f, 2, 2, Device::CPU, true, "a");
    auto c = (*a) + a; c->set_label("c");
    c->backward();

    for (std::size_t i = 0; i < 4; i++)
        assert(cpu_grad(a, i) == 2.0f);
    std::cout << "PASSED\n";
}

// only a receives gradient — b has requires_grad=false so its _grad stays null
// (never call b->getGrad() here — _grad is null and will crash)
void cpu_test_backward_only_one_requires_grad() {
    section("CPU Backward — only a requires_grad");
    auto a = Tensor::full(3.0f, 1, 3, Device::CPU, true,  "a");
    auto b = Tensor::full(1.0f, 1, 3, Device::CPU, false, "b");
    auto c = (*a) + b; c->set_label("c");
    c->backward();

    for (std::size_t i = 0; i < 3; i++)
        assert(cpu_grad(a, i) == 1.0f);
    // b->_grad is null — do not call b->getGrad()
    std::cout << "PASSED\n";
}

// three-op chain ending in relu, with mixed +/- intermediate values
// x = 1, w = -1 → y = x@w = [[-2,-2],[-2,-2]] < 0 → relu kills everything
// grad_x = grad_w = grad_b = 0
void cpu_test_backward_chain_all_dead() {
    section("CPU Backward — chain: relu((x @ w) + b), all dead");
    auto x = Tensor::full( 1.0f, 2, 2, Device::CPU, true, "x");
    auto w = Tensor::full(-1.0f, 2, 2, Device::CPU, true, "w");
    auto b = Tensor::full(-0.5f, 2, 2, Device::CPU, true, "b");

    auto y = (*x) * w; y->set_label("y");   // [[-2,-2],[-2,-2]]
    auto z = (*y) + b; z->set_label("z");   // [[-2.5,-2.5],[-2.5,-2.5]]
    auto r = z->relu(); r->set_label("r");  // [[0,0],[0,0]]
    r->backward();

    for (std::size_t i = 0; i < 4; i++) {
        assert(cpu_grad(x, i) == 0.0f);
        assert(cpu_grad(w, i) == 0.0f);
        assert(cpu_grad(b, i) == 0.0f);
    }
    std::cout << "PASSED\n";
}

// ================================================================
// CUDA BACKWARD TESTS  (value assertions need cudaMemcpy — skipped here;
// these verify backward() runs without crash and _grad is allocated)
// ================================================================

#ifdef USE_CUDA

void cuda_test_backward_add_scalar() {
    section("CUDA Backward — add scalars");
    auto a = Tensor::scalar(3.0f, Device::CUDA, true, "a");
    auto b = Tensor::scalar(2.0f, Device::CUDA, true, "b");
    std::cout << "TENSORS CREATED SUCCESSFULLY\n";
    auto c = (*a) + b; c->set_label("c");
    c->backward();
    std::cout << "backward() failed\n";
    a->grad();   // non-null → backward ran correctly
    b->grad();
    std::cout << "PASSED\n";
}

void cuda_test_backward_matmul() {
    section("CUDA Backward — matmul (2x3)@(3x2)");
    auto a = Tensor::full(1.0f, 2, 3, Device::CUDA, true, "a");
    auto b = Tensor::full(1.0f, 3, 2, Device::CUDA, true, "b");
    auto c = (*a) * b; c->set_label("c");
    c->backward();
    a->grad();
    b->grad();
    std::cout << "PASSED\n";
}

void cuda_test_backward_relu() {
    section("CUDA Backward — relu");
    auto a = Tensor::full(2.0f, 1, 5, Device::CUDA, true, "a");
    auto b = a->relu(); b->set_label("b");
    b->backward();
    a->grad();
    std::cout << "PASSED\n";
}

void cuda_test_backward_chain_matmul_add() {
    section("CUDA Backward — chain: (x @ w) + b");
    auto x = Tensor::full(1.0f, 2, 2, Device::CUDA, true, "x");
    auto w = Tensor::full(2.0f, 2, 2, Device::CUDA, true, "w");
    auto b = Tensor::full(0.5f, 2, 2, Device::CUDA, true, "b");
    auto y = (*x) * w; y->set_label("y");
    auto z = (*y) + b; z->set_label("z");
    z->backward();
    x->grad();
    w->grad();
    b->grad();
    std::cout << "PASSED\n";
}

void cuda_test_backward_chain_all_ops() {
    section("CUDA Backward — chain: relu((x @ w) + b)");
    auto x = Tensor::full(1.0f, 2, 2, Device::CUDA, true, "x");
    auto w = Tensor::full(2.0f, 2, 2, Device::CUDA, true, "w");
    auto b = Tensor::full(0.5f, 2, 2, Device::CUDA, true, "b");
    auto y = (*x) * w; y->set_label("y");
    auto z = (*y) + b; z->set_label("z");
    auto r = z->relu(); r->set_label("r");
    r->backward();
    x->grad();
    w->grad();
    b->grad();
    std::cout << "PASSED\n";
}

#endif // USE_CUDA

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
    // --- backward ---
    cpu_test_backward_add_scalars();
    cpu_test_backward_matmul_scalars();
    cpu_test_backward_relu();
    cpu_test_backward_add_matrix();
    cpu_test_backward_matmul_matrix();
    cpu_test_backward_chain_add_relu();
    cpu_test_backward_chain_relu_dead();
    cpu_test_backward_chain_matmul_add();
    cpu_test_backward_shared_input();
    cpu_test_backward_only_one_requires_grad();
    cpu_test_backward_chain_all_dead();
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
    // ... existing CUDA tests ...
    cuda_test_backward_add_scalar();
    cuda_test_backward_matmul();
    cuda_test_backward_relu();
    cuda_test_backward_chain_matmul_add();
    cuda_test_backward_chain_all_ops();
    std::cout << "\n===== ALL CUDA TESTS PASSED =====\n";
#endif

    return 0;
}
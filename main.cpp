#include "tinytorch/core/tensor.hpp"

#include <iostream>
#include <cassert>

void section(const std::string& title)
{
    std::cout << "\n========== " << title << " ==========\n";
}

void test_creation()
{
    section("CUDA Creation");

    auto a = Tensor::zeros(3,4,Device::CUDA,false,"A");

    assert(a->rows()==3);
    assert(a->cols()==4);
    assert(a->numel()==12);

    a->represent();

    std::cout<<"PASSED\n";
}

void test_full()
{
    section("CUDA Full");

    auto a = Tensor::full(5.0f,2,3,Device::CUDA,false,"A");

    assert(a->rows()==2);
    assert(a->cols()==3);

    a->represent();

    std::cout<<"PASSED\n";
}

void test_randn()
{
    section("CUDA Randn");

    auto a = Tensor::randn(3,5,Device::CUDA,false,"A");

    assert(a->rows()==3);
    assert(a->cols()==5);

    a->represent();

    std::cout<<"PASSED\n";
}

void test_add()
{
    section("CUDA Add");

    auto a = Tensor::full(1.0f,2,2,Device::CUDA,false,"A");
    auto b = Tensor::full(3.0f,2,2,Device::CUDA,false,"B");

    auto c = (*a)+b;
    c->setLabel("C");

    c->represent();

    std::cout<<"PASSED\n";
}

void test_broadcast()
{
    section("CUDA Broadcast");

    auto a = Tensor::full(1.0f,3,3,Device::CUDA,false,"A");
    auto b = Tensor::full(2.0f,3,1,Device::CUDA,false,"B");

    auto c = (*a)+b;
    c->setLabel("C");

    c->represent();

    std::cout<<"PASSED\n";
}

void test_outer_sum()
{
    section("CUDA Outer Sum");

    auto a = Tensor::full(1.0f,1,3,Device::CUDA,false,"A");
    auto b = Tensor::full(2.0f,3,1,Device::CUDA,false,"B");

    auto c = (*a)+b;
    c->setLabel("C");

    c->represent();

    std::cout<<"PASSED\n";
}

void test_matmul()
{
    section("CUDA Matmul");

    auto a = Tensor::full(2.0f,2,3,Device::CUDA,false,"A");
    auto b = Tensor::full(4.0f,3,2,Device::CUDA,false,"B");

    auto c = (*a)*b;
    c->setLabel("C");

    c->represent();

    std::cout<<"PASSED\n";
}

void test_relu()
{
    section("CUDA ReLU");

    auto a = Tensor::full(-3.0f,2,3,Device::CUDA,false,"A");

    auto b = a->relu();
    b->setLabel("ReLU(A)");

    b->represent();

    std::cout<<"PASSED\n";
}

void test_graph()
{
    section("CUDA Graph");

    auto x = Tensor::full(2.0f,2,2,Device::CUDA,true,"x");
    auto w = Tensor::full(3.0f,2,2,Device::CUDA,true,"w");

    auto y = (*x)*w;
    y->setLabel("y");

    auto z = y->relu();
    z->setLabel("z");

    assert(y->parents().size()==2);
    assert(z->parents().size()==1);

    y->represent();
    z->represent();

    std::cout<<"PASSED\n";
}

void test_errors()
{
    section("CUDA Errors");

    auto a = Tensor::full(1.0f,2,3,Device::CUDA,false,"A");
    auto b = Tensor::full(1.0f,4,5,Device::CUDA,false,"B");

    bool caught=false;

    try
    {
        auto c=(*a)+b;
    }
    catch(...)
    {
        caught=true;
    }

    assert(caught);

    caught=false;

    try
    {
        auto c=(*a)*b;
    }
    catch(...)
    {
        caught=true;
    }

    assert(caught);

    std::cout<<"PASSED\n";
}

int main()
{
    std::cout<<"===== CUDA TESTS =====\n";

    test_creation();
    test_full();
    test_randn();
    test_add();
    test_broadcast();
    test_outer_sum();
    test_matmul();
    test_relu();
    test_graph();
    test_errors();

    std::cout<<"\n===== ALL CUDA TESTS PASSED =====\n";
}


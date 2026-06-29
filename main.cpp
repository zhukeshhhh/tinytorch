#include "tinytorch/core/tensor.hpp"
#include <iostream>

#ifdef USE_CUDA
int main() {
    auto a = Tensor::scalar(2.0f, Device::CUDA, true, "a");

    auto b = Tensor::randn(10, 10, Device::CUDA, true, "b");

    auto x = Tensor::randn(10, 10, Device::CUDA, true, "x");

    auto y = x->mul(b); y->setLabel("y");

    auto c = (*a) * b; c->setLabel("c");

    auto d = (*b) * a; d->setLabel("d");
    b->represent();
    x->represent();

    y->represent();
    
    return 0;
}
#endif

#ifndef USE_CUDA
int main() {
    auto a = Tensor::scalar(2.0f, Device::CPU, true, "a");

    auto b = Tensor::randn(10, 10, Device::CPU, true, "b");

    auto c = (*a) * b; c->setLabel("c");

    b->represent();
    c->represent();
    
    return 0;
}

#endif

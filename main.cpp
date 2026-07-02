#include "tinytorch/core/tensor.hpp"
#include <iostream>

#ifdef USE_CUDA
int main() {
    auto a = Tensor::scalar(2.0f, Device::CUDA, true, "a");

    auto b = Tensor::randn(3, 3, Device::CUDA, true, "b");

    auto c = (*a) + b; c->set_label("c");

    auto d = (*c).exp(); d->set_label("d");

    auto e = (*d).log(); e->set_label("e");

    a->represent();
    b->represent();
    c->represent();
    d->represent();
    e->represent();
    
    return 0;
}
#endif

#ifndef USE_CUDA
int main() {
    auto a = Tensor::scalar(2.0f, Device::CPU, true, "a");

    auto b = Tensor::randn(10, 10, Device::CPU, true, "b");

    auto c = (*a) * b; c->set_label("c");

    auto d = (*c).exp();

    auto e = (*d).log();

    b->represent();
    c->represent();
    d->represent();
    
    return 0;
}

#endif

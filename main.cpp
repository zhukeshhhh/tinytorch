#include "tinytorch/core/tensor.hpp"

int main() {
    
    auto a = Tensor::full(1.0f, 3, 3, Device::CPU, false, "a");
    auto b = Tensor::full(2.0f, 3, 1, Device::CPU, false, "b");

    auto i = Tensor::randn(3, 3, Device::CPU, true, "i");

    auto c = (*a) + b; c->setLabel("c");
    auto d = (*a) * b; d->setLabel("d");

    a->represent();
    b->represent();
    c->represent();
    d->represent();

    return 0;
}
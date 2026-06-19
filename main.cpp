#include "core/tensor.cpp"

int main() {
    
    auto a = Tensor::full(1.0f, 3, 5, Device::CPU, true, "A");
    auto b = Tensor::full(3.0f, 3, 5, Device::CPU, true, "B");
    auto c = (*a) + b; c->setLabel("C");

    auto d = Tensor::full(2.0f, 5, 4, Device::CPU, true, "D"); d->setLabel("D");
    auto e = (*c) * d; e->setLabel("E");

    e->represent();
    d->represent();
    c->represent();

    return 0;
}
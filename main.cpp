#include "core/tensor.cpp"

int main() {
    
    auto a = Tensor::full(1.0f, 3, 3, Device::CPU, false, "a");
    auto b = Tensor::full(2.0f, 1, 1, Device::CPU, false, "b");

    auto c = (*a) + b; c->setLabel("c");
    a->represent();
    b->represent();
    c->represent();

    return 0;
}
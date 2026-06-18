#include "core/tensor.cpp"

int main() {

    auto a = Tensor(3UL, 5UL, Device::CPU, true, "a");
    auto b = Tensor(3UL, 5UL, Device::CPU, true, "b");

    auto c = Tensor((MatrixFactory::create(5.0f, 3, 5, Device::CPU)), true, "c");

    c.represent();

    return 0;
}
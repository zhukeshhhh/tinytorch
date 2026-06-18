#include "core/tensor.cpp"

int main() {

    auto a = Tensor(3, 5, Device::CPU, true, "a");
    auto b = Tensor(3, 5, Device::CPU, true, "b");

    auto c = Tensor(MatrixFactory::create(5.0f, 3, 5, Device::CPU), Device::CPU, true, "c");

    c.represent();

    return 0;
}
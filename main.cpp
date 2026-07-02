#include "tinytorch/core/tensor.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>


int main() {

    std::vector<float> vec = {1, 2, 3, 4, 5, 6, 7};

    auto a = Tensor::from_vector_1d(vec, Device::CUDA, true, "a-from-vector");

    auto d = Tensor::scalar(2.0f, Device::CUDA, true, "d");

    auto b = Tensor::full(2.0f, 3, 3, Device::CUDA, true, "b");

    auto c = (*a) + d; c->set_label("c");

    auto f = (*c) + d; f->set_label("F");

    c->represent();
    f->represent();
    
    return 0;
}
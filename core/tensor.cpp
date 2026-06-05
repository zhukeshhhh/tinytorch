#include <iostream>
#include <functional>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include "matrix.cpp"

class Tensor {

private:
    float* _data;
    std::size_t _rows;
    std::size_t _cols;
    float* _grad;
    std::function<void(const float*)> _gradfn;
    std::vector<std::shared_ptr<Tensor>> _parents;
    bool _requires_grad;

public:
    // single-value Tensor
    Tensor(
        float data,
        bool requires_grad = false,
        std::function<void(const float*)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    ) : _data{new float{data}},
        _rows{1},
        _cols{1},
        _grad{nullptr},
        _gradfn{gradfn},
        _parents{parents},
        _requires_grad{requires_grad}
    {
        // implement constructor for single-value Tensor
    }

    // vector Tensor : row vector by default
    Tensor(
        //float* data,
        std::size_t size,
        bool requires_grad = false,
        std::function<void(const float*)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    ) : //_data{new float[size]},
        _rows{1},
        _cols{size},
        _grad{nullptr},
        _gradfn{gradfn},
        _parents{parents},
        _requires_grad{requires_grad}
    {
        _data = (float*)malloc(sizeof(float) * size);
    }

    Tensor(
        float* data,
        std::size_t rows,
        std::size_t cols,
        bool requires_grad = false,
        std::function<void(const float*)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    ) : _data{new float[rows * cols]},
        _rows{rows},
        _cols{cols},
        _grad{nullptr},
        _gradfn{gradfn},
        _parents{parents},
        _requires_grad{requires_grad}
    {
        for (std::size_t i = 0; i < rows * cols; i++) {
            _data[i] = data[i];
        }
    }

    ~Tensor() {
        std::cout << "Tensor destroyed" << std::endl;
    }
};


int main() {

    float* data = (float*)malloc(sizeof(float) * 100);
    for (std::size_t i = 0; i < sizeof(data); i++) {
        data[i] = i;
    }

    std::shared_ptr<Tensor> ptr1 = std::make_shared<Tensor>(5, false);

    {
        std::shared_ptr<Tensor> ptr2 = ptr1;
        std::cout << "ptr2 inner scope use count = " << ptr2.use_count() << std::endl;
        std::cout << "ptr1 inner scope use count = " << ptr1.use_count() << std::endl;    
    }

    std::cout << "ptr1 global scope use count = " << ptr1.use_count() << std::endl;
}
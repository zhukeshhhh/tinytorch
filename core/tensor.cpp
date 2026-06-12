#include <iostream>
#include <functional>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <random>
#include "matrix.cpp"

class Tensor : public std::enable_shared_from_this<Tensor> {

private:
    Matrix* _data;
    std::vector<std::size_t> _shape;
    float* _grad;
    std::function<void(const Matrix&)> _gradfn;
    std::vector<std::shared_ptr<Tensor>> _parents;
    bool _requires_grad;

public:
    // single-value Tensor
    Tensor(
        float data,
        bool requires_grad = false,
        std::function<void(const Matrix&)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    ) : _data{new Matrix(1, 1)},
        _shape{1, 1},
        _grad{nullptr},
        _gradfn{gradfn},
        _parents{parents},
        _requires_grad{requires_grad}
    {
        _data->_values[0] = data;
    }

    // 1D Tensor

    Tensor(
        Matrix& data,
        bool requires_grad = false,
        std::function<void(const Matrix&)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    ) : _data{new Matrix(data)},
        _shape{data.getShape()},
        _grad{nullptr},
        _gradfn{gradfn},
        _parents{parents},
        _requires_grad{requires_grad}
    {
        
    }

    Tensor(
        Matrix& data,
        bool requires_grad = false,
        std::function<void(const Matrix&)> gradfn = nullptr,
        std::vector<std::shared_ptr<Tensor>> parents = {}
    ) : _data{new Matrix(data)},
        _shape{data.getShape()},
        _grad{nullptr},
        _gradfn{gradfn},
        _parents{parents},
        _requires_grad{requires_grad}
    {

    }

    float& item() {
        if (_data->getSize() != 1) {
            throw std::runtime_error("item() only works for single-value Tensors\n");
        }
        return _data->_values[0];
    }

    float& operator()(std::size_t i) {
        if (_shape[0] != 1) {
            throw std::runtime_error("Single indexing only works for 1D Tensors\n");
        }
        if (i >= _shape[0]) {
            throw std::runtime_error("Index is out of bounds");
        }
        return _data->_values[i];
    }

    float& operator()(std::size_t i, std::size_t j) {
        if (_shape[0] == 1 || _shape[1] == 1) {
            throw std::runtime_error("Double indexing only works on 2D Tensors\n");
        }
        if (i >= _shape[0] || j >= _shape[1]) {
            throw std::runtime_error("Index out of bounds\n");
        }
        return _data->_values[i + _shape[1] + j];
    }

    ~Tensor() {
        delete[] _data;
        delete[] _grad;
        std::cout << "Tensor destroyed" << std::endl;
    }
};


int main() {

    std::shared_ptr<Tensor> t1 = std::make_shared<Tensor>(5.0f, false);
    std::cout << t1->item() << std::endl;

    return 0;
}
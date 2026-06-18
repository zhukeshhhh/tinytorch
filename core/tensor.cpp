#include <iostream>
#include <functional>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <random>
#include <string>
#include "device.hpp"
#include "matrix.hpp"

class Tensor : public std::enable_shared_from_this<Tensor> {

private:
    std::shared_ptr<Matrix> _data;
    std::shared_ptr<Matrix> _grad;
    Device _device;
    std::function<void()> _gradfn;
    std::vector<std::shared_ptr<Tensor>> _parents;
    bool _requires_grad;
    std::string _label;

public:
    // single-value Tensor
    Tensor(float value, Device device = Device::CPU, bool requires_grad = false, std::string label = "")
        : _data{MatrixFactory::create(1, 1, device)},
          _grad{nullptr},
          _device{device},
          _gradfn{nullptr},
          _parents{},
          _requires_grad{requires_grad},
          _label{label}
    {
        _data->values()[0] = value;
    }

    Tensor(std::size_t rows, std::size_t cols, Device device = Device::CPU, bool requires_grad = false, std::string label = "")
        : _data{MatrixFactory::create(rows, cols, device)},
          _grad{nullptr},
          _device{device},
          _gradfn{nullptr},
          _parents{},
          _requires_grad{requires_grad},
          _label{label}
    {

    }

    Tensor(Matrix* data, Device device, bool requires_grad = false, std::string label = "")
        : _data{data},
          _grad{nullptr},
          _device{device},
          _gradfn{nullptr},
          _parents{},
          _requires_grad{requires_grad},
          _label{label}
    {

    }

    float& item() const {
        if (_data->size() != 1) {
            throw std::runtime_error("item() only works for single-value Tensors\n");
        }
        return _data->values()[0];
    }

    float& operator()(std::size_t i) {
        
        if (i >= _data->size()) {
            throw std::runtime_error("Index is out of bounds");
        }
        return _data->values()[i];
    }

    float& operator()(std::size_t i, std::size_t j) {
        if (_data->rows() == 1 || _data->cols() == 1) {
            throw std::runtime_error("Double indexing only works on 2D Tensors\n");
        }
        if (i >= _data->rows() || j >= _data->cols()) {
            throw std::runtime_error("Index out of bounds\n");
        }
        return _data->values()[i * _data->cols() + j];
    }


    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> other) {
        if (_device != other->_device)
            throw std::runtime_error("operator+ : Device mismatch\n");
        
        Matrix* result_data = _data->add(*other->_data);

        bool result_requires_grad = _requires_grad || other->_requires_grad;

        auto result = std::make_shared<Tensor>(result_data, _device, result_requires_grad);
        result->_parents = {shared_from_this(), other};

        if (result_requires_grad) {
            auto self = shared_from_this();
            result->_gradfn = [self, other, result_ptr = result.get()] {
                const Matrix& upstream_grad = *result_ptr->_grad;
                if (self->_requires_grad);
                    self->accumulateGrad(upstream_grad);
                if (other->_requires_grad)
                    other->accumulateGrad(upstream_grad);
            };
        }
        return result;
    }


    void accumulateGrad(const Matrix& incoming) {
        if (!_grad){
            _grad.reset(MatrixFactory::create(_data->rows(), _data->cols(), _device));
        }
        Matrix* updated = _grad->add(incoming);
        _grad.reset(updated);
    }

    std::string label() {
        return _label;
    }

    void setLabel(std::string label) {
        _label = label;
    }

    Device device() const {
        return _device;
    }

    std::size_t rows() const {
        return _data->rows();
    }

    std::size_t cols() const {
        return _data->cols();
    }

    ~Tensor() {
        std::cout << "Tensor "<< label() << " destroyed" << std::endl;
    }
};


int main() {
    return 0;
}
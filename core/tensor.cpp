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
          _grad{nullptr}


    float& item() {
        if (_data->getSize() != 1) {
            throw std::runtime_error("item() only works for single-value Tensors\n");
        }
        return _data->_values[0];
    }

    float& operator()(std::size_t i) {
        
        if (i >= _data->getSize()) {
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
        return _data->_values[i * _shape[1] + j];
    }


    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other) {
        //scalars
        if (_data->getSize() == 1 && other->_data->getSize() == 1) {
            float result = item() + other->item();
            return std::make_shared<Tensor>(result);
        }

        if (_shape[0] == other->_shape[0] && _shape[1] == other->_shape[1]) {
            Matrix result(_shape[0], _shape[1]);
            for (std::size_t i = 0; i < _data->getSize(); i++) {
                result._values[i] = operator()(i) + (*other)(i);
            }

            return std::make_shared<Tensor>(result);
        }

        else {
            throw std::runtime_error("Dimensions are inconsistent\n");
        }
    }


    const std::vector<std::size_t>& shape() {
        return _shape;
    }

    std::string getLabel() {
        return _label;
    }

    void setLabel(std::string label) {
        _label = label;
    }

    ~Tensor() {
        std::cout << "Tensor "<< getLabel() << " destroyed" << std::endl;
    }
};


int main() {

    auto m = std::make_shared<Tensor>(5.0f, false);
    auto n = std::make_shared<Tensor>(5.0f, false, "N");
    auto c = (*m) + n;
    c->setLabel("C");


    Matrix mat(3, 5);
    Tensor t(mat, false);


    std::cout << "Matrix " << mat._mshape[0] << " " << mat._mshape[1] << std::endl;
    std::cout << "Tensor " << t.shape()[0] << " " << t.shape()[1] << std::endl;
    for (std::size_t i = 0; i < t.shape()[0]; i++) {
        for (std::size_t j = 0; j < t.shape()[1]; j++) {
            std::cout << t(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Tensor " << c->getLabel() << " = " << c->item() << std::endl;


    return 0;
}
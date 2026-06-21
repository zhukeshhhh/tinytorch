#pragma once

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
#include "matrix_factory.hpp"

class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    std::shared_ptr<Matrix> _data;
    std::shared_ptr<Matrix> _grad;
    Device _device;
    std::function<void()> _gradfn;
    std::vector<std::shared_ptr<Tensor>> _parents;
    bool _requires_grad;
    std::string _label;

    Tensor(
        float value,
        Device device,
        bool requires_grad,
        std::string label
    );
    
    Tensor(
        std::size_t rows,
        std::size_t cols,
        Device device,
        bool requires_grad,
        std::string label
    );

    Tensor(
        Matrix* data,
        bool requires_grad,
        std::string label
    );


public:
    static std::shared_ptr<Tensor> scalar(
        float value,
        Device device,
        bool requires_grad,
        std::string label
    );
    
    static std::shared_ptr<Tensor> zeros(
        std::size_t rows,
        std::size_t cols,
        Device device,
        bool requires_grad,
        std::string label
    );

    static std::shared_ptr<Tensor> full(
        float value,
        std::size_t rows,
        std::size_t cols,
        Device device,
        bool requires_grad,
        std::string label
    );

    static std::shared_ptr<Tensor> randn(
        std::size_t size,
        Device device,
        bool requires_grad,
        std::string label
    );

    static std::shared_ptr<Tensor> randn(
        std::size_t rows,
        std::size_t cols,
        Device device,
        bool requires_grad,
        std::string label
    );

    float& item();
    float& operator()(std::size_t i);
    float& operator()(std::size_t i, std::size_t j);
    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> relu();
    void accumulateGrad(const Matrix& incoming);
    std::string label();
    void setLabel(std::string label);
    Device device() const;
    inline std::size_t rows() const { return _data->rows(); }
    inline std::size_t cols() const { return _data->cols(); }
    inline std::size_t size() const { return _data->rows() * _data->cols(); }
    std::vector<std::shared_ptr<Tensor>> parents();
    std::shared_ptr<Matrix> getGrad();
    void represent();

    ~Tensor();
};
#pragma once

#include <functional>
#include <vector>
#include <memory>
#include <unordered_set>
#include <random>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <string>
#include <cstddef>
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

    Tensor(
        std::vector<float> vector_1d,
        Device device,
        bool requires_grad,
        std::string label
    );

    Tensor(
        std::vector<std::vector<float>> vector_2d,
        Device device,
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

    static std::shared_ptr<Tensor> from_vector_1d(
        std::vector<float> vector_1d,
        Device device,
        bool requires_grad,
        std::string label
    );

    static std::shared_ptr<Tensor> from_vector_2d(
        std::vector<std::vector<float>> vector_2d,
        Device device,
        bool requires_grad,
        std::string label
    );

    float& item();
    float& operator()(std::size_t i);
    float& operator()(std::size_t i, std::size_t j);
    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> matmul(const std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> relu();
    std::shared_ptr<Tensor> softmax();
    std::shared_ptr<Tensor> neg();
    std::shared_ptr<Tensor> sub(const std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> exp();
    std::shared_ptr<Tensor> log();

    float sum() const;
    float mean() const;

    void detach();

    void accumulate_grad(const Matrix& incoming);
    void zero_grad();
    std::string label();
    void set_label(std::string label);
    Device device() const;
    inline std::size_t rows() const { return _data->rows(); }
    inline std::size_t cols() const { return _data->cols(); }
    inline std::size_t numel() const { return _data->numel(); }
    std::vector<std::shared_ptr<Tensor>> parents();
    std::shared_ptr<Matrix> grad();
    void represent();


    void build_topo(std::shared_ptr<Tensor> node,
        std::unordered_set<std::shared_ptr<Tensor>>& visited,
        std::vector<std::shared_ptr<Tensor>>& topo
    );
    void backward();

    ~Tensor();
};
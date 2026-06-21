#include "tinytorch/core/tensor.hpp"


Tensor::Tensor(float value, Device device = Device::CPU, bool requires_grad = false, std::string label = "")
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

Tensor::Tensor(std::size_t rows, std::size_t cols, Device device = Device::CPU, bool requires_grad = false, std::string label = "")
    : _data{MatrixFactory::create(rows, cols, device)},
        _grad{nullptr},
        _device{device},
        _gradfn{nullptr},
        _parents{},
        _requires_grad{requires_grad},
        _label{label}
{
}

Tensor::Tensor(Matrix* data, bool requires_grad = false, std::string label = "")
    : _data{data},
        _grad{nullptr},
        _device{data->device()},
        _gradfn{nullptr},
        _parents{},
        _requires_grad{requires_grad},
        _label{label}
{
}


std::shared_ptr<Tensor> Tensor::scalar(
    float value,
    Device device = Device::CPU,
    bool requires_grad = false,
    std::string label = ""
)
{
    return std::shared_ptr<Tensor>(new Tensor(value, device, requires_grad, label));
}

std::shared_ptr<Tensor> Tensor::zeros(
    std::size_t rows,
    std::size_t cols,
    Device device = Device::CPU,
    bool requires_grad = false,
    std::string label = ""
)
{
    return std::shared_ptr<Tensor>(new Tensor(rows, cols, device, requires_grad, label));
}

std::shared_ptr<Tensor> Tensor::full(
    float value,
    std::size_t rows,
    std::size_t cols,
    Device device = Device::CPU,
    bool requires_grad = false,
    std::string label = ""
)
{
    return std::shared_ptr<Tensor>(new Tensor(MatrixFactory::create(value, rows, cols, device), requires_grad, label));
}

std::shared_ptr<Tensor> Tensor::randn(
    std::size_t size,
    Device device = Device::CPU,
    bool requires_grad = false,
    std::string label = ""
)
{
    return std::shared_ptr<Tensor>(new Tensor(MatrixFactory::create(1, size, device)->randn(), requires_grad, label));
}

std::shared_ptr<Tensor> Tensor::randn(
    std::size_t rows,
    std::size_t cols,
    Device device = Device::CPU,
    bool requires_grad = false,
    std::string label = ""
)
{
    return std::shared_ptr<Tensor>(new Tensor(MatrixFactory::create(rows, cols, device)->randn(), requires_grad, label));
}

float& Tensor::item() {

    if (_data->size() != 1)
        throw std::runtime_error("item() only works for single-value Tensors\n");
    
    return _data->values()[0];
}

float& Tensor::operator()(std::size_t i) {
    
    if (i >= _data->size())
        throw std::runtime_error("Index is out of bounds");
    
    return _data->values()[i];
}

float& Tensor::operator()(std::size_t i, std::size_t j) {

    if (rows() == 1 || cols() == 1)
        throw std::runtime_error("Double indexing only works on 2D Tensors\n");
    
    if (i >= rows() || j >= cols())
        throw std::runtime_error("Index out of bounds\n");
    
    return _data->values()[i * _data->cols() + j];
}


std::shared_ptr<Tensor> Tensor::operator+(const std::shared_ptr<Tensor> other) {
    if (_device != other->_device)
        throw std::runtime_error("operator+ : Device mismatch\n");
    
    Matrix* result_data = _data->add(*other->_data);

    bool result_requires_grad = _requires_grad || other->_requires_grad;

    auto result = std::shared_ptr<Tensor>(new Tensor(result_data, result_requires_grad));

    result->_parents = {shared_from_this(), other};

    if (result_requires_grad) {
        auto self = shared_from_this();
        result->_gradfn = [self, other, result_ptr = result.get()] {
            const Matrix& upstream_grad = *result_ptr->_grad;

            if (self->_requires_grad)
                self->accumulateGrad(upstream_grad);

            if (other->_requires_grad)
                other->accumulateGrad(upstream_grad);
        };
    }

    return result;
}


std::shared_ptr<Tensor> Tensor::operator*(const std::shared_ptr<Tensor> other) {
    if (_device != other->_device)
        throw std::runtime_error("operator* : Device mismatch\n");

    Matrix* result_data = _data->matmul(*other->_data);

    bool result_requires_grad = _requires_grad || other->_requires_grad;

    auto result = std::shared_ptr<Tensor>(new Tensor(result_data, result_requires_grad));

    result->_parents = {shared_from_this(), other};

    if (result_requires_grad) {
        auto self = shared_from_this();

        result->_gradfn = [self, other, result_ptr = result.get()] {
            const Matrix& upstream_grad = *result_ptr->_grad;

            if (self->_requires_grad) {
                auto other_T = std::unique_ptr<Matrix>(other->_data->transpose());
                auto self_grad = std::unique_ptr<Matrix>(upstream_grad.matmul(*other_T));

                self->accumulateGrad(*self_grad);
            }

            if (other->_requires_grad) {
                auto self_T = std::unique_ptr<Matrix>(self->_data->transpose());
                auto other_grad = std::unique_ptr<Matrix>(self_T->matmul(upstream_grad));

                other->accumulateGrad(*other_grad);
            }
        };
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::relu() {
    Matrix* result_data = _data->relu();
    auto result = std::shared_ptr<Tensor>(new Tensor(result_data, _requires_grad));
    result->_parents = {shared_from_this()};

    if (_requires_grad) {
        auto self = shared_from_this();

        result->_gradfn = [self, result_ptr = result.get()] {
            const Matrix& upstream_grad = *result_ptr->_grad;
            Matrix* grad = self->_data->relu_backward(upstream_grad);
            self->accumulateGrad(*grad);
            delete grad;
        };
    }

    return result;
}


void Tensor::accumulateGrad(const Matrix& incoming) {
    if (!_grad)
        _grad.reset(MatrixFactory::create(_data->rows(), _data->cols(), _device));
    
    Matrix* updated = _grad->add(incoming);
    _grad.reset(updated);
}

std::string Tensor::label() {
    return _label;
}

void Tensor::setLabel(std::string label) {
    _label = label;
}

Device Tensor::device() const {
    return _device;
}



std::vector<std::shared_ptr<Tensor>> Tensor::parents() {
    return _parents;
}

std::shared_ptr<Matrix> Tensor::getGrad() {
    std::cout << "Tensor {" << _label << "}._grad : \n";
    for (std::size_t i = 0; i < rows() * cols(); i++) {
        std::cout << _grad->at(i) << " ";
        if ((i - 1) % cols() == 0) {
            std::cout << std::endl;
        }
    }

    return _grad;
}

void Tensor::represent() {
    std::cout << "Tensor {" << _label << "}:\n";

    if (_parents.empty()) {
        std::cout << _label << ".parents() = None\n";
    }

    if (!_parents.empty()) {
        std::cout << _label << ".parents() = {";
        std::cout << _parents[0]->label() << ", ";
        std::cout << _parents[1]->label();
        std::cout << "}\n";
    }
    
    
    for (std::size_t i = 0; i < rows() * cols(); i++) {
        if (i % cols() == 0)
            std::cout << "[";

        std::cout << (*this)(i);

        if ((i + 1) % cols() == 0) {
            std::cout << "]\n";
        }
            
        else
            std::cout << ", ";
        
        
    }

    std::cout << "Shape : (" << rows() << ", " << cols() << ")\n";

    std::cout << "====================================\n";
}

Tensor::~Tensor() {
    std::cout << "Tensor {"<< label() << "} destroyed" << std::endl;
}
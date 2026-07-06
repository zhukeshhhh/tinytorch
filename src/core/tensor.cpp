#include "tinytorch/core/tensor.hpp"


Tensor::Tensor(float value, Device device, bool requires_grad = false, std::string label = "")
    : _data{MatrixFactory::create(value, 1, 1, device)},
        _grad{nullptr},
        _device{device},
        _gradfn{nullptr},
        _parents{},
        _requires_grad{requires_grad},
        _label{label}
{
}

Tensor::Tensor(std::size_t rows, std::size_t cols, Device device, bool requires_grad = false, std::string label = "")
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

Tensor::Tensor(std::vector<float> data, Device device, bool requires_grad, std::string label)
    : _data{MatrixFactory::create(data, device)},
        _grad{nullptr},
        _device{device},
        _gradfn{nullptr},
        _parents{},
        _requires_grad{requires_grad},
        _label{label}
{
}

Tensor::Tensor(std::vector<std::vector<float>> data, Device device, bool requires_grad, std::string label)
    : _data{MatrixFactory::create(data, device)},
        _grad{nullptr},
        _device{device},
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


static unsigned long long resolve_seed(unsigned long long seed) {
    if (seed != 0) return seed;
    static std::random_device rd;
    return (static_cast<unsigned long long>(rd()) << 32) | rd();
}


std::shared_ptr<Tensor> Tensor::randn(
    std::size_t size,
    Device device = Device::CPU,
    bool requires_grad = false,
    std::string label = "",
    unsigned long long seed = 0
)
{
    unsigned long long s = resolve_seed(seed);
    return std::shared_ptr<Tensor>(new Tensor(&(MatrixFactory::create(1, size, device)->randn(s)), requires_grad, label));
}

std::shared_ptr<Tensor> Tensor::randn(
    std::size_t rows,
    std::size_t cols,
    Device device = Device::CPU,
    bool requires_grad = false,
    std::string label = "",
    unsigned long long seed = 0
)
{
    unsigned long long s = resolve_seed(seed);
    return std::shared_ptr<Tensor>(new Tensor(&(MatrixFactory::create(rows, cols, device)->randn(s)), requires_grad, label));
}


std::shared_ptr<Tensor> Tensor::from_vector_1d(
    std::vector<float> vector_1d,
    Device device,
    bool requires_grad,
    std::string label
)
{
    return std::shared_ptr<Tensor>(new Tensor(MatrixFactory::create(vector_1d, device), requires_grad, label));
}


std::shared_ptr<Tensor> Tensor::from_vector_2d(
    std::vector<std::vector<float>> vector_2d,
    Device device,
    bool requires_grad,
    std::string label
)
{
    return std::shared_ptr<Tensor>(new Tensor(MatrixFactory::create(vector_2d, device), requires_grad, label));
}

float Tensor::item() {

    if (_data->numel() != 1)
        throw std::runtime_error("item() only works for single-value Tensors\n");
    
    return _data->scalar_value();
}

float& Tensor::operator()(std::size_t i) {
    if (_device == Device::CUDA)
        throw std::runtime_error("operator()(i) is unsupported on CUDA tensors. Use to_host_vector()/scalar_value() or write a CUDA-safe setter\n");
    if (i >= _data->numel())
        throw std::runtime_error("Index is out of bounds");
    
    return _data->values()[i];
}

float& Tensor::operator()(std::size_t i, std::size_t j) {
    if (_device == Device::CUDA)
        throw std::runtime_error("operator()(i,j) is unsupported on CUDA tensors. Use to_host_vector()/scalar_value() or write a CUDA-safe setter\n");
    if (i >= rows() || j >= cols())
        throw std::runtime_error("Index is out of bounds\n");
    return _data->values()[i * _data->cols() + j];
}


std::shared_ptr<Tensor> Tensor::add(const std::shared_ptr<Tensor> other) {
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

            if (self->_requires_grad) {
                if (self->rows() == upstream_grad.rows() && self->cols() == upstream_grad.cols()) {
                    self->accumulate_grad(upstream_grad);
                } else {
                    auto reduced = std::unique_ptr<Matrix>(upstream_grad.reduce_to(self->rows(), self->cols()));
                    self->accumulate_grad(*reduced);
                }
            }

            if (other->_requires_grad) {
                if (other->rows() == upstream_grad.rows() && other->cols() == upstream_grad.cols()) {
                    other->accumulate_grad(upstream_grad);
                } else {
                    auto reduced = std::unique_ptr<Matrix>(upstream_grad.reduce_to(other->rows(), other->cols()));
                    other->accumulate_grad(*reduced);
                }
            }
        };
    }

    return result;
}


std::shared_ptr<Tensor> Tensor::operator+(const std::shared_ptr<Tensor> other) {
    return add(other);
}


std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor> other) {
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

            bool self_is_scalar  = self->_data->numel()  == 1;
            bool other_is_scalar = other->_data->numel() == 1;

            if (other_is_scalar && !self_is_scalar) {
                if (self->_requires_grad) {
                    float s = other->_data->scalar_value();
                    auto scalar_mat = std::unique_ptr<Matrix>(
                        MatrixFactory::create(s, upstream_grad.rows(), upstream_grad.cols(), self->_device));
                    auto self_grad = std::unique_ptr<Matrix>(upstream_grad.mul(*scalar_mat));
                    self->accumulate_grad(*self_grad);
                }
                if (other->_requires_grad) {
                    auto prod = std::unique_ptr<Matrix>(upstream_grad.mul(*self->_data));
                    auto other_grad = std::unique_ptr<Matrix>(
                        MatrixFactory::create(prod->sum(), 1, 1, other->_device));
                    other->accumulate_grad(*other_grad);
                }
                return;
            }

            if (self_is_scalar && !other_is_scalar) {
                if (other->_requires_grad) {
                    float s = self->_data->scalar_value();
                    auto scalar_mat = std::unique_ptr<Matrix>(
                        MatrixFactory::create(s, upstream_grad.rows(), upstream_grad.cols(), other->_device));
                    auto other_grad = std::unique_ptr<Matrix>(upstream_grad.mul(*scalar_mat));
                    other->accumulate_grad(*other_grad);
                }
                if (self->_requires_grad) {
                    auto prod = std::unique_ptr<Matrix>(upstream_grad.mul(*other->_data));
                    auto self_grad = std::unique_ptr<Matrix>(
                        MatrixFactory::create(prod->sum(), 1, 1, self->_device));
                    self->accumulate_grad(*self_grad);
                }
                return;
            }

            if (self->_requires_grad) {
                auto other_T = std::unique_ptr<Matrix>(other->_data->transpose());
                auto self_grad = std::unique_ptr<Matrix>(upstream_grad.matmul(*other_T));
                self->accumulate_grad(*self_grad);
            }
            if (other->_requires_grad) {
                auto self_T = std::unique_ptr<Matrix>(self->_data->transpose());
                auto other_grad = std::unique_ptr<Matrix>(self_T->matmul(upstream_grad));
                other->accumulate_grad(*other_grad);
            }
        };
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::operator*(const std::shared_ptr<Tensor> other) {
    return matmul(other);
}


std::shared_ptr<Tensor> Tensor::mul(const std::shared_ptr<Tensor> other) {
    if (_device != other->_device)
        throw std::runtime_error("operator* : Device mismatch\n");

    Matrix* result_data = _data->mul(*other->_data);

    bool result_requires_grad = _requires_grad || other->_requires_grad;

    auto result = std::shared_ptr<Tensor>(new Tensor(result_data, result_requires_grad));

    result->_parents = {shared_from_this(), other};

    if (result->_requires_grad) {
        auto self = shared_from_this();

        result->_gradfn = [self, other, result_ptr = result.get()] {
            const Matrix& upstream_grad = *result_ptr->_grad;

            if (self->_requires_grad) {
                auto self_grad = std::unique_ptr<Matrix>(upstream_grad.mul(*other->_data));
                self->accumulate_grad(*self_grad);
            }
            if (other->_requires_grad) {
                auto other_grad = std::unique_ptr<Matrix>(upstream_grad.mul(*self->_data));
                other->accumulate_grad(*other_grad);
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
            self->accumulate_grad(*grad);
            delete grad;
        };
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::softmax() {
    Matrix* result_data = _data->softmax();

    auto result = std::shared_ptr<Tensor>(new Tensor(result_data, _requires_grad));
    result->_parents = {shared_from_this()};

    if (_requires_grad) {
        auto self = shared_from_this();

        result->_gradfn = [self, result_ptr = result.get()] {
            const Matrix& upstream_grad = *result_ptr->_grad;
            const Matrix& s = *result_ptr->_data;

            Matrix* grad = s.softmax_backward(upstream_grad);
            self->accumulate_grad(*grad);
            delete grad;
        };
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::neg() {
    return mul(Tensor::full(-1.0f, rows(), cols(), _device));
}

std::shared_ptr<Tensor> Tensor::sub(const std::shared_ptr<Tensor> other) {
    return add(other->neg());
}

std::shared_ptr<Tensor> Tensor::operator-(const std::shared_ptr<Tensor> other) {
    return sub(other);
}


std::shared_ptr<Tensor> Tensor::exp() {
    Matrix* exp_result = _data->exp();
    auto result = std::shared_ptr<Tensor>(new Tensor(exp_result, _requires_grad));
    result->_parents = {shared_from_this()};

    if (_requires_grad) {
        auto self = shared_from_this();

        result->_gradfn = [self, exp_result, result_ptr = result.get()] {
            const Matrix& upstream_grad = *result_ptr->_grad;
            Matrix* grad = self->_data->exp_backward(upstream_grad, *exp_result);
            self->accumulate_grad(*grad);
            delete grad;
        };
    }

    return result;
}

std::shared_ptr<Tensor> Tensor::log() {
    Matrix* log_result = _data->log();
    auto result = std::shared_ptr<Tensor>(new Tensor(log_result, _requires_grad));
    result->_parents = {shared_from_this()};

    if (_requires_grad) {
        auto self = shared_from_this();

        result->_gradfn = [self, result_ptr = result.get()] {
            const Matrix& upstream_grad = *result_ptr->_grad;
            Matrix* grad = self->_data->log_backward(upstream_grad);
            self->accumulate_grad(*grad);
            delete grad;
        };
    }

    return result;
}

float Tensor::sum() const {
    return _data->sum();
}

float Tensor::mean() const {
    return _data->mean();
}

void Tensor::detach() {
    if (!_requires_grad && _parents.empty() && !_gradfn) return;
    _requires_grad = false;
    _gradfn = nullptr;
    _parents.clear();
}


void Tensor::sdg_step(float& learning_rate, float& batch_size) {
    if (!_requires_grad) return;
    if (_data->numel() != _grad->numel()) throw std::runtime_error("Data/gradient size mismatch during SDG step\n");
    _data->sdg_step(learning_rate, batch_size, _grad.get());
}


std::vector<std::size_t> Tensor::argmax() const {
    return _data->argmax();
}


void Tensor::accumulate_grad(const Matrix& incoming) {
    if (!_grad)
        _grad.reset(MatrixFactory::create(_data->rows(), _data->cols(), _device));
    
    Matrix* updated = _grad->add(incoming);
    _grad.reset(updated);
}

void Tensor::zero_grad() {
    _grad.reset(MatrixFactory::create(_data->rows(), _data->cols(), _device));
}

std::string Tensor::label() {
    return _label;
}

void Tensor::set_label(std::string label) {
    _label = label;
}

Device Tensor::device() const {
    return _device;
}

std::vector<std::shared_ptr<Tensor>> Tensor::parents() {
    return _parents;
}

std::shared_ptr<Matrix> Tensor::grad() {
    return _grad;
}

void Tensor::represent() {
    std::string device = (_device == Device::CPU) ? "CPU" : "CUDA";
    std::cout << "Tensor {" << _label << "} | Device {" << device << "}:\n";

    if (_parents.empty()) {
        std::cout << _label << ".parents() = None\n";
    }
    
    else {
        std::cout << _label << ".parents() = {";

        for (std::size_t i = 0; i < _parents.size(); ++i) {
            std::cout << _parents[i]->label();
            if (i + 1 != _parents.size())
                std::cout << ", ";
        }
        std::cout << "}\n";
    }

    _data->repr();

    std::cout << "Shape:(" << rows() << ", " << cols() << ") | Sum:(" << sum() << ") | Mean:(" << mean() << ")\n";

    std::cout << "======================================\n";
}

void Tensor::build_topo(std::shared_ptr<Tensor> node,
        std::unordered_set<std::shared_ptr<Tensor>>& visited,
        std::vector<std::shared_ptr<Tensor>>& topo)
    {
        if (visited.contains(node)) return;
        visited.insert(node);

        for (auto parent : node->parents()) {
            build_topo(parent, visited, topo);
        }

        topo.push_back(node);
    }

void Tensor::backward() {
    this->_grad = std::shared_ptr<Matrix>(MatrixFactory::create(1.0f, rows(), cols(), device()));

    std::unordered_set<std::shared_ptr<Tensor>> visited;
    std::vector<std::shared_ptr<Tensor>> topo;

    build_topo(shared_from_this(), visited, topo);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->_gradfn) {
            (*it)->_gradfn();
        }
    }
}

Tensor::~Tensor() {}
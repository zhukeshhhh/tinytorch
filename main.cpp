#include "core/tensor.cpp"


void listParents(std::shared_ptr<Tensor> tensor) {

    if (tensor->parents().empty())
        return;

    std::vector<std::shared_ptr<Tensor>> parent_pointers = tensor->parents();

    std::cout << tensor->label() << " parents are : ";
    std::cout << parent_pointers[0]->label() << " AND ";
    std::cout << parent_pointers[1]->label() << std::endl;
}

int main() {
    
    auto a = Tensor::full(2.0f, 3, 5, Device::CPU, true, "A-tensor");
    auto b = Tensor::full(3.0f, 3, 5, Device::CPU, true, "B-tensor");
    auto c = Tensor::full(2.0f, 3, 5, Device::CPU, true, "C-tensor");
    auto d = Tensor::zeros(3, 5, Device::CPU, true, "D-tensor");

    auto e = (*a) + b; e->setLabel("E-tensor");
    auto f = (*c) + d; f->setLabel("F-tensor");
 
    auto L = (*e) + f; L->setLabel("L-tensor");

    listParents(L);
    listParents(f);
    listParents(e);
    listParents(a);

    return 0;
}
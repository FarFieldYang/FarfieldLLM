#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include "../ad_value.h"
#include "../grad_utils.h"
#include "../node.h"

class MatmulNode : public Node{
private:
    Shape output_shape_;

public:
    MatmulNode(std::shared_ptr<ADimpl> input1, std::shared_ptr<ADimpl> input2):
        Node({std::move(input1), std::move(input2)}) {
        const Tensor& a = inputs()[0]->value();
        const Tensor& b = inputs()[1]->value();

        if(a.ndim() < 2 || b.ndim() != a.ndim())
            throw std::invalid_argument("Matmul input ranks must match and be >= 2");

        for(Dim dim = 0; dim < a.ndim() - 2; ++dim){
            if(a.shape()[dim] != b.shape()[dim])
                throw std::invalid_argument("Matmul batch shapes must match");
        }
        if(a.shape()[a.ndim() - 1] != b.shape()[b.ndim() - 2])
            throw std::invalid_argument("Matmul inner dimensions must match");

        output_shape_ = a.shape();
        output_shape_.back() = b.shape().back();
    }

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input1 = inputs()[0];
        const auto& input2 = inputs()[1];
        check_grad_shape(grad_out, output_shape_);

        Dim dim0 = input1->value().ndim() - 2;
        Dim dim1 = input1->value().ndim() - 1;
        Gradients grads(2);
        if(input1->requires_grad())
            grads[0] = grad_out.matmul(input2->value().transpose(dim0, dim1));
        if(input2->requires_grad())
            grads[1] = input1->value().transpose(dim0, dim1).matmul(grad_out);
        return grads;
    }
};

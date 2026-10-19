#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include "../ad_value.h"
#include "../grad_utils.h"
#include "../node.h"

class SumNode : public Node{
private:
    Dim dim_;
    Shape output_shape_;

public:
    SumNode(std::shared_ptr<ADimpl> input, Dim dim):
        Node({std::move(input)}), dim_(dim), output_shape_(inputs()[0]->value().shape()) {
        if(dim_ >= output_shape_.size())
            throw std::out_of_range("Sum dimension out of range");
        output_shape_.erase(output_shape_.begin() + dim_);
    }

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.unsqueeze(dim_).broadcast_to(input->value().shape());
        return grads;
    }
};

class MaxNode : public Node{
private:
    Dim dim_;
    Shape output_shape_;

public:
    MaxNode(std::shared_ptr<ADimpl> input, Dim dim):
        Node({std::move(input)}), dim_(dim), output_shape_(inputs()[0]->value().shape()) {
        if(dim_ >= output_shape_.size())
            throw std::out_of_range("Max dimension out of range");
        output_shape_.erase(output_shape_.begin() + dim_);
    }

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = input->value().max_scatter(grad_out, dim_);
        return grads;
    }
};

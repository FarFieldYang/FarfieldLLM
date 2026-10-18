#pragma once

#include <memory>
#include <utility>

#include "../ad_value.h"
#include "../grad_utils.h"
#include "../node.h"

class AddNode : public Node{
public:
    AddNode(std::shared_ptr<ADimpl> input1, std::shared_ptr<ADimpl> input2):
        Node({std::move(input1), std::move(input2)}) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input1 = inputs()[0];
        const auto& input2 = inputs()[1];
        check_binary_grad_shape(grad_out, input1->value().shape(), input2->value().shape());

        Gradients grads(2);
        if(input1->requires_grad())
            grads[0] = sum_to_shape(grad_out, input1->value().shape());
        if(input2->requires_grad())
            grads[1] = sum_to_shape(grad_out, input2->value().shape());
        return grads;
    }
};

class SubNode : public Node{
public:
    SubNode(std::shared_ptr<ADimpl> input1, std::shared_ptr<ADimpl> input2):
        Node({std::move(input1), std::move(input2)}) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input1 = inputs()[0];
        const auto& input2 = inputs()[1];
        check_binary_grad_shape(grad_out, input1->value().shape(), input2->value().shape());

        Gradients grads(2);
        if(input1->requires_grad())
            grads[0] = sum_to_shape(grad_out, input1->value().shape());
        if(input2->requires_grad())
            grads[1] = sum_to_shape(grad_out, input2->value().shape()) * -1.0f;
        return grads;
    }
};

class MulNode : public Node{
public:
    MulNode(std::shared_ptr<ADimpl> input1, std::shared_ptr<ADimpl> input2):
        Node({std::move(input1), std::move(input2)}) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input1 = inputs()[0];
        const auto& input2 = inputs()[1];
        check_binary_grad_shape(grad_out, input1->value().shape(), input2->value().shape());

        Gradients grads(2);
        if(input1->requires_grad())
            grads[0] = sum_to_shape(grad_out * input2->value(), input1->value().shape());
        if(input2->requires_grad())
            grads[1] = sum_to_shape(grad_out * input1->value(), input2->value().shape());
        return grads;
    }
};

class DivNode : public Node{
public:
    DivNode(std::shared_ptr<ADimpl> input1, std::shared_ptr<ADimpl> input2):
        Node({std::move(input1), std::move(input2)}) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input1 = inputs()[0];
        const auto& input2 = inputs()[1];
        check_binary_grad_shape(grad_out, input1->value().shape(), input2->value().shape());

        Gradients grads(2);
        if(input1->requires_grad())
            grads[0] = sum_to_shape(grad_out / input2->value(), input1->value().shape());
        if(input2->requires_grad()){
            Tensor grad = grad_out * input1->value() / (input2->value() * input2->value());
            grads[1] = sum_to_shape(std::move(grad), input2->value().shape()) * -1.0f;
        }
        return grads;
    }
};

#pragma once

#include <memory>
#include <utility>

#include "../ad_value.h"
#include "../grad_utils.h"
#include "../node.h"

class ScalarAddNode : public Node{
public:
    explicit ScalarAddNode(std::shared_ptr<ADimpl> input): Node({std::move(input)}) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, input->value().shape());

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out;
        return grads;
    }
};

class ScalarSubNode : public Node{
public:
    explicit ScalarSubNode(std::shared_ptr<ADimpl> input): Node({std::move(input)}) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, input->value().shape());

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out;
        return grads;
    }
};

class ScalarMulNode : public Node{
private:
    Scalar scalar_;

public:
    ScalarMulNode(std::shared_ptr<ADimpl> input, Scalar scalar):
        Node({std::move(input)}), scalar_(scalar) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, input->value().shape());

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out * scalar_;
        return grads;
    }
};

class ScalarDivNode : public Node{
private:
    Scalar scalar_;

public:
    ScalarDivNode(std::shared_ptr<ADimpl> input, Scalar scalar):
        Node({std::move(input)}), scalar_(scalar) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, input->value().shape());

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out / scalar_;
        return grads;
    }
};

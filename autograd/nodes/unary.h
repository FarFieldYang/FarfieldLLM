#pragma once

#include <memory>
#include <stdexcept>
#include <utility>

#include "../ad_value.h"
#include "../grad_utils.h"
#include "../node.h"

class ExpNode : public Node{
private:
    Tensor output_;

public:
    ExpNode(std::shared_ptr<ADimpl> input, Tensor output):
        Node({std::move(input)}), output_(std::move(output)) {
        if(output_.shape() != inputs()[0]->value().shape())
            throw std::invalid_argument("Output shape does not match unary input shape");
    }

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_.shape());

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out * output_;
        return grads;
    }
};

class LogNode : public Node{
public:
    explicit LogNode(std::shared_ptr<ADimpl> input): Node({std::move(input)}) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, input->value().shape());

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out / input->value();
        return grads;
    }
};

class SqrtNode : public Node{
private:
    Tensor output_;

public:
    SqrtNode(std::shared_ptr<ADimpl> input, Tensor output):
        Node({std::move(input)}), output_(std::move(output)) {
        if(output_.shape() != inputs()[0]->value().shape())
            throw std::invalid_argument("Output shape does not match unary input shape");
    }

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_.shape());

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out / (output_ * 2.0f);
        return grads;
    }
};

class TanhNode : public Node{
private:
    Tensor output_;

public:
    TanhNode(std::shared_ptr<ADimpl> input, Tensor output):
        Node({std::move(input)}), output_(std::move(output)) {
        if(output_.shape() != inputs()[0]->value().shape())
            throw std::invalid_argument("Output shape does not match unary input shape");
    }

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_.shape());

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out * (output_ * output_ * -1.0f + 1.0f);
        return grads;
    }
};

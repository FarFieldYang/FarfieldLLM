#pragma once

#include <memory>
#include <utility>

#include "../ad_value.h"
#include "../grad_utils.h"
#include "../node.h"

class ContiguousNode : public Node{
public:
    explicit ContiguousNode(std::shared_ptr<ADimpl> input): Node({std::move(input)}) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, input->value().shape());

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out;
        return grads;
    }
};

class ReshapeNode : public Node{
private:
    Shape output_shape_;

public:
    ReshapeNode(std::shared_ptr<ADimpl> input, const Shape& shape):
        Node({std::move(input)}), output_shape_(inputs()[0]->value().reshape(shape).shape()) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.contiguous().reshape(input->value().shape());
        return grads;
    }
};

class TransposeNode : public Node{
private:
    Dim dim0_;
    Dim dim1_;
    Shape output_shape_;

public:
    TransposeNode(std::shared_ptr<ADimpl> input, Dim dim0, Dim dim1):
        Node({std::move(input)}), dim0_(dim0), dim1_(dim1),
        output_shape_(inputs()[0]->value().transpose(dim0, dim1).shape()) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.transpose(dim0_, dim1_);
        return grads;
    }
};

class PermuteNode : public Node{
private:
    Dims inverse_dims_;
    Shape output_shape_;

public:
    PermuteNode(std::shared_ptr<ADimpl> input, const Dims& dims):
        Node({std::move(input)}), inverse_dims_(dims.size()),
        output_shape_(inputs()[0]->value().permute(dims).shape()) {
        for(Dim dim = 0; dim < dims.size(); ++dim)
            inverse_dims_[dims[dim]] = dim;
    }

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.permute(inverse_dims_);
        return grads;
    }
};

class UnsqueezeNode : public Node{
private:
    Dim dim_;
    Shape output_shape_;

public:
    UnsqueezeNode(std::shared_ptr<ADimpl> input, Dim dim):
        Node({std::move(input)}), dim_(dim), output_shape_(inputs()[0]->value().unsqueeze(dim).shape()) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.squeeze(dim_);
        return grads;
    }
};

class SqueezeNode : public Node{
private:
    Dim dim_;
    Shape output_shape_;

public:
    SqueezeNode(std::shared_ptr<ADimpl> input, Dim dim):
        Node({std::move(input)}), dim_(dim), output_shape_(inputs()[0]->value().squeeze(dim).shape()) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.unsqueeze(dim_);
        return grads;
    }
};

class BroadcastNode : public Node{
private:
    Shape output_shape_;

public:
    BroadcastNode(std::shared_ptr<ADimpl> input, const Shape& shape):
        Node({std::move(input)}), output_shape_(inputs()[0]->value().broadcast_to(shape).shape()) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = sum_to_shape(grad_out, input->value().shape());
        return grads;
    }
};

class FlattenNode : public Node{
private:
    Shape output_shape_;

public:
    explicit FlattenNode(std::shared_ptr<ADimpl> input):
        Node({std::move(input)}), output_shape_(inputs()[0]->value().flatten().shape()) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.contiguous().reshape(input->value().shape());
        return grads;
    }
};

class NarrowNode : public Node{
private:
    Dim dim_;
    Index start_;
    std::size_t length_;
    Shape output_shape_;

public:
    NarrowNode(std::shared_ptr<ADimpl> input, Dim dim, Index start, std::size_t length):
        Node({std::move(input)}), dim_(dim), start_(start), length_(length),
        output_shape_(inputs()[0]->value().narrow(dim, start, length).shape()) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.slice_scatter(input->value().shape(), dim_, start_, start_ + length_, 1);
        return grads;
    }
};

class SliceNode : public Node{
private:
    Dim dim_;
    Index start_;
    Index end_;
    std::size_t step_;
    Shape output_shape_;

public:
    SliceNode(std::shared_ptr<ADimpl> input, Dim dim, Index start, Index end, std::size_t step):
        Node({std::move(input)}), dim_(dim), start_(start), end_(end), step_(step),
        output_shape_(inputs()[0]->value().slice(dim, start, end, step).shape()) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.slice_scatter(input->value().shape(), dim_, start_, end_, step_);
        return grads;
    }
};

class SelectNode : public Node{
private:
    Dim dim_;
    Index index_;
    Shape output_shape_;

public:
    SelectNode(std::shared_ptr<ADimpl> input, Dim dim, Index index):
        Node({std::move(input)}), dim_(dim), index_(index),
        output_shape_(inputs()[0]->value().select(dim, index).shape()) {}

    Gradients backward(const Tensor& grad_out) const override {
        const auto& input = inputs()[0];
        check_grad_shape(grad_out, output_shape_);

        Gradients grads(1);
        if(input->requires_grad())
            grads[0] = grad_out.unsqueeze(dim_).slice_scatter(input->value().shape(), dim_, index_, index_ + 1, 1);
        return grads;
    }
};

#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tensor.h"

class ADimpl;

class Node {
public:
    using Inputs = std::vector<std::shared_ptr<ADimpl>>;
    using Gradients = std::vector<std::optional<Tensor>>;

private:
    Inputs inputs_;

protected:
    explicit Node(Inputs inputs) : inputs_(std::move(inputs)) {
        for (const auto& input : inputs_) {
            if (!input)
                throw std::invalid_argument("Node input cannot be null");
        }
    }

public:
    virtual ~Node() = default;

    const Inputs& inputs() const noexcept {
        return inputs_;
    }

    virtual Gradients backward(const Tensor& grad_out) const = 0;
};

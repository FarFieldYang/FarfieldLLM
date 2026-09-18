#pragma once

#include <vector>

#include "tensor.h"

using GradValue = Tensor;

class GradNode {
public:
    virtual ~GradNode() = default;

    virtual std::vector<GradValue>
    backward(const GradValue& grad_output) = 0;
};
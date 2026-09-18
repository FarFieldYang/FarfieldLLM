#pragma once

#include <memory>
#include <vector>

class Tensor;
class GradNode;

using GradValue = Tensor;

class AutogradMeta {
public:
    bool requires_grad = false;

    std::shared_ptr<GradValue> grad;
    std::shared_ptr<GradNode> grad_fn;
};

#pragma once

#include <memory>

#include "tensor_data.h"

class Tensor{

private:
    TensorData data_;
public:
    explicit Tensor(TensorData data);

    std::size_t size() const;
    std::size_t ndim() const;
    const std::vector<std::size_t>& shape() const;

    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator+(Scalar scalar) const;
    Tensor operator*(Scalar scalar) const;
    Tensor sum() const;

    void backward();
};

Tensor operator*(Scalar scalar, const Tensor& other);
Tensor operator+(Scalar scalar, const Tensor& other);

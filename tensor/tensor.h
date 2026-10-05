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
//cuda_backend
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator+(Scalar scalar) const;
    Tensor operator-(Scalar scalar) const;
    Tensor operator*(Scalar scalar) const;
    Tensor operator/(Scalar scalar) const;
    Tensor exp()const;
    Tensor log()const;
    Tensor sqrt()const;
    Tensor tanh()const;
    Tensor sum(Dim dim)const;
    Tensor max(Dim dim)const;
    Tensor matmul(const Tensor& other) const;
//TensorData_backend
    Tensor contiguous() const;
//TensorData_view
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(Dim dim0, Dim dim1) const;
    Tensor permute(const Dims& dims) const;
    Tensor unsqueeze(Dim dim) const;
    Tensor squeeze(Dim dim) const;
    Tensor broadcast_to(const Shape& new_shape) const;
    Tensor narrow(Dim dim, Index start, std::size_t length) const;
    Tensor slice(Dim dim, Index start, Index end, std::size_t step = 1) const;
    Tensor select(Dim dim, Index index) const;
    Tensor flatten() const;
};

Tensor operator*(Scalar scalar, const Tensor& other);
Tensor operator+(Scalar scalar, const Tensor& other);

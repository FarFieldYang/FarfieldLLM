#include <utility>

#include "tensor.h"
#include "backend.h"

Tensor::Tensor(TensorData data): data_(std::move(data)){}

//metadata
std::size_t Tensor::size() const{
    return data_.numel();
}

std::size_t Tensor::ndim() const{
    return data_.ndim();
}

const std::vector<std::size_t>& Tensor::shape() const{
    return data_.shape();
}

//binary
Tensor Tensor::operator+(const Tensor& other) const{
    return Tensor(backend::add(data_, other.data_));
}

Tensor Tensor::operator-(const Tensor& other) const{
    return Tensor(backend::sub(data_, other.data_));
}

Tensor Tensor::operator*(const Tensor& other) const{
    return Tensor(backend::mul(data_, other.data_));
}

Tensor Tensor::operator/(const Tensor& other) const{
    return Tensor(backend::div(data_, other.data_));
}

//scalar binary
Tensor Tensor::operator+(Scalar scalar) const{
    return Tensor(backend::add(data_, scalar));
}

Tensor Tensor::operator-(Scalar scalar) const{
    return Tensor(backend::sub(data_, scalar));
}

Tensor Tensor::operator*(Scalar scalar) const{
    return Tensor(backend::mul(data_, scalar));
}

Tensor Tensor::operator/(Scalar scalar) const{
    return Tensor(backend::div(data_, scalar));
}

//unary
Tensor Tensor::exp() const{
    return Tensor(backend::exp(data_));
}

Tensor Tensor::log() const{
    return Tensor(backend::log(data_));
}

Tensor Tensor::sqrt() const{
    return Tensor(backend::sqrt(data_));
}

Tensor Tensor::tanh() const{
    return Tensor(backend::tanh(data_));
}

//reduction
Tensor Tensor::sum(Dim dim) const{
    return Tensor(backend::sum(data_, dim));
}

Tensor Tensor::max(Dim dim) const{
    return Tensor(backend::max(data_, dim));
}

//matrix
Tensor Tensor::matmul(const Tensor& other) const{
    return Tensor(backend::matmul(data_, other.data_));
}

//TensorData Backend
Tensor Tensor::contiguous() const{
    return Tensor(backend::contiguous(data_));
}

//TensorData view
Tensor Tensor::reshape(const Shape& new_shape) const{
    return Tensor(backend::reshape(data_, new_shape));
}

Tensor Tensor::transpose(Dim dim0, Dim dim1) const{
    return Tensor(backend::transpose(data_, dim0, dim1));
}

Tensor Tensor::permute(const Dims& dims) const{
    return Tensor(backend::permute(data_, dims));
}

Tensor Tensor::unsqueeze(Dim dim) const{
    return Tensor(backend::unsqueeze(data_, dim));
}

Tensor Tensor::squeeze(Dim dim) const{
    return Tensor(backend::squeeze(data_, dim));
}

Tensor Tensor::broadcast_to(const Shape& new_shape) const{
    return Tensor(backend::broadcast_to(data_, new_shape));
}

Tensor Tensor::narrow(Dim dim, Index start, std::size_t length) const{
    return Tensor(backend::narrow(data_, dim, start, length));
}

Tensor Tensor::slice(Dim dim, Index start, Index end, std::size_t step) const{
    return Tensor(backend::slice(data_, dim, start, end, step));
}

Tensor Tensor::select(Dim dim, Index index) const{
    return Tensor(backend::select(data_, dim, index));
}

Tensor Tensor::flatten() const{
    return Tensor(backend::flatten(data_));
}

//scalar left-hand operators
Tensor operator*(Scalar scalar, const Tensor& other){
    return other * scalar;
}

Tensor operator+(Scalar scalar, const Tensor& other){
    return other + scalar;
}
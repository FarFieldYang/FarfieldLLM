#include "tensor.h"
#include "backend.h"

Tensor Tensor::operator+(const Tensor& other) const{

}

Tensor Tensor::operator*(const Tensor& other) const{

}

Tensor Tensor::operator+(Scalar scalar) const{

}

Tensor Tensor::operator*(Scalar scalar) const{

}

Tensor Tensor::sum() const{

}

Tensor operator*(const Scalar& scalar, const Tensor& other){

}

Tensor Tensor::exp() const {
    return Tensor(backend::exp(data_));
}
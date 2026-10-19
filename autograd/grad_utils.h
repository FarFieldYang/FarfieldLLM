#pragma once

#include <stdexcept>

#include "tensor.h"

inline void check_grad_shape(const Tensor& grad, const Shape& shape){
    if(grad.shape() != shape)
        throw std::invalid_argument("Gradient shape does not match output shape");
}

inline void check_binary_grad_shape(const Tensor& grad, const Shape& shape1, const Shape& shape2){
    if(grad.shape() != TensorData::broadcast_shape(shape1, shape2))
        throw std::invalid_argument("Gradient shape does not match binary output shape");
}

inline Tensor sum_to_shape(Tensor grad, const Shape& shape){
    if(grad.ndim() < shape.size() || TensorData::broadcast_shape(shape, grad.shape()) != grad.shape())
        throw std::invalid_argument("Gradient cannot be reduced to input shape");

    while(grad.ndim() > shape.size())
        grad = grad.sum(0);

    for(Dim dim = 0; dim < shape.size(); ++dim){
        if(shape[dim] == 1 && grad.shape()[dim] != 1)
            grad = grad.sum(dim).unsqueeze(dim);
    }

    return grad;
}

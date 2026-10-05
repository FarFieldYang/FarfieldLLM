#pragma once

#include "tensor_data.h"

namespace backend {
//binary
TensorData add(const TensorData& input1, const TensorData& input2);
TensorData sub(const TensorData& input1, const TensorData& input2);
TensorData mul(const TensorData& input1, const TensorData& input2);
TensorData div(const TensorData& input1, const TensorData& input2);

//scalar_binary
TensorData add(const TensorData& input, Scalar scalar);
TensorData sub(const TensorData& input, Scalar scalar);
TensorData mul(const TensorData& input, Scalar scalar);
TensorData div(const TensorData& input, Scalar scalar);

//unary
TensorData exp(const TensorData& input);
TensorData log(const TensorData& input);
TensorData sqrt(const TensorData& input);
TensorData tanh(const TensorData& input);

//reduction
TensorData sum(const TensorData& input, Dim dim);
TensorData max(const TensorData& input, Dim dim);

//matrix
TensorData matmul(const TensorData& input1, const TensorData& input2);

//TensorData Backend
TensorData contiguous(const TensorData& input);
//TensorData view
TensorData reshape(const TensorData& input, const Shape& new_shape);
TensorData transpose(const TensorData& input, Dim dim0, Dim dim1);
TensorData permute(const TensorData& input, const Dims& dims);
TensorData unsqueeze(const TensorData& input, Dim dim);
TensorData squeeze(const TensorData& input, Dim dim);
TensorData broadcast_to(const TensorData& input, const Shape& new_shape);
TensorData narrow(const TensorData& input, Dim dim, Index start, std::size_t length);
TensorData slice(const TensorData& input, Dim dim, Index start, Index end, std::size_t step = 1);
TensorData select(const TensorData& input, Dim dim, Index index);
TensorData flatten(const TensorData& input);
}
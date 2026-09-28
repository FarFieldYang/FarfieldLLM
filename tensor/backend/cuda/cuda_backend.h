#pragma once

#include "tensor_data.h"

namespace cuda_backend {
//binary
void add(const TensorData& input1, const TensorData& input2, TensorData& output);
void sub(const TensorData& input1, const TensorData& input2, TensorData& output);
void mul(const TensorData& input1, const TensorData& input2, TensorData& output);
void div(const TensorData& input1, const TensorData& input2, TensorData& output);

//unary
void exp(const TensorData& input, TensorData& output);
void log(const TensorData& input, TensorData& output);
void sqrt(const TensorData& input, TensorData& output);
void tanh(const TensorData& input, TensorData& output);

//reduction
void sum(const TensorData& input, TensorData& output, Dim dim);
void max(const TensorData& input, TensorData& output, Dim dim);

//matrix
void matmul(const TensorData& input1, const TensorData& input2, TensorData& output);
}

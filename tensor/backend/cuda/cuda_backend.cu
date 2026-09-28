#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "cuda_backend.h"

namespace {
    constexpr int BLOCK_SIZE = 256;

    enum class BinaryOp{
        Add,
        Sub,
        Mul,
        Div
    };
    enum class UnaryOp{
        Exp,
        Log,
        Sqrt,
        Tanh
    };

//binary
    template <BinaryOp Op>
    __device__ Scalar apply_binary(Scalar x, Scalar y){
        if constexpr (Op == BinaryOp::Add) return x + y;
        else if constexpr (Op == BinaryOp::Sub) return x - y;
        else if constexpr (Op == BinaryOp::Mul) return x * y;
        else if constexpr (Op == BinaryOp::Div) return x / y;
    }

    template <BinaryOp Op>
    __global__ void binary_kernel(
        const Scalar* input1,
        const Scalar* input2,
        Scalar* output,
        Numel n){
        Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) output[i] = apply_binary<Op>(input1[i], input2[i]);
    }

    template <BinaryOp Op>
    void launch_binary(const TensorData& input1, const TensorData& input2, TensorData& output){
        const Scalar* input_ptr1 =
            static_cast<const Scalar*>(input1.storage().raw_data())
            + input1.offset();
        const Scalar* input_ptr2 =
            static_cast<const Scalar*>(input2.storage().raw_data())
            + input2.offset();
        Scalar* output_ptr =
            static_cast<Scalar*>(output.storage().raw_data())
            + output.offset();
        Numel n = input1.numel();
        if(n == 0) return;
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        binary_kernel<Op><<<blocks, BLOCK_SIZE>>>(input_ptr1, input_ptr2, output_ptr, n);
    }


//unary
    template <UnaryOp Op>
    __device__ Scalar apply_unary(Scalar x){
        if constexpr (Op == UnaryOp::Exp) return expf(x);
        else if constexpr (Op == UnaryOp::Log) return logf(x);
        else if constexpr (Op == UnaryOp::Sqrt) return sqrtf(x);
        else if constexpr (Op == UnaryOp::Tanh) return tanhf(x);
    }

    template <UnaryOp Op>
    __global__ void unary_kernel(
        const Scalar* input,
        Scalar* output,
        Numel n){
        Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) output[i] = apply_unary<Op>(input[i]);
    }

    template <UnaryOp Op>
    void launch_unary(const TensorData& input, TensorData& output){
        const Scalar* input_ptr =
            static_cast<const Scalar*>(input.storage().raw_data())
            + input.offset();
        Scalar* output_ptr =
            static_cast<Scalar*>(output.storage().raw_data())
            + output.offset();
        Numel n = input.numel();
        if(n == 0) return;
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        unary_kernel<Op><<<blocks, BLOCK_SIZE>>>(input_ptr, output_ptr, n);
    }
} //namespace


namespace cuda_backend {
//binary
    void add(const TensorData& input1, const TensorData& input2, TensorData& output){
        launch_binary<BinaryOp::Add>(input1, input2, output);
    }

    void sub(const TensorData& input1, const TensorData& input2, TensorData& output){
        launch_binary<BinaryOp::Sub>(input1, input2, output);
    }

    void mul(const TensorData& input1, const TensorData& input2, TensorData& output){
        launch_binary<BinaryOp::Mul>(input1, input2, output);
    }

    void div(const TensorData& input1, const TensorData& input2, TensorData& output){
        launch_binary<BinaryOp::Div>(input1, input2, output);
    }

//unary
    void exp(const TensorData& input, TensorData& output){
        launch_unary<UnaryOp::Exp>(input, output);
    }
    
    void log(const TensorData& input, TensorData& output){
        launch_unary<UnaryOp::Log>(input, output);
    }
    
    void sqrt(const TensorData& input, TensorData& output){
        launch_unary<UnaryOp::Sqrt>(input, output);
    }

    void tanh(const TensorData& input, TensorData& output){
        launch_unary<UnaryOp::Tanh>(input, output);
    }
}
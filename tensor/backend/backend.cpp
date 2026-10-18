#include <utility>
#include <stdexcept>
#include <cstddef>
#include <memory>

#include "backend.h"
#include "cuda/cuda_backend.h"

namespace {
    Numel shape_numel(const Shape& shape){
        Numel numel = 1;
        for(auto dim : shape) numel *= dim;
        return numel;
    } 

    TensorData empty(const Shape& shape){
        auto storage = std::make_shared<GPUStorage>(shape_numel(shape));
        return TensorData(std::move(storage), shape);
    }    
}

namespace backend {
    TensorData full(const Shape& shape, Scalar scalar){
        TensorData output = empty(shape);
        cuda_backend::fill(output, scalar);
        return output;
    }

//binary
    TensorData add(const TensorData& input1, const TensorData& input2){
        Shape output_shape = TensorData::broadcast_shape(input1.shape(), input2.shape());
        auto a = input1.broadcast_to(output_shape);
        auto b = input2.broadcast_to(output_shape);
        TensorData output = empty(output_shape);
        cuda_backend::add(a, b, output);
        return output;
    }

    TensorData sub(const TensorData& input1, const TensorData& input2){
        Shape output_shape = TensorData::broadcast_shape(input1.shape(), input2.shape());
        auto a = input1.broadcast_to(output_shape);
        auto b = input2.broadcast_to(output_shape);
        TensorData output = empty(output_shape);
        cuda_backend::sub(a, b, output);
        return output;
    }

    TensorData mul(const TensorData& input1, const TensorData& input2){
        Shape output_shape = TensorData::broadcast_shape(input1.shape(), input2.shape());
        auto a = input1.broadcast_to(output_shape);
        auto b = input2.broadcast_to(output_shape);
        TensorData output = empty(output_shape);
        cuda_backend::mul(a, b, output);
        return output;
    }

    TensorData div(const TensorData& input1, const TensorData& input2){
        Shape output_shape = TensorData::broadcast_shape(input1.shape(), input2.shape());
        auto a = input1.broadcast_to(output_shape);
        auto b = input2.broadcast_to(output_shape);
        TensorData output = empty(output_shape);
        cuda_backend::div(a, b, output);
        return output;
    }

//scalar_binary
    TensorData add(const TensorData& input, Scalar scalar){
        Shape output_shape = input.shape();
        TensorData output = empty(output_shape);
        cuda_backend::add(input, scalar,output);
        return output;
    }

    TensorData sub(const TensorData& input, Scalar scalar){
        Shape output_shape = input.shape();
        TensorData output = empty(output_shape);
        cuda_backend::sub(input, scalar,output);
        return output;
    }

    TensorData mul(const TensorData& input, Scalar scalar){
        Shape output_shape = input.shape();
        TensorData output = empty(output_shape);
        cuda_backend::mul(input, scalar,output);
        return output;
    }

    TensorData div(const TensorData& input, Scalar scalar){
        Shape output_shape = input.shape();
        TensorData output = empty(output_shape);
        cuda_backend::div(input, scalar,output);
        return output;
    }

//unary
    TensorData exp(const TensorData& input){
        Shape output_shape = input.shape();
        TensorData output = empty(output_shape);
        cuda_backend::exp(input, output);
        return output;
    }

    TensorData log(const TensorData& input){
        Shape output_shape = input.shape();
        TensorData output = empty(output_shape);
        cuda_backend::log(input, output);
        return output;
    }

    TensorData sqrt(const TensorData& input){
        Shape output_shape = input.shape();
        TensorData output = empty(output_shape);
        cuda_backend::sqrt(input, output);
        return output;
    }

    TensorData tanh(const TensorData& input){
        Shape output_shape = input.shape();
        TensorData output = empty(output_shape);
        cuda_backend::tanh(input, output);
        return output;
    }

    //reduction
    TensorData sum(const TensorData& input, Dim dim){
        if(dim >= input.ndim())
            throw std::out_of_range("sum dimension out of range");

        Shape output_shape = input.shape();
        output_shape.erase(output_shape.begin() + dim);
        TensorData output = empty(output_shape);
        cuda_backend::sum(input, output, dim);
        return output;
    }

    TensorData max(const TensorData& input, Dim dim){
        if(dim >= input.ndim())
            throw std::out_of_range("max dimension out of range");

        Shape output_shape = input.shape();
        output_shape.erase(output_shape.begin() + dim);
        TensorData output = empty(output_shape);
        cuda_backend::max(input, output, dim);
        return output;
    }

    TensorData max_scatter(const TensorData& input, const TensorData& source, Dim dim){
        if(dim >= input.ndim())
            throw std::out_of_range("max_scatter dimension out of range");

        Shape source_shape = input.shape();
        source_shape.erase(source_shape.begin() + dim);
        if(source.shape() != source_shape)
            throw std::invalid_argument("max_scatter source shape mismatch");

        TensorData output = empty(input.shape());
        cuda_backend::max_scatter(input, source, output, dim);
        return output;
    }

    //matrix
    TensorData matmul(const TensorData& input1, const TensorData& input2){
        if(input1.ndim() < 2 || input2.ndim() != input1.ndim())
            throw std::invalid_argument("matmul input ranks must match and be >= 2");

        std::size_t ndim = input1.ndim();

        for(std::size_t i = 0; i < ndim - 2; ++i)
            if(input1.shape()[i] != input2.shape()[i])
                throw std::invalid_argument("matmul batch shapes must match");
        if(input1.shape()[ndim - 1] != input2.shape()[ndim - 2])
            throw std::invalid_argument("matmul inner dimensions must match");
        
        Shape output_shape = input1.shape();
        output_shape[ndim - 1] = input2.shape()[ndim - 1];

        auto a = input1.is_contiguous() ? input1 : contiguous(input1);
        auto b = input2.is_contiguous() ? input2 : contiguous(input2);

        TensorData output = empty(output_shape);
        cuda_backend::matmul(a, b, output);

        return output;
    }

    //TensorData Backend
    TensorData contiguous(const TensorData& input){
        if(input.is_contiguous())
            return input;
        TensorData output = empty(input.shape());
        cuda_backend::contiguous(input, output);
        return output;
    }
    //TensorData view
    TensorData reshape(const TensorData& input, const Shape& new_shape){return input.reshape(new_shape);}
    TensorData transpose(const TensorData& input, Dim dim0, Dim dim1){return input.transpose(dim0, dim1);}
    TensorData permute(const TensorData& input, const Dims& dims){return input.permute(dims);}
    TensorData unsqueeze(const TensorData& input, Dim dim){return input.unsqueeze(dim);}
    TensorData squeeze(const TensorData& input, Dim dim){return input.squeeze(dim);}
    TensorData broadcast_to(const TensorData& input, const Shape& new_shape){return input.broadcast_to(new_shape);}
    TensorData narrow(const TensorData& input, Dim dim, Index start, std::size_t length){return input.narrow(dim, start, length);}
    TensorData slice(const TensorData& input, Dim dim, Index start, Index end, std::size_t step){return input.slice(dim, start, end, step);}
    TensorData slice_scatter(const TensorData& source, const Shape& output_shape, Dim dim, Index start, Index end, std::size_t step){
        if(dim >= output_shape.size())
            throw std::out_of_range("slice_scatter dimension out of range");
        if(step == 0)
            throw std::invalid_argument("slice_scatter step cannot be zero");
        if(start > end)
            throw std::invalid_argument("slice_scatter start cannot be greater than end");
        if(end > output_shape[dim])
            throw std::out_of_range("slice_scatter end out of range");

        Shape source_shape = output_shape;
        std::size_t span = end - start;
        source_shape[dim] = span == 0 ? 0 : 1 + (span - 1) / step;
        if(source.shape() != source_shape)
            throw std::invalid_argument("slice_scatter source shape mismatch");

        TensorData output = empty(output_shape);
        cuda_backend::slice_scatter(source, output, dim, start, end, step);
        return output;
    }

    TensorData select(const TensorData& input, Dim dim, Index index){return input.select(dim, index);}
    TensorData flatten(const TensorData& input){return input.flatten();}
}

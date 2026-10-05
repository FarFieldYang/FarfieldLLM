#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <cublasLt.h>

#include "cuda_backend.h"

namespace {
    constexpr int BLOCK_SIZE = 256;
    constexpr int MAX_DIMS = 8;

    struct TensorMeta{
        int ndim;
        Index shape[MAX_DIMS];
        Index strides[MAX_DIMS];
    };

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
    enum class ReduceOp{
        Sum,
        Max
    };

//index
    TensorMeta make_meta(const TensorData& data){
        if (data.ndim() > MAX_DIMS)
            throw std::invalid_argument("Tensor rank exceeds MAX_DIMS");

        TensorMeta meta{};
        meta.ndim = static_cast<int>(data.ndim());

        for (int d = 0; d < meta.ndim; ++d) {
            meta.shape[d] = data.shape()[d];
            meta.strides[d] = data.strides()[d];
        }

        return meta;
    }

    __device__ Index index_to_position(Index index, const TensorMeta& meta){
        Index position{};
        for(int d = meta.ndim - 1; d >= 0; --d){
            Index coord = index % meta.shape[d];
            index /= meta.shape[d];

            position += coord * meta.strides[d];
        }
        return position;
    }

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
        Numel n,
        TensorMeta meta1,
        TensorMeta meta2){
        Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) {
            Index pos1 = index_to_position(i, meta1);
            Index pos2 = index_to_position(i, meta2);
            output[i] = apply_binary<Op>(input1[pos1], input2[pos2]);
        }
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
        Numel n = output.numel();
        TensorMeta meta1 = make_meta(input1);
        TensorMeta meta2 = make_meta(input2);

        if(n == 0) return;
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        binary_kernel<Op><<<blocks, BLOCK_SIZE>>>(input_ptr1, input_ptr2, output_ptr, n, meta1, meta2);
    }

//scalar_binary
    template <BinaryOp Op>
    __global__ void scalar_binary_kernel(
        const Scalar* input,
        Scalar scalar,
        Scalar* output,
        Numel n,
        TensorMeta meta){
        Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) {
            Index pos = index_to_position(i, meta);
            output[i] = apply_binary<Op>(input[pos], scalar);
        }
    }

    template <BinaryOp Op>
    void launch_scalar_binary(const TensorData& input, Scalar scalar, TensorData& output){
        const Scalar* input_ptr =
            static_cast<const Scalar*>(input.storage().raw_data())
            + input.offset();
        Scalar* output_ptr =
            static_cast<Scalar*>(output.storage().raw_data())
            + output.offset();
        Numel n = output.numel();
        TensorMeta meta = make_meta(input);

        if(n == 0) return;
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        scalar_binary_kernel<Op><<<blocks, BLOCK_SIZE>>>(input_ptr, scalar, output_ptr, n, meta);
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
        Numel n,
        TensorMeta meta){
        Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (i < n) {
            Index pos = index_to_position(i, meta);
            output[i] = apply_unary<Op>(input[pos]);
        }
    }

    template <UnaryOp Op>
    void launch_unary(const TensorData& input, TensorData& output){
        const Scalar* input_ptr =
            static_cast<const Scalar*>(input.storage().raw_data())
            + input.offset();
        Scalar* output_ptr =
            static_cast<Scalar*>(output.storage().raw_data())
            + output.offset();
        Numel n = output.numel();
        TensorMeta meta = make_meta(input);

        if(n == 0) return;
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        unary_kernel<Op><<<blocks, BLOCK_SIZE>>>(input_ptr, output_ptr, n, meta);
    }

//reduction
    template <ReduceOp Op>
    __device__ Scalar apply_reduce(Scalar x, Scalar y){
        if constexpr (Op == ReduceOp::Sum) return x + y;
        else if constexpr (Op == ReduceOp::Max) return fmaxf(x, y);
    }

    template <ReduceOp Op>
    __device__ Scalar reduce_identity() {
        if constexpr (Op == ReduceOp::Sum) return 0.0f;
        else if constexpr (Op == ReduceOp::Max) return -INFINITY;
    }

    __device__ Index index_to_reduce_base(
        Index out_idx,
        Dim reduce_dim,
        const TensorMeta& meta
    ){
        Index idx = out_idx;
        Index pos{};
        for(int d = meta.ndim - 1; d >= 0; --d){
            if (static_cast<Dim>(d) == reduce_dim) continue;
            Index coord = idx % meta.shape[d];
            idx /= meta.shape[d];
            
            pos += coord * meta.strides[d];
        }
        return pos;
    }

    template <ReduceOp Op>
    __global__ void reduce_kernel(
        const Scalar* input,
        Scalar* output,
        TensorMeta meta,
        Dim reduce_dim
    ){
        Index out_idx = static_cast<Index>(blockIdx.x);
        Index base = index_to_reduce_base(out_idx, reduce_dim, meta);


        Scalar local_value = reduce_identity<Op>();
        for(Index i = threadIdx.x; i < meta.shape[reduce_dim]; i += blockDim.x){
            local_value = apply_reduce<Op>(local_value, input[base + i * meta.strides[reduce_dim]]);
        }

        __shared__ Scalar shared[BLOCK_SIZE];
        shared[threadIdx.x] = local_value;
        __syncthreads();

        for(unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2){
            if(threadIdx.x < stride)
                shared[threadIdx.x] = apply_reduce<Op>(shared[threadIdx.x], shared[threadIdx.x + stride]);
            __syncthreads();
        }

        if(threadIdx.x == 0) output[out_idx] = shared[0];
    }

    template <ReduceOp Op>
    void launch_reduce(const TensorData& input, TensorData& output, Dim dim){
        const Scalar* input_ptr =
            static_cast<const Scalar*>(input.storage().raw_data())
            + input.offset();
        Scalar* output_ptr =
            static_cast<Scalar*>(output.storage().raw_data())
            + output.offset();
        TensorMeta meta = make_meta(input);
        Numel n = output.numel();
        if(n == 0) return;
        reduce_kernel<Op><<<static_cast<int>(n), BLOCK_SIZE>>>(input_ptr, output_ptr, meta, dim);
    }

//matrix
    cublasLtHandle_t get_handle(){
        static cublasLtHandle_t handle = []{
            cublasLtHandle_t h;
            cublasLtCreate(&h);
            return h;
        }();
        return handle;
    }

    class MatrixLayout {
    private:
        cublasLtMatrixLayout_t desc_{};

    public:
        MatrixLayout(Index rows, Index cols) {
            cublasStatus_t status = cublasLtMatrixLayoutCreate(
                &desc_,
                CUDA_R_32F,
                rows,
                cols,
                cols
            );

            if (status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error(
                    "Failed to create cuBLASLt matrix layout"
                );

            cublasLtOrder_t order = CUBLASLT_ORDER_ROW;

            status = cublasLtMatrixLayoutSetAttribute(
                desc_,
                CUBLASLT_MATRIX_LAYOUT_ORDER,
                &order,
                sizeof(order)
            );

            if (status != CUBLAS_STATUS_SUCCESS) {
                cublasLtMatrixLayoutDestroy(desc_);
                desc_ = nullptr;

                throw std::runtime_error(
                    "Failed to set cuBLASLt matrix layout order"
                );
            }
        }

        ~MatrixLayout() {
            if (desc_)
                cublasLtMatrixLayoutDestroy(desc_);
        }

        MatrixLayout(const MatrixLayout&) = delete;
        MatrixLayout& operator=(const MatrixLayout&) = delete;

        cublasLtMatrixLayout_t get() const {
            return desc_;
        }

        void set_batch(int32_t batch_count, int64_t batch_stride) {
            cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(
                desc_,
                CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                &batch_count,
                sizeof(batch_count)
            );

            if (status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Failed to set batch count");

            status = cublasLtMatrixLayoutSetAttribute(
                desc_,
                CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                &batch_stride,
                sizeof(batch_stride)
            );

            if (status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("Failed to set batch stride");
        }
    };

    class MatmulDesc {
    private:
        cublasLtMatmulDesc_t desc_{};

    public:
        MatmulDesc() {
            cublasStatus_t status = cublasLtMatmulDescCreate(
                &desc_,
                CUBLAS_COMPUTE_32F,
                CUDA_R_32F
            );

            if (status != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error(
                    "Failed to create cuBLASLt matmul descriptor"
                );
        }

        ~MatmulDesc() {
            if (desc_)
                cublasLtMatmulDescDestroy(desc_);
        }

        MatmulDesc(const MatmulDesc&) = delete;
        MatmulDesc& operator=(const MatmulDesc&) = delete;

        cublasLtMatmulDesc_t get() const {
            return desc_;
        }
    };

    int32_t compute_batch_count(
        const Shape& a_shape,
        const Shape& b_shape,
        std::size_t rank
    ) {
        int32_t batch_count = 1;

        for (std::size_t d = 0; d < rank - 2; ++d) {
            if (a_shape[d] != b_shape[d])
                throw std::invalid_argument("Matmul batch shapes must match");
            batch_count *= static_cast<int32_t>(a_shape[d]);
        }

        return batch_count;
    }

    void launch_matmul(const TensorData& input1, const TensorData& input2, TensorData& output){
        if(input1.ndim() < 2 || input2.ndim() != input1.ndim())
            throw std::invalid_argument("Matmul input ranks must match and be >= 2");
        if(!input1.is_contiguous()
            || !input2.is_contiguous()
            || !output.is_contiguous())
            throw std::invalid_argument("Matmul requires contiguous tensors ");

        std::size_t ndim = input1.ndim();
        Index M = input1.shape()[ndim - 2];
        Index K = input1.shape()[ndim - 1];
        Index N = input2.shape()[ndim - 1];

        if(input2.shape()[ndim - 2] != K)
            throw std::invalid_argument("Matmul inner dimensions must match");

        Shape expected_output_shape = input1.shape();
        expected_output_shape[ndim - 1] = N;
        if(output.shape() != expected_output_shape)
            throw std::invalid_argument("Matmul output shape mismatch");

        const Scalar* input_ptr1 =
            static_cast<const Scalar*>(input1.storage().raw_data())
            + input1.offset();

        const Scalar* input_ptr2 =
            static_cast<const Scalar*>(input2.storage().raw_data())
            + input2.offset();

        Scalar* output_ptr =
            static_cast<Scalar*>(output.storage().raw_data())
            + output.offset();

        cublasLtHandle_t handle = get_handle();
        MatmulDesc op_desc;    
        MatrixLayout input1_desc(M, K);
        MatrixLayout input2_desc(K, N);
        MatrixLayout output_desc(M, N);

        int32_t batch_count = compute_batch_count(input1.shape(), input2.shape(), ndim);
        int64_t input1_batch_stride = static_cast<int64_t>(M) * K;
        int64_t input2_batch_stride = static_cast<int64_t>(K) * N;
        int64_t output_batch_stride = static_cast<int64_t>(M) * N;               
        if(batch_count > 1){
        input1_desc.set_batch(batch_count, input1_batch_stride);
        input2_desc.set_batch(batch_count, input2_batch_stride);
        output_desc.set_batch(batch_count, output_batch_stride);
        }

        Scalar alpha = 1.0f;
        Scalar beta = 0.0f;

        cublasStatus_t status = cublasLtMatmul(
            handle,
            op_desc.get(),

            &alpha,

            input_ptr1,
            input1_desc.get(),

            input_ptr2,
            input2_desc.get(),
            
            &beta,

            output_ptr,
            output_desc.get(),

            output_ptr,
            output_desc.get(),

            nullptr,
            nullptr,
            0,
            nullptr
        );
        if(status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLASLt matmul failed");
    }

//TensorData Backend
    __global__ void contiguous_kernel(const Scalar* input, Scalar* output, Numel n, TensorMeta meta){
        Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
        if(i < n) output[i] = input[index_to_position(i, meta)];
    }

    void launch_contiguous(const TensorData& input, TensorData& output){
        const Scalar* input_ptr = 
            static_cast<const Scalar*>(input.storage().raw_data())
            + input.offset();
        Scalar* output_ptr = 
            static_cast<Scalar*>(output.storage().raw_data())
            + output.offset(); 
        Numel n = output.numel();
        TensorMeta meta = make_meta(input);

        if(n == 0) return;
        int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        contiguous_kernel<<<blocks, BLOCK_SIZE>>>(input_ptr, output_ptr, n, meta);
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

//scalar_backend
    void add(const TensorData& input, Scalar scalar, TensorData& output){
        launch_scalar_binary<BinaryOp::Add>(input, scalar, output);
    }

    void sub(const TensorData& input, Scalar scalar, TensorData& output){
        launch_scalar_binary<BinaryOp::Sub>(input, scalar, output);
    }

    void mul(const TensorData& input, Scalar scalar, TensorData& output){
        launch_scalar_binary<BinaryOp::Mul>(input, scalar, output);
    }
    void div(const TensorData& input, Scalar scalar, TensorData& output){
        launch_scalar_binary<BinaryOp::Div>(input, scalar, output);
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

//reduction
    void sum(const TensorData& input, TensorData& output, Dim dim){
        launch_reduce<ReduceOp::Sum>(input, output, dim);
    }

    void max(const TensorData& input, TensorData& output, Dim dim){
        launch_reduce<ReduceOp::Max>(input, output, dim);
    }

//matrix
    void matmul(const TensorData& input1, const TensorData& input2, TensorData& output){
        launch_matmul(input1, input2, output);
    }
    
//TensorData Backend
    void contiguous(const TensorData& input, TensorData& output){
    launch_contiguous(input, output);
    }
}
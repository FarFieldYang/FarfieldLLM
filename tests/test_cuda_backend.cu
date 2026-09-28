#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "storage/storage.h"
#include "tensor_data.h"
#include "backend/cuda/cuda_backend.h"


namespace {

constexpr Numel N = 32ULL * 1024 * 1024;
constexpr Scalar TOLERANCE = 1e-5f;


void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr
            << "CUDA error in " << what
            << ": " << cudaGetErrorString(err)
            << '\n';
        std::exit(1);
    }
}


template <typename CpuOp>
void test_unary(
    const std::string& name,
    void (*gpu_op)(const TensorData&, TensorData&),
    CpuOp cpu_op,
    const std::vector<Scalar>& host_input,
    std::vector<Scalar>& host_cpu_output,
    std::vector<Scalar>& host_gpu_output,
    TensorData& input,
    TensorData& output
) {
    // --------------------------------------------------------
    // CPU reference + benchmark
    // --------------------------------------------------------

    auto cpu_start =
        std::chrono::high_resolution_clock::now();

    for (Numel i = 0; i < N; ++i) {
        host_cpu_output[i] = cpu_op(host_input[i]);
    }

    auto cpu_end =
        std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();


    // --------------------------------------------------------
    // GPU benchmark
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");

    gpu_op(input, output);

    check_cuda(cudaGetLastError(), "kernel launch");

    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float gpu_ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(&gpu_ms, start, stop),
        "cudaEventElapsedTime"
    );

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");


    // --------------------------------------------------------
    // Copy result back
    // --------------------------------------------------------

    check_cuda(
        cudaMemcpy(
            host_gpu_output.data(),
            output.storage().raw_data(),
            N * sizeof(Scalar),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy DeviceToHost"
    );


    // --------------------------------------------------------
    // Correctness
    // --------------------------------------------------------

    Scalar max_error = 0.0f;

    for (Numel i = 0; i < N; ++i) {
        Scalar error =
            std::abs(host_cpu_output[i] - host_gpu_output[i]);

        if (error > max_error) {
            max_error = error;
        }
    }

    bool passed = max_error <= TOLERANCE;

    std::cout
        << name
        << "\n  CPU:       " << cpu_ms << " ms"
        << "\n  GPU:       " << gpu_ms << " ms"
        << "\n  Speedup:   " << cpu_ms / gpu_ms << "x"
        << "\n  Max error: " << max_error
        << "\n  Result:    " << (passed ? "PASS" : "FAIL")
        << "\n\n";

    if (!passed) {
        std::cerr << name << " failed correctness check.\n";
        std::exit(1);
    }
}


template <typename CpuOp>
void test_binary(
    const std::string& name,
    void (*gpu_op)(
        const TensorData&,
        const TensorData&,
        TensorData&
    ),
    CpuOp cpu_op,
    const std::vector<Scalar>& host_input1,
    const std::vector<Scalar>& host_input2,
    std::vector<Scalar>& host_cpu_output,
    std::vector<Scalar>& host_gpu_output,
    TensorData& input1,
    TensorData& input2,
    TensorData& output
) {
    // --------------------------------------------------------
    // CPU reference + benchmark
    // --------------------------------------------------------

    auto cpu_start =
        std::chrono::high_resolution_clock::now();

    for (Numel i = 0; i < N; ++i) {
        host_cpu_output[i] =
            cpu_op(host_input1[i], host_input2[i]);
    }

    auto cpu_end =
        std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();


    // --------------------------------------------------------
    // GPU benchmark
    // --------------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");

    gpu_op(input1, input2, output);

    check_cuda(cudaGetLastError(), "kernel launch");

    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float gpu_ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(&gpu_ms, start, stop),
        "cudaEventElapsedTime"
    );

    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");


    // --------------------------------------------------------
    // Copy result back
    // --------------------------------------------------------

    check_cuda(
        cudaMemcpy(
            host_gpu_output.data(),
            output.storage().raw_data(),
            N * sizeof(Scalar),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy DeviceToHost"
    );


    // --------------------------------------------------------
    // Correctness
    // --------------------------------------------------------

    Scalar max_error = 0.0f;

    for (Numel i = 0; i < N; ++i) {
        Scalar error =
            std::abs(host_cpu_output[i] - host_gpu_output[i]);

        if (error > max_error) {
            max_error = error;
        }
    }

    bool passed = max_error <= TOLERANCE;

    std::cout
        << name
        << "\n  CPU:       " << cpu_ms << " ms"
        << "\n  GPU:       " << gpu_ms << " ms"
        << "\n  Speedup:   " << cpu_ms / gpu_ms << "x"
        << "\n  Max error: " << max_error
        << "\n  Result:    " << (passed ? "PASS" : "FAIL")
        << "\n\n";

    if (!passed) {
        std::cerr << name << " failed correctness check.\n";
        std::exit(1);
    }
}

} // namespace


int main() {
    std::cout
        << "========================================\n"
        << "FarfieldLLM CUDA Backend Test\n"
        << "========================================\n";

    std::cout
        << "Elements:    " << N << '\n'
        << "Tensor size: "
        << (N * sizeof(Scalar)) / (1024.0 * 1024.0)
        << " MB\n\n";


    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    std::vector<Scalar> host_input1(N);
    std::vector<Scalar> host_input2(N);

    std::vector<Scalar> host_cpu_output(N);
    std::vector<Scalar> host_gpu_output(N);


    // Keep values:
    //
    // - positive for log / sqrt
    // - small enough for exp
    // - input2 never zero for div

    for (Numel i = 0; i < N; ++i) {
        host_input1[i] =
            0.1f +
            static_cast<Scalar>(i % 1000) / 500.0f;

        host_input2[i] =
            0.5f +
            static_cast<Scalar>(i % 777) / 777.0f;
    }


    // --------------------------------------------------------
    // GPU Storage / TensorData
    // --------------------------------------------------------

    Device gpu_device{
        DeviceType::GPU,
        0
    };

    auto input_storage1 =
        std::make_shared<GPUStorage>(
            N,
            gpu_device
        );

    auto input_storage2 =
        std::make_shared<GPUStorage>(
            N,
            gpu_device
        );

    auto output_storage =
        std::make_shared<GPUStorage>(
            N,
            gpu_device
        );

    Shape shape = {N};

    TensorData input1(input_storage1, shape);
    TensorData input2(input_storage2, shape);
    TensorData output(output_storage, shape);


    // --------------------------------------------------------
    // Host -> GPU
    // --------------------------------------------------------

    check_cuda(
        cudaMemcpy(
            input_storage1->raw_data(),
            host_input1.data(),
            N * sizeof(Scalar),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy input1 HostToDevice"
    );

    check_cuda(
        cudaMemcpy(
            input_storage2->raw_data(),
            host_input2.data(),
            N * sizeof(Scalar),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy input2 HostToDevice"
    );


    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    cuda_backend::exp(input1, output);

    check_cuda(
        cudaDeviceSynchronize(),
        "CUDA warmup"
    );


    std::cout
        << "--------------- Unary ------------------\n\n";


    test_unary(
        "exp",
        cuda_backend::exp,
        [](Scalar x) {
            return std::exp(x);
        },
        host_input1,
        host_cpu_output,
        host_gpu_output,
        input1,
        output
    );


    test_unary(
        "log",
        cuda_backend::log,
        [](Scalar x) {
            return std::log(x);
        },
        host_input1,
        host_cpu_output,
        host_gpu_output,
        input1,
        output
    );


    test_unary(
        "sqrt",
        cuda_backend::sqrt,
        [](Scalar x) {
            return std::sqrt(x);
        },
        host_input1,
        host_cpu_output,
        host_gpu_output,
        input1,
        output
    );


    test_unary(
        "tanh",
        cuda_backend::tanh,
        [](Scalar x) {
            return std::tanh(x);
        },
        host_input1,
        host_cpu_output,
        host_gpu_output,
        input1,
        output
    );


    std::cout
        << "--------------- Binary -----------------\n\n";


    test_binary(
        "add",
        cuda_backend::add,
        [](Scalar x, Scalar y) {
            return x + y;
        },
        host_input1,
        host_input2,
        host_cpu_output,
        host_gpu_output,
        input1,
        input2,
        output
    );


    test_binary(
        "sub",
        cuda_backend::sub,
        [](Scalar x, Scalar y) {
            return x - y;
        },
        host_input1,
        host_input2,
        host_cpu_output,
        host_gpu_output,
        input1,
        input2,
        output
    );


    test_binary(
        "mul",
        cuda_backend::mul,
        [](Scalar x, Scalar y) {
            return x * y;
        },
        host_input1,
        host_input2,
        host_cpu_output,
        host_gpu_output,
        input1,
        input2,
        output
    );


    test_binary(
        "div",
        cuda_backend::div,
        [](Scalar x, Scalar y) {
            return x / y;
        },
        host_input1,
        host_input2,
        host_cpu_output,
        host_gpu_output,
        input1,
        input2,
        output
    );


    std::cout
        << "========================================\n"
        << "All CUDA backend tests passed!\n"
        << "========================================\n";

    return 0;
}
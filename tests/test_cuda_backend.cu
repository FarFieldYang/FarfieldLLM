#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
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

constexpr int REDUCE_WARMUP_ITERS = 5;
constexpr int REDUCE_BENCH_ITERS = 30;

using ReduceFn =
    void (*)(const TensorData&, TensorData&, Dim);


// ============================================================
// CUDA helpers
// ============================================================

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr
            << "CUDA error in " << what
            << ": " << cudaGetErrorString(err)
            << '\n';
        std::exit(1);
    }
}


void sync_after_kernel(const char* what) {
    check_cuda(cudaGetLastError(), what);
    check_cuda(
        cudaDeviceSynchronize(),
        "cudaDeviceSynchronize"
    );
}


std::shared_ptr<GPUStorage> make_gpu_storage(
    const std::vector<Scalar>& host,
    Device device
) {
    auto storage =
        std::make_shared<GPUStorage>(
            static_cast<Numel>(host.size()),
            device
        );

    if (!host.empty()) {
        check_cuda(
            cudaMemcpy(
                storage->raw_data(),
                host.data(),
                host.size() * sizeof(Scalar),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy HostToDevice"
        );
    }

    return storage;
}


std::vector<Scalar> copy_contiguous_to_host(
    const TensorData& tensor
) {
    std::vector<Scalar> host(tensor.numel());

    if (tensor.numel() == 0)
        return host;

    const Scalar* src =
        static_cast<const Scalar*>(
            tensor.storage().raw_data()
        ) + tensor.offset();

    check_cuda(
        cudaMemcpy(
            host.data(),
            src,
            tensor.numel() * sizeof(Scalar),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy DeviceToHost"
    );

    return host;
}


void require_close(
    const std::string& name,
    const std::vector<Scalar>& actual,
    const std::vector<Scalar>& expected,
    Scalar tolerance = TOLERANCE
) {
    if (actual.size() != expected.size()) {
        std::cerr
            << name
            << " size mismatch: actual="
            << actual.size()
            << ", expected="
            << expected.size()
            << '\n';
        std::exit(1);
    }

    Scalar max_error = 0.0f;
    Numel bad_index = 0;

    for (Numel i = 0; i < actual.size(); ++i) {
        Scalar error =
            std::abs(actual[i] - expected[i]);

        if (error > max_error) {
            max_error = error;
            bad_index = i;
        }
    }

    if (max_error > tolerance) {
        std::cerr
            << name
            << " FAIL"
            << "\n  max error: " << max_error
            << "\n  index:     " << bad_index
            << "\n  actual:    " << actual[bad_index]
            << "\n  expected:  " << expected[bad_index]
            << '\n';
        std::exit(1);
    }

    std::cout
        << name
        << " PASS"
        << "  (max error " << max_error << ")\n";
}


// ============================================================
// Original contiguous unary / binary tests + benchmarks
// ============================================================

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


    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(
        cudaEventCreate(&start),
        "cudaEventCreate(start)"
    );

    check_cuda(
        cudaEventCreate(&stop),
        "cudaEventCreate(stop)"
    );

    check_cuda(
        cudaEventRecord(start),
        "cudaEventRecord(start)"
    );

    gpu_op(input, output);

    check_cuda(
        cudaGetLastError(),
        "kernel launch"
    );

    check_cuda(
        cudaEventRecord(stop),
        "cudaEventRecord(stop)"
    );

    check_cuda(
        cudaEventSynchronize(stop),
        "cudaEventSynchronize(stop)"
    );

    float gpu_ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(
            &gpu_ms,
            start,
            stop
        ),
        "cudaEventElapsedTime"
    );

    check_cuda(
        cudaEventDestroy(start),
        "cudaEventDestroy(start)"
    );

    check_cuda(
        cudaEventDestroy(stop),
        "cudaEventDestroy(stop)"
    );


    check_cuda(
        cudaMemcpy(
            host_gpu_output.data(),
            output.storage().raw_data(),
            N * sizeof(Scalar),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy DeviceToHost"
    );


    Scalar max_error = 0.0f;

    for (Numel i = 0; i < N; ++i) {
        Scalar error =
            std::abs(
                host_cpu_output[i]
                -
                host_gpu_output[i]
            );

        if (error > max_error)
            max_error = error;
    }

    bool passed =
        max_error <= TOLERANCE;

    std::cout
        << name
        << "\n  CPU:       " << cpu_ms << " ms"
        << "\n  GPU:       " << gpu_ms << " ms"
        << "\n  Speedup:   " << cpu_ms / gpu_ms << "x"
        << "\n  Max error: " << max_error
        << "\n  Result:    "
        << (passed ? "PASS" : "FAIL")
        << "\n\n";

    if (!passed) {
        std::cerr
            << name
            << " failed correctness check.\n";
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
    auto cpu_start =
        std::chrono::high_resolution_clock::now();

    for (Numel i = 0; i < N; ++i) {
        host_cpu_output[i] =
            cpu_op(
                host_input1[i],
                host_input2[i]
            );
    }

    auto cpu_end =
        std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(
            cpu_end - cpu_start
        ).count();


    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(
        cudaEventCreate(&start),
        "cudaEventCreate(start)"
    );

    check_cuda(
        cudaEventCreate(&stop),
        "cudaEventCreate(stop)"
    );

    check_cuda(
        cudaEventRecord(start),
        "cudaEventRecord(start)"
    );

    gpu_op(
        input1,
        input2,
        output
    );

    check_cuda(
        cudaGetLastError(),
        "kernel launch"
    );

    check_cuda(
        cudaEventRecord(stop),
        "cudaEventRecord(stop)"
    );

    check_cuda(
        cudaEventSynchronize(stop),
        "cudaEventSynchronize(stop)"
    );

    float gpu_ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(
            &gpu_ms,
            start,
            stop
        ),
        "cudaEventElapsedTime"
    );

    check_cuda(
        cudaEventDestroy(start),
        "cudaEventDestroy(start)"
    );

    check_cuda(
        cudaEventDestroy(stop),
        "cudaEventDestroy(stop)"
    );


    check_cuda(
        cudaMemcpy(
            host_gpu_output.data(),
            output.storage().raw_data(),
            N * sizeof(Scalar),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy DeviceToHost"
    );


    Scalar max_error = 0.0f;

    for (Numel i = 0; i < N; ++i) {
        Scalar error =
            std::abs(
                host_cpu_output[i]
                -
                host_gpu_output[i]
            );

        if (error > max_error)
            max_error = error;
    }

    bool passed =
        max_error <= TOLERANCE;

    std::cout
        << name
        << "\n  CPU:       " << cpu_ms << " ms"
        << "\n  GPU:       " << gpu_ms << " ms"
        << "\n  Speedup:   " << cpu_ms / gpu_ms << "x"
        << "\n  Max error: " << max_error
        << "\n  Result:    "
        << (passed ? "PASS" : "FAIL")
        << "\n\n";

    if (!passed) {
        std::cerr
            << name
            << " failed correctness check.\n";
        std::exit(1);
    }
}


// ============================================================
// Stride-aware correctness tests
// ============================================================

void test_unary_transpose(
    Device gpu_device
) {
    std::vector<Scalar> host{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    auto input_storage =
        make_gpu_storage(
            host,
            gpu_device
        );

    TensorData base(
        input_storage,
        Shape{2, 3}
    );

    TensorData view =
        base.transpose(0, 1);

    auto output_storage =
        std::make_shared<GPUStorage>(
            view.numel(),
            gpu_device
        );

    TensorData output(
        output_storage,
        view.shape()
    );

    cuda_backend::tanh(
        view,
        output
    );

    sync_after_kernel(
        "unary transpose tanh"
    );

    std::vector<Scalar> logical_view{
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f
    };

    std::vector<Scalar> expected;
    expected.reserve(
        logical_view.size()
    );

    for (Scalar x : logical_view)
        expected.push_back(
            std::tanh(x)
        );

    require_close(
        "stride-aware unary: transpose + tanh",
        copy_contiguous_to_host(output),
        expected
    );
}


void test_binary_transpose(
    Device gpu_device
) {
    std::vector<Scalar> host_a{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    std::vector<Scalar> host_b{
        10.0f, 20.0f,
        30.0f, 40.0f,
        50.0f, 60.0f
    };

    auto storage_a =
        make_gpu_storage(
            host_a,
            gpu_device
        );

    auto storage_b =
        make_gpu_storage(
            host_b,
            gpu_device
        );

    TensorData a_base(
        storage_a,
        Shape{2, 3}
    );

    TensorData a_view =
        a_base.transpose(0, 1);

    TensorData b(
        storage_b,
        Shape{3, 2}
    );

    auto output_storage =
        std::make_shared<GPUStorage>(
            6,
            gpu_device
        );

    TensorData output(
        output_storage,
        Shape{3, 2}
    );

    cuda_backend::add(
        a_view,
        b,
        output
    );

    sync_after_kernel(
        "binary transpose add"
    );

    std::vector<Scalar> expected{
        11.0f, 24.0f,
        32.0f, 45.0f,
        53.0f, 66.0f
    };

    require_close(
        "stride-aware binary: transpose + add",
        copy_contiguous_to_host(output),
        expected
    );
}


void test_binary_broadcast(
    Device gpu_device
) {
    std::vector<Scalar> host_a{
         1.0f,  2.0f,  3.0f,  4.0f,
        10.0f, 20.0f, 30.0f, 40.0f
    };

    std::vector<Scalar> host_b{
        100.0f, 200.0f, 300.0f, 400.0f,
          5.0f,   6.0f,   7.0f,   8.0f,
         -1.0f,  -2.0f,  -3.0f,  -4.0f
    };

    auto storage_a =
        make_gpu_storage(
            host_a,
            gpu_device
        );

    auto storage_b =
        make_gpu_storage(
            host_b,
            gpu_device
        );

    TensorData a(
        storage_a,
        Shape{2, 1, 4}
    );

    TensorData b(
        storage_b,
        Shape{1, 3, 4}
    );

    TensorData a_view =
        a.broadcast_to(
            Shape{2, 3, 4}
        );

    TensorData b_view =
        b.broadcast_to(
            Shape{2, 3, 4}
        );

    auto output_storage =
        std::make_shared<GPUStorage>(
            24,
            gpu_device
        );

    TensorData output(
        output_storage,
        Shape{2, 3, 4}
    );

    cuda_backend::add(
        a_view,
        b_view,
        output
    );

    sync_after_kernel(
        "binary broadcast add"
    );

    std::vector<Scalar> expected;
    expected.reserve(24);

    for (Index i = 0; i < 2; ++i) {
        for (Index j = 0; j < 3; ++j) {
            for (Index k = 0; k < 4; ++k) {
                expected.push_back(
                    host_a[i * 4 + k]
                    +
                    host_b[j * 4 + k]
                );
            }
        }
    }

    require_close(
        "stride-aware binary: broadcast + add",
        copy_contiguous_to_host(output),
        expected
    );
}


void test_sum_all_dims(
    Device gpu_device
) {
    std::vector<Scalar> host(24);

    for (Index i = 0; i < host.size(); ++i)
        host[i] =
            static_cast<Scalar>(i + 1);

    auto storage =
        make_gpu_storage(
            host,
            gpu_device
        );

    TensorData input(
        storage,
        Shape{2, 3, 4}
    );


    {
        auto out_storage =
            std::make_shared<GPUStorage>(
                12,
                gpu_device
            );

        TensorData output(
            out_storage,
            Shape{3, 4}
        );

        cuda_backend::sum(
            input,
            output,
            0
        );

        sync_after_kernel(
            "sum dim0"
        );

        std::vector<Scalar> expected(12);

        for (Index j = 0; j < 3; ++j) {
            for (Index k = 0; k < 4; ++k) {
                expected[j * 4 + k] =
                    host[j * 4 + k]
                    +
                    host[12 + j * 4 + k];
            }
        }

        require_close(
            "reduction sum: dim 0",
            copy_contiguous_to_host(output),
            expected
        );
    }


    {
        auto out_storage =
            std::make_shared<GPUStorage>(
                8,
                gpu_device
            );

        TensorData output(
            out_storage,
            Shape{2, 4}
        );

        cuda_backend::sum(
            input,
            output,
            1
        );

        sync_after_kernel(
            "sum dim1"
        );

        std::vector<Scalar> expected(8);

        for (Index i = 0; i < 2; ++i) {
            for (Index k = 0; k < 4; ++k) {
                Scalar value = 0.0f;

                for (Index j = 0; j < 3; ++j) {
                    value +=
                        host[
                            i * 12
                            +
                            j * 4
                            +
                            k
                        ];
                }

                expected[
                    i * 4 + k
                ] = value;
            }
        }

        require_close(
            "reduction sum: middle dim",
            copy_contiguous_to_host(output),
            expected
        );
    }


    {
        auto out_storage =
            std::make_shared<GPUStorage>(
                6,
                gpu_device
            );

        TensorData output(
            out_storage,
            Shape{2, 3}
        );

        cuda_backend::sum(
            input,
            output,
            2
        );

        sync_after_kernel(
            "sum dim2"
        );

        std::vector<Scalar> expected(6);

        for (Index i = 0; i < 2; ++i) {
            for (Index j = 0; j < 3; ++j) {
                Scalar value = 0.0f;

                for (Index k = 0; k < 4; ++k) {
                    value +=
                        host[
                            i * 12
                            +
                            j * 4
                            +
                            k
                        ];
                }

                expected[
                    i * 3 + j
                ] = value;
            }
        }

        require_close(
            "reduction sum: last dim",
            copy_contiguous_to_host(output),
            expected
        );
    }
}


void test_max(
    Device gpu_device
) {
    std::vector<Scalar> host{
         -5.0f,  2.0f,  3.0f, -4.0f,
         -1.0f, 10.0f, -7.0f,  8.0f,
         -9.0f,  6.0f,  5.0f,  4.0f,

         15.0f, -2.0f, 13.0f,  1.0f,
          0.0f, 12.0f, 17.0f, -8.0f,
          9.0f, 16.0f, -5.0f, 14.0f
    };

    auto storage =
        make_gpu_storage(
            host,
            gpu_device
        );

    TensorData input(
        storage,
        Shape{2, 3, 4}
    );

    auto out_storage =
        std::make_shared<GPUStorage>(
            8,
            gpu_device
        );

    TensorData output(
        out_storage,
        Shape{2, 4}
    );

    cuda_backend::max(
        input,
        output,
        1
    );

    sync_after_kernel(
        "max dim1"
    );

    std::vector<Scalar> expected(8);

    for (Index i = 0; i < 2; ++i) {
        for (Index k = 0; k < 4; ++k) {
            Scalar value =
                host[i * 12 + k];

            for (Index j = 1; j < 3; ++j) {
                value =
                    std::max(
                        value,
                        host[
                            i * 12
                            +
                            j * 4
                            +
                            k
                        ]
                    );
            }

            expected[
                i * 4 + k
            ] = value;
        }
    }

    require_close(
        "reduction max: middle dim",
        copy_contiguous_to_host(output),
        expected
    );
}


void test_reduce_transpose(
    Device gpu_device
) {
    std::vector<Scalar> host(24);

    for (Index i = 0; i < host.size(); ++i)
        host[i] =
            static_cast<Scalar>(i + 1);

    auto storage =
        make_gpu_storage(
            host,
            gpu_device
        );

    TensorData base(
        storage,
        Shape{2, 3, 4}
    );

    TensorData view =
        base.transpose(0, 1);

    auto out_storage =
        std::make_shared<GPUStorage>(
            12,
            gpu_device
        );

    TensorData output(
        out_storage,
        Shape{3, 4}
    );

    cuda_backend::sum(
        view,
        output,
        1
    );

    sync_after_kernel(
        "sum transpose"
    );

    std::vector<Scalar> expected(12);

    for (Index j = 0; j < 3; ++j) {
        for (Index k = 0; k < 4; ++k) {
            expected[
                j * 4 + k
            ] =
                host[j * 4 + k]
                +
                host[
                    12
                    +
                    j * 4
                    +
                    k
                ];
        }
    }

    require_close(
        "stride-aware reduction: transpose + sum",
        copy_contiguous_to_host(output),
        expected
    );
}


void test_reduce_broadcast(
    Device gpu_device
) {
    std::vector<Scalar> host{
        2.0f,
        -3.0f,
        5.0f
    };

    auto storage =
        make_gpu_storage(
            host,
            gpu_device
        );

    TensorData base(
        storage,
        Shape{1, 3}
    );

    TensorData view =
        base.broadcast_to(
            Shape{4, 3}
        );


    auto sum_storage =
        std::make_shared<GPUStorage>(
            3,
            gpu_device
        );

    TensorData sum_output(
        sum_storage,
        Shape{3}
    );

    cuda_backend::sum(
        view,
        sum_output,
        0
    );

    sync_after_kernel(
        "sum broadcast"
    );

    require_close(
        "stride-aware reduction: broadcast stride=0 sum",
        copy_contiguous_to_host(sum_output),
        std::vector<Scalar>{
            8.0f,
            -12.0f,
            20.0f
        }
    );


    auto max_storage =
        std::make_shared<GPUStorage>(
            3,
            gpu_device
        );

    TensorData max_output(
        max_storage,
        Shape{3}
    );

    cuda_backend::max(
        view,
        max_output,
        0
    );

    sync_after_kernel(
        "max broadcast"
    );

    require_close(
        "stride-aware reduction: broadcast stride=0 max",
        copy_contiguous_to_host(max_output),
        host
    );
}


void test_reduce_slice(
    Device gpu_device
) {
    std::vector<Scalar> host{
         1.0f,  2.0f,  3.0f,
         4.0f,  5.0f,  6.0f,

         7.0f,  8.0f,  9.0f,
        10.0f, 11.0f, 12.0f
    };

    auto storage =
        make_gpu_storage(
            host,
            gpu_device
        );

    TensorData base(
        storage,
        Shape{2, 6}
    );

    TensorData view =
        base.slice(
            1,
            1,
            6,
            2
        );


    {
        auto out_storage =
            std::make_shared<GPUStorage>(
                2,
                gpu_device
            );

        TensorData output(
            out_storage,
            Shape{2}
        );

        cuda_backend::sum(
            view,
            output,
            1
        );

        sync_after_kernel(
            "sum slice"
        );

        require_close(
            "stride-aware reduction: slice + sum",
            copy_contiguous_to_host(output),
            std::vector<Scalar>{
                12.0f,
                30.0f
            }
        );
    }


    {
        auto out_storage =
            std::make_shared<GPUStorage>(
                2,
                gpu_device
            );

        TensorData output(
            out_storage,
            Shape{2}
        );

        cuda_backend::max(
            view,
            output,
            1
        );

        sync_after_kernel(
            "max slice"
        );

        require_close(
            "stride-aware reduction: slice + max",
            copy_contiguous_to_host(output),
            std::vector<Scalar>{
                6.0f,
                12.0f
            }
        );
    }
}


void test_reduce_to_scalar(
    Device gpu_device
) {
    std::vector<Scalar> host{
        1.5f,
        2.5f,
        -1.0f,
        4.0f
    };

    auto storage =
        make_gpu_storage(
            host,
            gpu_device
        );

    TensorData input(
        storage,
        Shape{4}
    );

    auto out_storage =
        std::make_shared<GPUStorage>(
            1,
            gpu_device
        );

    TensorData output(
        out_storage,
        Shape{}
    );

    cuda_backend::sum(
        input,
        output,
        0
    );

    sync_after_kernel(
        "sum to scalar"
    );

    require_close(
        "reduction: 1D -> 0D scalar",
        copy_contiguous_to_host(output),
        std::vector<Scalar>{7.0f}
    );

    if (
        output.ndim() != 0
        ||
        output.numel() != 1
    ) {
        std::cerr
            << "0D scalar metadata FAIL"
            << "\n  ndim:  "
            << output.ndim()
            << "\n  numel: "
            << output.numel()
            << '\n';
        std::exit(1);
    }

    std::cout
        << "0D scalar metadata PASS\n";
}


void run_stride_and_reduction_correctness(
    Device gpu_device
) {
    std::cout
        << "========================================\n"
        << "Stride-aware / Reduction Correctness\n"
        << "========================================\n\n";

    test_unary_transpose(
        gpu_device
    );

    test_binary_transpose(
        gpu_device
    );

    test_binary_broadcast(
        gpu_device
    );

    test_sum_all_dims(
        gpu_device
    );

    test_max(
        gpu_device
    );

    test_reduce_transpose(
        gpu_device
    );

    test_reduce_broadcast(
        gpu_device
    );

    test_reduce_slice(
        gpu_device
    );

    test_reduce_to_scalar(
        gpu_device
    );

    std::cout
        << "\nStride-aware and reduction correctness passed!\n\n";
}


// ============================================================
// Original 32M-element benchmarks
// ============================================================

void run_original_benchmarks(
    Device gpu_device
) {
    std::cout
        << "========================================\n"
        << "Original Contiguous CUDA Benchmarks\n"
        << "========================================\n";

    std::cout
        << "Elements:    " << N << '\n'
        << "Tensor size: "
        << (N * sizeof(Scalar))
            / (1024.0 * 1024.0)
        << " MB\n\n";


    std::vector<Scalar> host_input1(N);
    std::vector<Scalar> host_input2(N);

    std::vector<Scalar> host_cpu_output(N);
    std::vector<Scalar> host_gpu_output(N);


    for (Numel i = 0; i < N; ++i) {
        host_input1[i] =
            0.1f
            +
            static_cast<Scalar>(
                i % 1000
            ) / 500.0f;

        host_input2[i] =
            0.5f
            +
            static_cast<Scalar>(
                i % 777
            ) / 777.0f;
    }


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


    Shape shape{N};

    TensorData input1(
        input_storage1,
        shape
    );

    TensorData input2(
        input_storage2,
        shape
    );

    TensorData output(
        output_storage,
        shape
    );


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


    cuda_backend::exp(
        input1,
        output
    );

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
}


// ============================================================
// Reduction benchmarks
// ============================================================

Shape reduced_shape(
    const Shape& shape,
    Dim dim
) {
    Shape out = shape;

    out.erase(
        out.begin()
        +
        static_cast<std::ptrdiff_t>(dim)
    );

    return out;
}


Numel shape_numel(
    const Shape& shape
) {
    Numel n = 1;

    for (Index size : shape)
        n *= size;

    return n;
}


std::shared_ptr<GPUStorage> make_benchmark_storage(
    Numel n,
    Device device
) {
    std::vector<Scalar> host(n);

    for (Numel i = 0; i < n; ++i) {
        host[i] =
            static_cast<Scalar>(
                static_cast<int>(
                    i % 2001
                )
                -
                1000
            ) / 1000.0f;
    }

    return make_gpu_storage(
        host,
        device
    );
}


float benchmark_reduce(
    ReduceFn op,
    const TensorData& input,
    TensorData& output,
    Dim dim
) {
    for (
        int i = 0;
        i < REDUCE_WARMUP_ITERS;
        ++i
    ) {
        op(
            input,
            output,
            dim
        );
    }

    check_cuda(
        cudaGetLastError(),
        "reduction warmup kernel launch"
    );

    check_cuda(
        cudaDeviceSynchronize(),
        "reduction warmup sync"
    );


    cudaEvent_t start;
    cudaEvent_t stop;

    check_cuda(
        cudaEventCreate(&start),
        "cudaEventCreate(start)"
    );

    check_cuda(
        cudaEventCreate(&stop),
        "cudaEventCreate(stop)"
    );

    check_cuda(
        cudaEventRecord(start),
        "cudaEventRecord(start)"
    );


    for (
        int i = 0;
        i < REDUCE_BENCH_ITERS;
        ++i
    ) {
        op(
            input,
            output,
            dim
        );
    }


    check_cuda(
        cudaGetLastError(),
        "reduction benchmark kernel launch"
    );

    check_cuda(
        cudaEventRecord(stop),
        "cudaEventRecord(stop)"
    );

    check_cuda(
        cudaEventSynchronize(stop),
        "cudaEventSynchronize(stop)"
    );


    float total_ms = 0.0f;

    check_cuda(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        ),
        "cudaEventElapsedTime"
    );


    check_cuda(
        cudaEventDestroy(start),
        "cudaEventDestroy(start)"
    );

    check_cuda(
        cudaEventDestroy(stop),
        "cudaEventDestroy(stop)"
    );


    return
        total_ms
        /
        static_cast<float>(
            REDUCE_BENCH_ITERS
        );
}


void print_reduce_result(
    const std::string& case_name,
    const std::string& op_name,
    const TensorData& input,
    const TensorData& output,
    Dim dim,
    float ms
) {
    const double bytes =
        static_cast<double>(
            input.numel()
            +
            output.numel()
        )
        *
        sizeof(Scalar);

    const double logical_bw_gbs =
        (bytes / 1e9)
        /
        (ms / 1000.0);

    const double gelems_per_second =
        (
            static_cast<double>(
                input.numel()
            )
            /
            1e9
        )
        /
        (ms / 1000.0);


    std::cout
        << std::left
        << std::setw(28)
        << case_name

        << std::setw(7)
        << op_name

        << "dim="
        << std::setw(2)
        << dim
        << "  "

        << std::right
        << std::fixed
        << std::setprecision(4)
        << std::setw(9)
        << ms
        << " ms  "

        << std::setprecision(2)
        << std::setw(9)
        << logical_bw_gbs
        << " GB/s  "

        << std::setw(8)
        << gelems_per_second
        << " GElem/s\n";
}


void run_reduce_benchmark_case(
    const std::string& case_name,
    const TensorData& input,
    Dim dim,
    Device device
) {
    Shape out_shape =
        reduced_shape(
            input.shape(),
            dim
        );

    Numel out_numel =
        shape_numel(
            out_shape
        );

    auto output_storage =
        std::make_shared<GPUStorage>(
            out_numel,
            device
        );

    TensorData output(
        output_storage,
        out_shape
    );


    float sum_ms =
        benchmark_reduce(
            cuda_backend::sum,
            input,
            output,
            dim
        );

    print_reduce_result(
        case_name,
        "sum",
        input,
        output,
        dim,
        sum_ms
    );


    float max_ms =
        benchmark_reduce(
            cuda_backend::max,
            input,
            output,
            dim
        );

    print_reduce_result(
        case_name,
        "max",
        input,
        output,
        dim,
        max_ms
    );
}


void print_shape(
    const Shape& shape
) {
    std::cout << '[';

    for (
        std::size_t i = 0;
        i < shape.size();
        ++i
    ) {
        if (i != 0)
            std::cout << ',';

        std::cout
            << shape[i];
    }

    std::cout << ']';
}


void run_reduction_benchmarks(
    Device gpu_device
) {
    std::cout
        << "\n========================================\n"
        << "Reduction Benchmarks\n"
        << "========================================\n"
        << "Warmup iterations: "
        << REDUCE_WARMUP_ITERS
        << '\n'
        << "Timed iterations:  "
        << REDUCE_BENCH_ITERS
        << "\n\n";


    // --------------------------------------------------------
    // 1. Friendly contiguous last-dim reduction
    // --------------------------------------------------------

    {
        Shape shape{
            8192,
            4096
        };

        auto storage =
            make_benchmark_storage(
                8192ULL
                *
                4096ULL,
                gpu_device
            );

        TensorData input(
            storage,
            shape
        );

        std::cout
            << "Case 1 input shape: ";

        print_shape(
            input.shape()
        );

        std::cout
            << "  contiguous last-dim reduction\n";

        run_reduce_benchmark_case(
            "contiguous last dim",
            input,
            1,
            gpu_device
        );

        std::cout << '\n';
    }


    // --------------------------------------------------------
    // 2. Contiguous tensor, but reduce first dimension
    // --------------------------------------------------------

    {
        Shape shape{
            4096,
            8192
        };

        auto storage =
            make_benchmark_storage(
                4096ULL
                *
                8192ULL,
                gpu_device
            );

        TensorData input(
            storage,
            shape
        );

        std::cout
            << "Case 2 input shape: ";

        print_shape(
            input.shape()
        );

        std::cout
            << "  contiguous first-dim reduction\n";

        run_reduce_benchmark_case(
            "contiguous first dim",
            input,
            0,
            gpu_device
        );

        std::cout << '\n';
    }


    // --------------------------------------------------------
    // 3. Genuine non-contiguous transpose view
    // --------------------------------------------------------

    {
        Shape base_shape{
            4096,
            8192
        };

        auto storage =
            make_benchmark_storage(
                4096ULL
                *
                8192ULL,
                gpu_device
            );

        TensorData base(
            storage,
            base_shape
        );

        TensorData view =
            base.transpose(
                0,
                1
            );

        std::cout
            << "Case 3 input shape: ";

        print_shape(
            view.shape()
        );

        std::cout
            << "  transpose view, strides=["
            << view.strides()[0]
            << ','
            << view.strides()[1]
            << "]\n";

        run_reduce_benchmark_case(
            "transpose reduce dim1",
            view,
            1,
            gpu_device
        );

        std::cout << '\n';
    }


    // --------------------------------------------------------
    // 4. Transformer-ish 3D last-dim reduction
    // --------------------------------------------------------

    {
        Shape shape{
            1024,
            8,
            4096
        };

        auto storage =
            make_benchmark_storage(
                1024ULL
                *
                8ULL
                *
                4096ULL,
                gpu_device
            );

        TensorData input(
            storage,
            shape
        );

        std::cout
            << "Case 4 input shape: ";

        print_shape(
            input.shape()
        );

        std::cout
            << "  Transformer-ish last-dim reduction\n";

        run_reduce_benchmark_case(
            "3D last dim",
            input,
            2,
            gpu_device
        );

        std::cout << '\n';
    }


    // --------------------------------------------------------
    // 5. Small reduction width
    // --------------------------------------------------------

    {
        Shape shape{
            262144,
            128
        };

        auto storage =
            make_benchmark_storage(
                262144ULL
                *
                128ULL,
                gpu_device
            );

        TensorData input(
            storage,
            shape
        );

        std::cout
            << "Case 5 input shape: ";

        print_shape(
            input.shape()
        );

        std::cout
            << "  small reduction width\n";

        run_reduce_benchmark_case(
            "reduce width 128",
            input,
            1,
            gpu_device
        );

        std::cout << '\n';
    }


    std::cout
        << "Reduction benchmark notes:\n"
        << "  * GB/s = logical input+output bandwidth.\n"
        << "  * Host<->device copies are NOT timed.\n"
        << "  * Each number is the average of "
        << REDUCE_BENCH_ITERS
        << " launches after warmup.\n";
}

} // namespace


int main() {
    std::cout
        << "========================================\n"
        << "FarfieldLLM CUDA Backend Test\n"
        << "========================================\n\n";


    Device gpu_device{
        DeviceType::GPU,
        0
    };


    // New correctness tests first so stride/reduction bugs
    // fail quickly before the large benchmarks.
    run_stride_and_reduction_correctness(
        gpu_device
    );


    // Preserve the original unary/binary 32M-element tests.
    run_original_benchmarks(
        gpu_device
    );


    // Performance characterization for the new reductions.
    run_reduction_benchmarks(
        gpu_device
    );


    std::cout
        << "\n========================================\n"
        << "All CUDA backend tests passed!\n"
        << "========================================\n";


    return 0;
}

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
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

using ReduceFn = void (*)(const TensorData&, TensorData&, Dim);

// ============================================================
// Helpers
// ============================================================

void check_cuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in " << what << ": "
                  << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

void sync_after_kernel(const char* what) {
    check_cuda(cudaGetLastError(), what);
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

Numel shape_numel(const Shape& shape) {
    Numel n = 1;
    for (Index size : shape)
        n *= size;
    return n;
}

std::shared_ptr<GPUStorage> make_gpu_storage(
    const std::vector<Scalar>& host
) {
    auto storage = std::make_shared<GPUStorage>(host.size());

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

TensorData make_output(const Shape& shape) {
    return TensorData(
        std::make_shared<GPUStorage>(shape_numel(shape)),
        shape
    );
}

std::vector<Scalar> copy_contiguous_to_host(
    const TensorData& tensor
) {
    std::vector<Scalar> host(tensor.numel());
    if (host.empty())
        return host;

    const Scalar* src =
        static_cast<const Scalar*>(tensor.storage().raw_data())
        + tensor.offset();

    check_cuda(
        cudaMemcpy(
            host.data(),
            src,
            host.size() * sizeof(Scalar),
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
    Scalar atol = TOLERANCE,
    Scalar rtol = 0.0f
) {
    if (actual.size() != expected.size()) {
        std::cerr << name << " size mismatch: actual="
                  << actual.size() << ", expected="
                  << expected.size() << '\n';
        std::exit(1);
    }

    Scalar max_error = 0.0f;

    for (Numel i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            std::cerr << name << " FAIL: non-finite value at " << i
                      << ", actual=" << actual[i]
                      << ", expected=" << expected[i] << '\n';
            std::exit(1);
        }

        Scalar error = std::abs(actual[i] - expected[i]);
        max_error = std::max(max_error, error);

        if (error > atol + rtol * std::abs(expected[i])) {
            std::cerr << name << " FAIL"
                      << "\n  index:     " << i
                      << "\n  actual:    " << actual[i]
                      << "\n  expected:  " << expected[i]
                      << "\n  error:     " << error << '\n';
            std::exit(1);
        }
    }

    std::cout << name << " PASS (max error "
              << max_error << ")\n";
}

template <typename Fn>
void require_invalid_argument(const std::string& name, Fn fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        std::cout << name << " PASS\n";
        return;
    } catch (const std::exception& error) {
        std::cerr << name << " FAIL: wrong exception: "
                  << error.what() << '\n';
        std::exit(1);
    }

    std::cerr << name << " FAIL: expected std::invalid_argument\n";
    std::exit(1);
}

// ============================================================
// Existing stride-aware correctness tests
// ============================================================

void test_unary_transpose() {
    TensorData base(
        make_gpu_storage({1, 2, 3, 4, 5, 6}),
        Shape{2, 3}
    );
    TensorData view = base.transpose(0, 1);
    TensorData output = make_output(view.shape());

    cuda_backend::tanh(view, output);
    sync_after_kernel("unary transpose tanh");

    std::vector<Scalar> expected{1, 4, 2, 5, 3, 6};
    for (Scalar& x : expected)
        x = std::tanh(x);

    require_close(
        "stride-aware unary: transpose + tanh",
        copy_contiguous_to_host(output),
        expected
    );
}

void test_binary_transpose() {
    TensorData base(
        make_gpu_storage({1, 2, 3, 4, 5, 6}),
        Shape{2, 3}
    );
    TensorData view = base.transpose(0, 1);
    TensorData b(
        make_gpu_storage({10, 20, 30, 40, 50, 60}),
        Shape{3, 2}
    );
    TensorData output = make_output(Shape{3, 2});

    cuda_backend::add(view, b, output);
    sync_after_kernel("binary transpose add");

    require_close(
        "stride-aware binary: transpose + add",
        copy_contiguous_to_host(output),
        {11, 24, 32, 45, 53, 66}
    );
}

void test_binary_broadcast() {
    std::vector<Scalar> host_a{
        1, 2, 3, 4,
        10, 20, 30, 40
    };
    std::vector<Scalar> host_b{
        100, 200, 300, 400,
        5, 6, 7, 8,
        -1, -2, -3, -4
    };

    TensorData a(make_gpu_storage(host_a), Shape{2, 1, 4});
    TensorData b(make_gpu_storage(host_b), Shape{1, 3, 4});
    TensorData av = a.broadcast_to(Shape{2, 3, 4});
    TensorData bv = b.broadcast_to(Shape{2, 3, 4});
    TensorData output = make_output(Shape{2, 3, 4});

    cuda_backend::add(av, bv, output);
    sync_after_kernel("binary broadcast add");

    std::vector<Scalar> expected;
    for (Index i = 0; i < 2; ++i)
        for (Index j = 0; j < 3; ++j)
            for (Index k = 0; k < 4; ++k)
                expected.push_back(
                    host_a[i * 4 + k] + host_b[j * 4 + k]
                );

    require_close(
        "stride-aware binary: broadcast + add",
        copy_contiguous_to_host(output),
        expected
    );
}

void test_sum_all_dims() {
    std::vector<Scalar> host(24);
    for (Index i = 0; i < host.size(); ++i)
        host[i] = static_cast<Scalar>(i + 1);

    TensorData input(make_gpu_storage(host), Shape{2, 3, 4});

    {
        TensorData output = make_output(Shape{3, 4});
        std::vector<Scalar> expected(12);

        for (Index j = 0; j < 3; ++j)
            for (Index k = 0; k < 4; ++k)
                expected[j * 4 + k] =
                    host[j * 4 + k] + host[12 + j * 4 + k];

        cuda_backend::sum(input, output, 0);
        sync_after_kernel("sum dim0");
        require_close(
            "reduction sum: dim 0",
            copy_contiguous_to_host(output),
            expected
        );
    }

    {
        TensorData output = make_output(Shape{2, 4});
        std::vector<Scalar> expected(8, 0.0f);

        for (Index i = 0; i < 2; ++i)
            for (Index k = 0; k < 4; ++k)
                for (Index j = 0; j < 3; ++j)
                    expected[i * 4 + k] += host[i * 12 + j * 4 + k];

        cuda_backend::sum(input, output, 1);
        sync_after_kernel("sum dim1");
        require_close(
            "reduction sum: middle dim",
            copy_contiguous_to_host(output),
            expected
        );
    }

    {
        TensorData output = make_output(Shape{2, 3});
        std::vector<Scalar> expected(6, 0.0f);

        for (Index i = 0; i < 2; ++i)
            for (Index j = 0; j < 3; ++j)
                for (Index k = 0; k < 4; ++k)
                    expected[i * 3 + j] += host[i * 12 + j * 4 + k];

        cuda_backend::sum(input, output, 2);
        sync_after_kernel("sum dim2");
        require_close(
            "reduction sum: last dim",
            copy_contiguous_to_host(output),
            expected
        );
    }
}

void test_max() {
    std::vector<Scalar> host{
        -5, 2, 3, -4,
        -1, 10, -7, 8,
        -9, 6, 5, 4,
        15, -2, 13, 1,
        0, 12, 17, -8,
        9, 16, -5, 14
    };
    TensorData input(make_gpu_storage(host), Shape{2, 3, 4});
    TensorData output = make_output(Shape{2, 4});
    std::vector<Scalar> expected(8);

    for (Index i = 0; i < 2; ++i) {
        for (Index k = 0; k < 4; ++k) {
            Scalar value = host[i * 12 + k];
            for (Index j = 1; j < 3; ++j)
                value = std::max(value, host[i * 12 + j * 4 + k]);
            expected[i * 4 + k] = value;
        }
    }

    cuda_backend::max(input, output, 1);
    sync_after_kernel("max dim1");
    require_close(
        "reduction max: middle dim",
        copy_contiguous_to_host(output),
        expected
    );
}

void test_reduce_transpose() {
    std::vector<Scalar> host(24);
    for (Index i = 0; i < host.size(); ++i)
        host[i] = static_cast<Scalar>(i + 1);

    TensorData base(make_gpu_storage(host), Shape{2, 3, 4});
    TensorData view = base.transpose(0, 1);
    TensorData output = make_output(Shape{3, 4});
    std::vector<Scalar> expected(12);

    for (Index j = 0; j < 3; ++j)
        for (Index k = 0; k < 4; ++k)
            expected[j * 4 + k] =
                host[j * 4 + k] + host[12 + j * 4 + k];

    cuda_backend::sum(view, output, 1);
    sync_after_kernel("sum transpose");
    require_close(
        "stride-aware reduction: transpose + sum",
        copy_contiguous_to_host(output),
        expected
    );
}

void test_reduce_broadcast() {
    std::vector<Scalar> host{2, -3, 5};
    TensorData base(make_gpu_storage(host), Shape{1, 3});
    TensorData view = base.broadcast_to(Shape{4, 3});

    TensorData sum_output = make_output(Shape{3});
    cuda_backend::sum(view, sum_output, 0);
    sync_after_kernel("sum broadcast");
    require_close(
        "stride-aware reduction: broadcast stride=0 sum",
        copy_contiguous_to_host(sum_output),
        {8, -12, 20}
    );

    TensorData max_output = make_output(Shape{3});
    cuda_backend::max(view, max_output, 0);
    sync_after_kernel("max broadcast");
    require_close(
        "stride-aware reduction: broadcast stride=0 max",
        copy_contiguous_to_host(max_output),
        host
    );
}

void test_reduce_slice() {
    TensorData base(
        make_gpu_storage({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
        Shape{2, 6}
    );
    TensorData view = base.slice(1, 1, 6, 2);

    TensorData sum_output = make_output(Shape{2});
    cuda_backend::sum(view, sum_output, 1);
    sync_after_kernel("sum slice");
    require_close(
        "stride-aware reduction: slice + sum",
        copy_contiguous_to_host(sum_output),
        {12, 30}
    );

    TensorData max_output = make_output(Shape{2});
    cuda_backend::max(view, max_output, 1);
    sync_after_kernel("max slice");
    require_close(
        "stride-aware reduction: slice + max",
        copy_contiguous_to_host(max_output),
        {6, 12}
    );
}

void test_reduce_to_scalar() {
    TensorData input(
        make_gpu_storage({1.5f, 2.5f, -1.0f, 4.0f}),
        Shape{4}
    );
    TensorData output = make_output(Shape{});

    cuda_backend::sum(input, output, 0);
    sync_after_kernel("sum to scalar");
    require_close(
        "reduction: 1D -> 0D scalar",
        copy_contiguous_to_host(output),
        {7}
    );

    if (output.ndim() != 0 || output.numel() != 1) {
        std::cerr << "0D scalar metadata FAIL\n";
        std::exit(1);
    }
    std::cout << "0D scalar metadata PASS\n";
}

void run_stride_and_reduction_correctness() {
    std::cout << "\nStride-aware / Reduction Correctness\n";
    test_unary_transpose();
    test_binary_transpose();
    test_binary_broadcast();
    test_sum_all_dims();
    test_max();
    test_reduce_transpose();
    test_reduce_broadcast();
    test_reduce_slice();
    test_reduce_to_scalar();
}

// ============================================================
// Existing 2D matmul correctness tests
// ============================================================

std::vector<Scalar> cpu_matmul(
    const std::vector<Scalar>& a,
    const std::vector<Scalar>& b,
    Index m,
    Index k,
    Index n
) {
    std::vector<Scalar> output(m * n);

    for (Index row = 0; row < m; ++row) {
        for (Index col = 0; col < n; ++col) {
            double value = 0.0;
            for (Index inner = 0; inner < k; ++inner) {
                value += static_cast<double>(a[row * k + inner])
                       * static_cast<double>(b[inner * n + col]);
            }
            output[row * n + col] = static_cast<Scalar>(value);
        }
    }
    return output;
}

void test_matmul_case(
    const std::string& name,
    const std::vector<Scalar>& a_host,
    const std::vector<Scalar>& b_host,
    Index m,
    Index k,
    Index n
) {
    TensorData a(make_gpu_storage(a_host), Shape{m, k});
    TensorData b(make_gpu_storage(b_host), Shape{k, n});
    TensorData output(
        make_gpu_storage(std::vector<Scalar>(m * n, 123.0f)),
        Shape{m, n}
    );

    std::cout << "Running " << name << '\n';
    cuda_backend::matmul(a, b, output);
    sync_after_kernel(name.c_str());

    require_close(
        name,
        copy_contiguous_to_host(output),
        cpu_matmul(a_host, b_host, m, k, n),
        1e-4f,
        1e-4f
    );
}

void test_matmul_offset() {
    const std::vector<Scalar> a_host{
        1, 2, 3, 4,
        -1, 0, 2, -2
    };
    const std::vector<Scalar> b_host{
        1, 2, 3, 4,
        5, 6, 7, 8,
        -1, -2, -3, -4,
        2, 0, 1, -1
    };

    std::vector<Scalar> a_padded(4 * 4, -777.0f);
    std::vector<Scalar> b_padded(6 * 4, -888.0f);
    std::copy(a_host.begin(), a_host.end(), a_padded.begin() + 4);
    std::copy(b_host.begin(), b_host.end(), b_padded.begin() + 4);

    TensorData a_base(make_gpu_storage(a_padded), Shape{4, 4});
    TensorData b_base(make_gpu_storage(b_padded), Shape{6, 4});
    TensorData a = a_base.narrow(0, 1, 2);
    TensorData b = b_base.narrow(0, 1, 4);

    TensorData output_base(
        make_gpu_storage(std::vector<Scalar>(4 * 4, 999.0f)),
        Shape{4, 4}
    );
    TensorData output = output_base.narrow(0, 1, 2);

    if (!a.is_contiguous() || !b.is_contiguous()
        || !output.is_contiguous()
        || a.offset() == 0 || b.offset() == 0
        || output.offset() == 0) {
        std::cerr << "matmul offset test setup FAIL\n";
        std::exit(1);
    }

    std::cout << "Running matmul: contiguous offset views\n";
    cuda_backend::matmul(a, b, output);
    sync_after_kernel("matmul contiguous offset views");

    require_close(
        "matmul: contiguous offset views",
        copy_contiguous_to_host(output),
        cpu_matmul(a_host, b_host, 2, 4, 4),
        1e-4f,
        1e-4f
    );

    std::vector<Scalar> whole = copy_contiguous_to_host(output_base);
    for (Index i = 0; i < 4; ++i) {
        if (whole[i] != 999.0f || whole[12 + i] != 999.0f) {
            std::cerr << "matmul output guard rows FAIL\n";
            std::exit(1);
        }
    }
    std::cout << "matmul: output guard rows PASS\n";
}

void run_matmul_correctness() {
    std::cout << "\n2D Matmul Correctness\n";

    test_matmul_case(
        "matmul: rectangular 2x3 @ 3x4",
        {1, 2, 3, -1, 0, 2},
        {
            1, 2, 3, 4,
            5, 6, 7, 8,
            -1, -2, -3, -4
        },
        2, 3, 4
    );

    test_matmul_case(
        "matmul: right identity",
        {
            1, -2, 3, 4,
            5, 6, -7, 8,
            -9, 10, 11, -12
        },
        {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        },
        3, 4, 4
    );

    {
        constexpr Index m = 17;
        constexpr Index k = 23;
        constexpr Index n = 19;

        std::vector<Scalar> a(m * k);
        std::vector<Scalar> b(k * n);

        for (Index i = 0; i < a.size(); ++i)
            a[i] = static_cast<Scalar>(
                static_cast<int>((i * 7) % 31) - 15
            ) / 8.0f;

        for (Index i = 0; i < b.size(); ++i)
            b[i] = static_cast<Scalar>(
                static_cast<int>((i * 11) % 29) - 14
            ) / 8.0f;

        test_matmul_case(
            "matmul: irregular 17x23 @ 23x19",
            a, b, m, k, n
        );
    }

    test_matmul_offset();
}

// ============================================================
// New contiguous high-dimensional matmul tests
// Batch shapes match exactly; no batch broadcasting.
// ============================================================

Shape matrix_shape(
    const Shape& batch_shape,
    Index rows,
    Index cols
) {
    Shape shape = batch_shape;
    shape.push_back(rows);
    shape.push_back(cols);
    return shape;
}

std::vector<Scalar> make_matmul_data(
    Numel batches,
    Index rows,
    Index cols,
    Index seed
) {
    const Numel matrix_size = rows * cols;
    std::vector<Scalar> host(batches * matrix_size);

    for (Index batch = 0; batch < batches; ++batch) {
        for (Index i = 0; i < matrix_size; ++i) {
            // Different batches have different data.
            const int value = static_cast<int>(
                (i * 7 + batch * 11 + seed) % 31
            ) - 15;

            host[batch * matrix_size + i] =
                static_cast<Scalar>(value) / 8.0f
                + static_cast<Scalar>(batch) / 16.0f;
        }
    }
    return host;
}

std::vector<Scalar> cpu_batched_matmul(
    const std::vector<Scalar>& a,
    const std::vector<Scalar>& b,
    Numel batches,
    Index m,
    Index k,
    Index n
) {
    std::vector<Scalar> output(batches * m * n);

    for (Index batch = 0; batch < batches; ++batch) {
        for (Index row = 0; row < m; ++row) {
            for (Index col = 0; col < n; ++col) {
                double value = 0.0;

                for (Index inner = 0; inner < k; ++inner) {
                    value += static_cast<double>(
                        a[(batch * m + row) * k + inner]
                    ) * static_cast<double>(
                        b[(batch * k + inner) * n + col]
                    );
                }

                output[(batch * m + row) * n + col] =
                    static_cast<Scalar>(value);
            }
        }
    }

    return output;
}

void require_all_batches_close(
    const std::string& name,
    const std::vector<Scalar>& actual,
    const std::vector<Scalar>& expected,
    Numel batches,
    Numel matrix_size
) {
    if (actual.size() != batches * matrix_size
        || expected.size() != batches * matrix_size) {
        std::cerr << name << " FAIL: batch output size mismatch\n";
        std::exit(1);
    }

    for (Index batch = 0; batch < batches; ++batch) {
        const auto begin =
            static_cast<std::ptrdiff_t>(batch * matrix_size);
        const auto end =
            static_cast<std::ptrdiff_t>((batch + 1) * matrix_size);

        require_close(
            name + " batch " + std::to_string(batch),
            std::vector<Scalar>(
                actual.begin() + begin, actual.begin() + end
            ),
            std::vector<Scalar>(
                expected.begin() + begin, expected.begin() + end
            ),
            1e-4f,
            1e-4f
        );
    }
}

void test_batched_matmul_case(
    const std::string& name,
    const Shape& batch_shape,
    Index m,
    Index k,
    Index n
) {
    const Numel batches = shape_numel(batch_shape);
    const auto a_host = make_matmul_data(batches, m, k, 3);
    const auto b_host = make_matmul_data(batches, k, n, 17);

    TensorData a(
        make_gpu_storage(a_host),
        matrix_shape(batch_shape, m, k)
    );
    TensorData b(
        make_gpu_storage(b_host),
        matrix_shape(batch_shape, k, n)
    );
    TensorData output(
        make_gpu_storage(
            std::vector<Scalar>(batches * m * n, 123.0f)
        ),
        matrix_shape(batch_shape, m, n)
    );

    if (!a.is_contiguous()
        || !b.is_contiguous()
        || !output.is_contiguous()) {
        std::cerr << name << " FAIL: test setup is not contiguous\n";
        std::exit(1);
    }

    std::cout << "Running " << name
              << " (batches=" << batches << ")\n";

    cuda_backend::matmul(a, b, output);
    sync_after_kernel(name.c_str());

    require_all_batches_close(
        name,
        copy_contiguous_to_host(output),
        cpu_batched_matmul(a_host, b_host, batches, m, k, n),
        batches,
        m * n
    );
}

void test_batched_matmul_offset() {
    // Logical shapes:
    // A: [2, 2, 4, 8]
    // B: [2, 2, 8, 4]
    // D: [2, 2, 4, 4]
    //
    // Each view is obtained by narrowing the first dimension
    // of a padded base tensor. All views remain contiguous.
    constexpr Index m = 4;
    constexpr Index k = 8;
    constexpr Index n = 4;
    constexpr Numel batches = 4;

    const Numel a_prefix = 2 * m * k;
    const Numel b_prefix = 2 * k * n;
    const Numel output_prefix = 2 * m * n;

    const auto a_host = make_matmul_data(batches, m, k, 5);
    const auto b_host = make_matmul_data(batches, k, n, 19);

    std::vector<Scalar> a_padded(4 * 2 * m * k, -777.0f);
    std::vector<Scalar> b_padded(4 * 2 * k * n, -888.0f);

    std::copy(
        a_host.begin(), a_host.end(),
        a_padded.begin() + static_cast<std::ptrdiff_t>(a_prefix)
    );
    std::copy(
        b_host.begin(), b_host.end(),
        b_padded.begin() + static_cast<std::ptrdiff_t>(b_prefix)
    );

    TensorData a_base(
        make_gpu_storage(a_padded),
        Shape{4, 2, m, k}
    );
    TensorData b_base(
        make_gpu_storage(b_padded),
        Shape{4, 2, k, n}
    );
    TensorData output_base(
        make_gpu_storage(
            std::vector<Scalar>(4 * 2 * m * n, 999.0f)
        ),
        Shape{4, 2, m, n}
    );

    TensorData a = a_base.narrow(0, 1, 2);
    TensorData b = b_base.narrow(0, 1, 2);
    TensorData output = output_base.narrow(0, 1, 2);

    if (!a.is_contiguous()
        || !b.is_contiguous()
        || !output.is_contiguous()
        || a.offset() != a_prefix
        || b.offset() != b_prefix
        || output.offset() != output_prefix) {
        std::cerr << "batched matmul offset test setup FAIL\n";
        std::exit(1);
    }

    std::cout << "Running matmul: 4D contiguous offset views\n";

    cuda_backend::matmul(a, b, output);
    sync_after_kernel("4D matmul offset views");

    const auto expected =
        cpu_batched_matmul(a_host, b_host, batches, m, k, n);

    require_all_batches_close(
        "matmul: 4D contiguous offset views",
        copy_contiguous_to_host(output),
        expected,
        batches,
        m * n
    );

    // Check the entire allocation, including untouched guard regions.
    std::vector<Scalar> expected_whole(4 * 2 * m * n, 999.0f);
    std::copy(
        expected.begin(), expected.end(),
        expected_whole.begin()
            + static_cast<std::ptrdiff_t>(output_prefix)
    );

    require_close(
        "matmul: 4D output guard regions",
        copy_contiguous_to_host(output_base),
        expected_whole,
        1e-4f,
        1e-4f
    );
}

void test_matmul_invalid_arguments() {
    std::cout << "\nMatmul Argument Validation\n";

    {
        TensorData a = make_output(Shape{4});
        TensorData b = make_output(Shape{4});
        TensorData output = make_output(Shape{});

        require_invalid_argument("matmul: reject rank < 2", [&] {
            cuda_backend::matmul(a, b, output);
        });
    }

    {
        TensorData a = make_output(Shape{2, 4, 4});
        TensorData b = make_output(Shape{4, 4});
        TensorData output = make_output(Shape{2, 4, 4});

        require_invalid_argument("matmul: reject unequal ranks", [&] {
            cuda_backend::matmul(a, b, output);
        });
    }

    {
        TensorData a = make_output(Shape{2, 4, 8});
        TensorData b = make_output(Shape{2, 7, 4});
        TensorData output = make_output(Shape{2, 4, 4});

        require_invalid_argument(
            "matmul: reject mismatched inner dimensions",
            [&] { cuda_backend::matmul(a, b, output); }
        );
    }

    {
        // Batch totals are both 6, but batch shapes differ.
        // Matching only the batch product would be incorrect.
        TensorData a = make_output(Shape{2, 3, 4, 4});
        TensorData b = make_output(Shape{3, 2, 4, 4});
        TensorData output = make_output(Shape{2, 3, 4, 4});

        require_invalid_argument(
            "matmul: reject unequal batch shapes with equal totals",
            [&] { cuda_backend::matmul(a, b, output); }
        );
    }

    {
        TensorData a = make_output(Shape{2, 4, 8});
        TensorData b = make_output(Shape{2, 8, 4});
        TensorData output = make_output(Shape{2, 4, 5});

        require_invalid_argument(
            "matmul: reject wrong output shape",
            [&] { cuda_backend::matmul(a, b, output); }
        );
    }

    {
        TensorData a_base = make_output(Shape{2, 4, 8});
        TensorData a = a_base.transpose(1, 2);
        TensorData b = make_output(Shape{2, 4, 4});
        TensorData output = make_output(Shape{2, 8, 4});

        require_invalid_argument(
            "matmul: reject non-contiguous input1",
            [&] { cuda_backend::matmul(a, b, output); }
        );
    }

    {
        TensorData a = make_output(Shape{2, 4, 8});
        TensorData b_base = make_output(Shape{2, 4, 8});
        TensorData b = b_base.transpose(1, 2);
        TensorData output = make_output(Shape{2, 4, 4});

        require_invalid_argument(
            "matmul: reject non-contiguous input2",
            [&] { cuda_backend::matmul(a, b, output); }
        );
    }

    {
        TensorData a = make_output(Shape{2, 4, 8});
        TensorData b = make_output(Shape{2, 8, 4});
        TensorData output_base = make_output(Shape{2, 4, 4});
        TensorData output = output_base.transpose(1, 2);

        require_invalid_argument(
            "matmul: reject non-contiguous output",
            [&] { cuda_backend::matmul(a, b, output); }
        );
    }
}

void run_batched_matmul_correctness() {
    std::cout << "\nHigh-dimensional Contiguous Matmul Correctness\n";

    test_batched_matmul_case(
        "matmul: 3D [3,4,8] @ [3,8,4]",
        Shape{3}, 4, 8, 4
    );

    test_batched_matmul_case(
        "matmul: 4D [2,3,4,8] @ [2,3,8,4]",
        Shape{2, 3}, 4, 8, 4
    );

    test_batched_matmul_case(
        "matmul: 5D [2,3,2,4,8] @ [2,3,2,8,4]",
        Shape{2, 3, 2}, 4, 8, 4
    );

    test_batched_matmul_case(
        "matmul: singleton batch dim [2,1,3,4,8]",
        Shape{2, 1, 3}, 4, 8, 4
    );

    test_batched_matmul_case(
        "matmul: high rank with batch count 1",
        Shape{1, 1, 1}, 4, 8, 4
    );

    test_batched_matmul_case(
        "matmul: irregular batched 17x23 @ 23x19",
        Shape{3}, 17, 23, 19
    );

    test_batched_matmul_offset();
    test_matmul_invalid_arguments();
}

// ============================================================
// Existing 32M-element unary / binary tests and benchmarks
// ============================================================

template <typename Fn>
float time_gpu_once(Fn fn) {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check_cuda(cudaEventRecord(start), "cudaEventRecord(start)");

    fn();

    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms = 0.0f;
    check_cuda(
        cudaEventElapsedTime(&ms, start, stop),
        "cudaEventElapsedTime"
    );
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
    return ms;
}

void report_elementwise(
    const std::string& name,
    double cpu_ms,
    float gpu_ms,
    const std::vector<Scalar>& actual,
    const std::vector<Scalar>& expected
) {
    require_close(name, actual, expected);

    std::cout << "  CPU:     " << cpu_ms << " ms"
              << "\n  GPU:     " << gpu_ms << " ms"
              << "\n  Speedup: " << cpu_ms / gpu_ms << "x\n\n";
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
    auto start = std::chrono::high_resolution_clock::now();
    for (Numel i = 0; i < N; ++i)
        host_cpu_output[i] = cpu_op(host_input[i]);
    auto stop = std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    float gpu_ms = time_gpu_once([&] {
        gpu_op(input, output);
    });

    const Scalar* src =
        static_cast<const Scalar*>(output.storage().raw_data())
        + output.offset();

    check_cuda(
        cudaMemcpy(
            host_gpu_output.data(), src,
            N * sizeof(Scalar), cudaMemcpyDeviceToHost
        ),
        "unary output copy"
    );

    report_elementwise(
        name, cpu_ms, gpu_ms, host_gpu_output, host_cpu_output
    );
}

template <typename CpuOp>
void test_binary(
    const std::string& name,
    void (*gpu_op)(const TensorData&, const TensorData&, TensorData&),
    CpuOp cpu_op,
    const std::vector<Scalar>& host_input1,
    const std::vector<Scalar>& host_input2,
    std::vector<Scalar>& host_cpu_output,
    std::vector<Scalar>& host_gpu_output,
    TensorData& input1,
    TensorData& input2,
    TensorData& output
) {
    auto start = std::chrono::high_resolution_clock::now();
    for (Numel i = 0; i < N; ++i)
        host_cpu_output[i] = cpu_op(host_input1[i], host_input2[i]);
    auto stop = std::chrono::high_resolution_clock::now();

    double cpu_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    float gpu_ms = time_gpu_once([&] {
        gpu_op(input1, input2, output);
    });

    const Scalar* src =
        static_cast<const Scalar*>(output.storage().raw_data())
        + output.offset();

    check_cuda(
        cudaMemcpy(
            host_gpu_output.data(), src,
            N * sizeof(Scalar), cudaMemcpyDeviceToHost
        ),
        "binary output copy"
    );

    report_elementwise(
        name, cpu_ms, gpu_ms, host_gpu_output, host_cpu_output
    );
}

void run_original_benchmarks() {
    std::cout << "\nOriginal Contiguous CUDA Benchmarks"
              << "\nElements: " << N
              << "\nTensor size: "
              << N * sizeof(Scalar) / (1024.0 * 1024.0)
              << " MB\n\n";

    std::vector<Scalar> host_input1(N);
    std::vector<Scalar> host_input2(N);
    std::vector<Scalar> host_cpu_output(N);
    std::vector<Scalar> host_gpu_output(N);

    for (Numel i = 0; i < N; ++i) {
        host_input1[i] =
            0.1f + static_cast<Scalar>(i % 1000) / 500.0f;
        host_input2[i] =
            0.5f + static_cast<Scalar>(i % 777) / 777.0f;
    }

    TensorData input1(make_gpu_storage(host_input1), Shape{N});
    TensorData input2(make_gpu_storage(host_input2), Shape{N});
    TensorData output = make_output(Shape{N});

    cuda_backend::exp(input1, output);
    sync_after_kernel("CUDA warmup");

    std::cout << "--------------- Unary ------------------\n\n";

    test_unary(
        "exp", cuda_backend::exp,
        [](Scalar x) { return std::exp(x); },
        host_input1, host_cpu_output, host_gpu_output, input1, output
    );
    test_unary(
        "log", cuda_backend::log,
        [](Scalar x) { return std::log(x); },
        host_input1, host_cpu_output, host_gpu_output, input1, output
    );
    test_unary(
        "sqrt", cuda_backend::sqrt,
        [](Scalar x) { return std::sqrt(x); },
        host_input1, host_cpu_output, host_gpu_output, input1, output
    );
    test_unary(
        "tanh", cuda_backend::tanh,
        [](Scalar x) { return std::tanh(x); },
        host_input1, host_cpu_output, host_gpu_output, input1, output
    );

    std::cout << "--------------- Binary -----------------\n\n";

    test_binary(
        "add", cuda_backend::add,
        [](Scalar x, Scalar y) { return x + y; },
        host_input1, host_input2, host_cpu_output, host_gpu_output,
        input1, input2, output
    );
    test_binary(
        "sub", cuda_backend::sub,
        [](Scalar x, Scalar y) { return x - y; },
        host_input1, host_input2, host_cpu_output, host_gpu_output,
        input1, input2, output
    );
    test_binary(
        "mul", cuda_backend::mul,
        [](Scalar x, Scalar y) { return x * y; },
        host_input1, host_input2, host_cpu_output, host_gpu_output,
        input1, input2, output
    );
    test_binary(
        "div", cuda_backend::div,
        [](Scalar x, Scalar y) { return x / y; },
        host_input1, host_input2, host_cpu_output, host_gpu_output,
        input1, input2, output
    );
}

// ============================================================
// Existing reduction benchmarks
// ============================================================

Shape reduced_shape(const Shape& shape, Dim dim) {
    Shape out = shape;
    out.erase(out.begin() + static_cast<std::ptrdiff_t>(dim));
    return out;
}

std::shared_ptr<GPUStorage> make_benchmark_storage(Numel n) {
    std::vector<Scalar> host(n);
    for (Numel i = 0; i < n; ++i) {
        host[i] = static_cast<Scalar>(
            static_cast<int>(i % 2001) - 1000
        ) / 1000.0f;
    }
    return make_gpu_storage(host);
}

float benchmark_reduce(
    ReduceFn op,
    const TensorData& input,
    TensorData& output,
    Dim dim
) {
    for (int i = 0; i < REDUCE_WARMUP_ITERS; ++i)
        op(input, output, dim);

    sync_after_kernel("reduction warmup");

    float total_ms = time_gpu_once([&] {
        for (int i = 0; i < REDUCE_BENCH_ITERS; ++i)
            op(input, output, dim);
    });
    return total_ms / static_cast<float>(REDUCE_BENCH_ITERS);
}

void print_reduce_result(
    const std::string& case_name,
    const std::string& op_name,
    const TensorData& input,
    const TensorData& output,
    Dim dim,
    float ms
) {
    double bytes =
        (static_cast<double>(input.numel())
         + static_cast<double>(output.numel())) * sizeof(Scalar);

    double bandwidth = (bytes / 1e9) / (ms / 1000.0);
    double throughput =
        (static_cast<double>(input.numel()) / 1e9) / (ms / 1000.0);

    std::cout << std::left << std::setw(28) << case_name
              << std::setw(7) << op_name
              << "dim=" << dim << "  "
              << std::right << std::fixed << std::setprecision(4)
              << std::setw(9) << ms << " ms  "
              << std::setprecision(2)
              << std::setw(9) << bandwidth << " GB/s  "
              << std::setw(8) << throughput << " GElem/s\n";
}

void run_reduce_benchmark_case(
    const std::string& name,
    const TensorData& input,
    Dim dim
) {
    TensorData output = make_output(reduced_shape(input.shape(), dim));

    float sum_ms = benchmark_reduce(
        cuda_backend::sum, input, output, dim
    );
    print_reduce_result(name, "sum", input, output, dim, sum_ms);

    float max_ms = benchmark_reduce(
        cuda_backend::max, input, output, dim
    );
    print_reduce_result(name, "max", input, output, dim, max_ms);
}

void print_shape(const Shape& shape) {
    std::cout << '[';
    for (Index i = 0; i < shape.size(); ++i) {
        if (i != 0)
            std::cout << ',';
        std::cout << shape[i];
    }
    std::cout << ']';
}

void run_reduction_benchmarks() {
    std::cout << "\nReduction Benchmarks"
              << "\nWarmup iterations: " << REDUCE_WARMUP_ITERS
              << "\nTimed iterations:  " << REDUCE_BENCH_ITERS
              << "\n\n";

    {
        Shape shape{8192, 4096};
        TensorData input(
            make_benchmark_storage(shape_numel(shape)), shape
        );
        std::cout << "Case 1 input shape: ";
        print_shape(input.shape());
        std::cout << "  contiguous last-dim reduction\n";
        run_reduce_benchmark_case("contiguous last dim", input, 1);
        std::cout << '\n';
    }

    {
        Shape shape{4096, 8192};
        TensorData input(
            make_benchmark_storage(shape_numel(shape)), shape
        );
        std::cout << "Case 2 input shape: ";
        print_shape(input.shape());
        std::cout << "  contiguous first-dim reduction\n";
        run_reduce_benchmark_case("contiguous first dim", input, 0);
        std::cout << '\n';
    }

    {
        Shape shape{4096, 8192};
        TensorData base(
            make_benchmark_storage(shape_numel(shape)), shape
        );
        TensorData view = base.transpose(0, 1);
        std::cout << "Case 3 input shape: ";
        print_shape(view.shape());
        std::cout << "  transpose view, strides=["
                  << view.strides()[0] << ','
                  << view.strides()[1] << "]\n";
        run_reduce_benchmark_case("transpose reduce dim1", view, 1);
        std::cout << '\n';
    }

    {
        Shape shape{1024, 8, 4096};
        TensorData input(
            make_benchmark_storage(shape_numel(shape)), shape
        );
        std::cout << "Case 4 input shape: ";
        print_shape(input.shape());
        std::cout << "  Transformer-ish last-dim reduction\n";
        run_reduce_benchmark_case("3D last dim", input, 2);
        std::cout << '\n';
    }

    {
        Shape shape{262144, 128};
        TensorData input(
            make_benchmark_storage(shape_numel(shape)), shape
        );
        std::cout << "Case 5 input shape: ";
        print_shape(input.shape());
        std::cout << "  small reduction width\n";
        run_reduce_benchmark_case("reduce width 128", input, 1);
        std::cout << '\n';
    }

    std::cout
        << "Reduction benchmark notes:\n"
        << "  * GB/s = logical input+output bandwidth.\n"
        << "  * Host<->device copies are NOT timed.\n"
        << "  * Each number is the average of "
        << REDUCE_BENCH_ITERS << " launches after warmup.\n";
}

} // namespace

int main() {
    try {
        std::cout
            << "========================================\n"
            << "FarfieldLLM CUDA Backend Test\n"
            << "========================================\n";

        check_cuda(cudaSetDevice(0), "cudaSetDevice(0)");

        run_stride_and_reduction_correctness();
        run_matmul_correctness();
        run_batched_matmul_correctness();
        run_original_benchmarks();
        run_reduction_benchmarks();

        std::cout
            << "\n========================================\n"
            << "All CUDA backend tests passed!\n"
            << "========================================\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CUDA backend test FAILED: "
                  << error.what() << '\n';
        return 1;
    }
}
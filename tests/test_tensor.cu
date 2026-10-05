#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "tensor.h"
#include "backend/backend.h"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(error)
        );
    }
}

void sync_cuda() {
    check_cuda(cudaGetLastError(), "CUDA launch");
    check_cuda(cudaDeviceSynchronize(), "CUDA synchronize");
}

Numel numel(const Shape& shape) {
    Numel result = 1;
    for (Index size : shape)
        result *= size;
    return result;
}

TensorData make_data(
    const Shape& shape,
    const std::vector<Scalar>& values
) {
    require(
        numel(shape) == values.size(),
        "Test input shape/value count mismatch"
    );

    auto storage = std::make_shared<GPUStorage>(values.size());

    if (!values.empty()) {
        check_cuda(
            cudaMemcpy(
                storage->raw_data(),
                values.data(),
                values.size() * sizeof(Scalar),
                cudaMemcpyHostToDevice
            ),
            "Upload input"
        );
    }

    return TensorData(storage, shape);
}

TensorData make_zeros(const Shape& shape) {
    return make_data(shape, std::vector<Scalar>(numel(shape), 0.0f));
}

// Read TensorData in logical row-major order.
// Read the storage directly rather than using backend::contiguous,
// so contiguous conversion can be checked independently.
std::vector<Scalar> read_data(const TensorData& data) {
    sync_cuda();

    std::vector<Scalar> result(data.numel());
    if (result.empty())
        return result;

    std::vector<Scalar> storage(data.storage().size());

    check_cuda(
        cudaMemcpy(
            storage.data(),
            data.storage().raw_data(),
            storage.size() * sizeof(Scalar),
            cudaMemcpyDeviceToHost
        ),
        "Download storage"
    );

    for (Index flat = 0; flat < data.numel(); ++flat) {
        Index remaining = flat;
        Index position = data.offset();

        for (std::size_t d = data.ndim(); d > 0; --d) {
            const Index coordinate = remaining % data.shape()[d - 1];
            remaining /= data.shape()[d - 1];
            position += coordinate * data.strides()[d - 1];
        }

        require(position < storage.size(), "View exceeds storage");
        result[flat] = storage[position];
    }

    return result;
}

void check_values(
    const std::string& name,
    const std::vector<Scalar>& actual,
    const std::vector<Scalar>& expected,
    Scalar atol = 1e-5f,
    Scalar rtol = 1e-5f
) {
    require(actual.size() == expected.size(), name + ": value count");

    for (Index i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            throw std::runtime_error(
                name + ": non-finite value at " + std::to_string(i)
            );
        }

        const Scalar error = std::abs(actual[i] - expected[i]);
        const Scalar limit = atol + rtol * std::abs(expected[i]);

        if (error > limit) {
            throw std::runtime_error(
                name + ": index " + std::to_string(i)
                + ", actual=" + std::to_string(actual[i])
                + ", expected=" + std::to_string(expected[i])
            );
        }
    }
}

void check_metadata(
    const std::string& name,
    const Tensor& tensor,
    const Shape& expected_shape
) {
    require(tensor.shape() == expected_shape, name + ": Tensor shape");
    require(
        tensor.ndim() == expected_shape.size(),
        name + ": Tensor rank"
    );
    require(
        tensor.size() == numel(expected_shape),
        name + ": Tensor size"
    );
}

// Tensor result: metadata and successful CUDA execution.
// Backend result: metadata and actual numerical values.
void check_case(
    const std::string& name,
    const Tensor& tensor_result,
    const TensorData& backend_result,
    const Shape& expected_shape,
    const std::vector<Scalar>& expected_values,
    Scalar atol = 1e-5f,
    Scalar rtol = 1e-5f
) {
    check_metadata(name, tensor_result, expected_shape);

    require(
        backend_result.shape() == expected_shape,
        name + ": backend shape"
    );

    check_values(
        name,
        read_data(backend_result),
        expected_values,
        atol,
        rtol
    );

    std::cout << name
              << " PASS [Tensor metadata + backend values]\n";
}

template <typename Exception, typename Fn>
void expect_throw(const std::string& name, Fn fn) {
    try {
        fn();
    } catch (const Exception&) {
        std::cout << name << " PASS\n";
        return;
    } catch (const std::exception& error) {
        throw std::runtime_error(
            name + ": wrong exception: " + error.what()
        );
    }

    throw std::runtime_error(name + ": expected exception");
}

template <typename Fn>
std::vector<Scalar> map_values(
    const std::vector<Scalar>& values,
    Fn operation
) {
    std::vector<Scalar> result;
    result.reserve(values.size());

    for (Scalar value : values)
        result.push_back(operation(value));

    return result;
}

// ============================================================
// Construction and metadata
// ============================================================

void test_metadata() {
    Tensor matrix(make_data({2, 3}, {1, 2, 3, 4, 5, 6}));
    check_metadata("matrix metadata", matrix, {2, 3});

    Tensor scalar(make_data({}, {7}));
    check_metadata("scalar metadata", scalar, {});

    Tensor empty(make_data({0, 3}, {}));
    check_metadata("empty metadata", empty, {0, 3});

    Tensor copied = matrix;
    check_metadata("copy metadata", copied, {2, 3});
    check_metadata("original metadata", matrix, {2, 3});

    std::cout << "construction / metadata PASS\n";
}

// ============================================================
// Binary operators and automatic broadcasting
// ============================================================

void test_binary() {
    const std::vector<Scalar> av{1, 2, 3, 4, 5, 6};
    const std::vector<Scalar> bv{2, 4, 8};

    TensorData ad = make_data({2, 3}, av);
    TensorData bd = make_data({3}, bv);
    const Tensor a(ad);
    const Tensor b(bd);

    std::vector<Scalar> add(6), sub(6), mul(6), div(6);

    for (Index i = 0; i < 6; ++i) {
        const Scalar x = av[i];
        const Scalar y = bv[i % 3];

        add[i] = x + y;
        sub[i] = x - y;
        mul[i] = x * y;
        div[i] = x / y;
    }

    check_case(
        "tensor + tensor, rank broadcasting",
        a + b, backend::add(ad, bd), {2, 3}, add
    );
    check_case(
        "tensor - tensor, rank broadcasting",
        a - b, backend::sub(ad, bd), {2, 3}, sub
    );
    check_case(
        "tensor * tensor, rank broadcasting",
        a * b, backend::mul(ad, bd), {2, 3}, mul
    );
    check_case(
        "tensor / tensor, rank broadcasting",
        a / b, backend::div(ad, bd), {2, 3}, div
    );

    TensorData xd = make_data({2, 1, 3}, av);
    TensorData yd = make_data({1, 4, 1}, {2, 4, 8, 16});
    const Tensor x(xd);
    const Tensor y(yd);

    std::vector<Scalar> expected;
    const std::vector<Scalar> yv{2, 4, 8, 16};

    for (Index batch = 0; batch < 2; ++batch)
        for (Index row = 0; row < 4; ++row)
            for (Index col = 0; col < 3; ++col)
                expected.push_back(av[batch * 3 + col] + yv[row]);

    check_case(
        "binary multi-axis broadcasting",
        x + y, backend::add(xd, yd), {2, 4, 3}, expected
    );

    TensorData view = ad.transpose(0, 1);
    const Tensor tview = a.transpose(0, 1);

    check_case(
        "binary transpose view",
        tview + tview,
        backend::add(view, view),
        {3, 2},
        {2, 8, 4, 10, 6, 12}
    );

    check_values("binary input A unchanged", read_data(ad), av);
    check_values("binary input B unchanged", read_data(bd), bv);
}

// ============================================================
// Scalar operators
// ============================================================

void test_scalar_operators() {
    TensorData base = make_data({2, 3}, {1, 2, 3, 4, 5, 6});
    TensorData data = base.transpose(0, 1);
    const Tensor input(data);

    const std::vector<Scalar> logical{1, 4, 2, 5, 3, 6};

    check_case(
        "tensor + scalar on view",
        input + 2.0f,
        backend::add(data, 2.0f),
        {3, 2},
        map_values(logical, [](Scalar x) { return x + 2.0f; })
    );

    check_case(
        "tensor - scalar on view",
        input - 2.0f,
        backend::sub(data, 2.0f),
        {3, 2},
        map_values(logical, [](Scalar x) { return x - 2.0f; })
    );

    check_case(
        "tensor * scalar on view",
        input * 2.0f,
        backend::mul(data, 2.0f),
        {3, 2},
        map_values(logical, [](Scalar x) { return x * 2.0f; })
    );

    check_case(
        "tensor / scalar on view",
        input / 2.0f,
        backend::div(data, 2.0f),
        {3, 2},
        map_values(logical, [](Scalar x) { return x / 2.0f; })
    );

    check_case(
        "scalar + tensor",
        2.0f + input,
        backend::add(data, 2.0f),
        {3, 2},
        map_values(logical, [](Scalar x) { return 2.0f + x; })
    );

    check_case(
        "scalar * tensor",
        2.0f * input,
        backend::mul(data, 2.0f),
        {3, 2},
        map_values(logical, [](Scalar x) { return 2.0f * x; })
    );

    TensorData scalar_data = make_data({}, {3.0f});
    const Tensor scalar(scalar_data);

    check_case(
        "0D scalar arithmetic",
        scalar + 2.0f,
        backend::add(scalar_data, 2.0f),
        {},
        {5}
    );

    check_values(
        "scalar operations input unchanged",
        read_data(base),
        {1, 2, 3, 4, 5, 6}
    );
}

// ============================================================
// Unary operations
// ============================================================

void test_unary() {
    TensorData base = make_data(
        {2, 3}, {0.25f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f}
    );
    TensorData data = base.transpose(0, 1);
    const Tensor input(data);

    const std::vector<Scalar> logical{
        0.25f, 2.0f, 0.5f, 3.0f, 1.0f, 4.0f
    };

    check_case(
        "exp on transpose",
        input.exp(), backend::exp(data), {3, 2},
        map_values(logical, [](Scalar x) { return std::exp(x); })
    );

    check_case(
        "log on transpose",
        input.log(), backend::log(data), {3, 2},
        map_values(logical, [](Scalar x) { return std::log(x); })
    );

    check_case(
        "sqrt on transpose",
        input.sqrt(), backend::sqrt(data), {3, 2},
        map_values(logical, [](Scalar x) { return std::sqrt(x); })
    );

    check_case(
        "tanh on transpose",
        input.tanh(), backend::tanh(data), {3, 2},
        map_values(logical, [](Scalar x) { return std::tanh(x); })
    );

    TensorData offset_data = base.narrow(0, 1, 1);
    const Tensor offset_tensor(offset_data);

    check_case(
        "unary with nonzero offset",
        offset_tensor.log(),
        backend::log(offset_data),
        {1, 3},
        map_values(
            std::vector<Scalar>{2, 3, 4},
            [](Scalar x) { return std::log(x); }
        )
    );
}

// ============================================================
// Reductions
// ============================================================

void test_reductions() {
    std::vector<Scalar> values(24);
    for (Index i = 0; i < values.size(); ++i)
        values[i] = static_cast<Scalar>(static_cast<int>(i) - 11);

    TensorData data = make_data({2, 3, 4}, values);
    const Tensor input(data);

    for (Dim dim = 0; dim < 3; ++dim) {
        Shape output_shape{2, 3, 4};
        output_shape.erase(output_shape.begin() + dim);

        std::vector<Scalar> sums(numel(output_shape), 0.0f);
        std::vector<Scalar> maxima(numel(output_shape));

        for (Index batch = 0; batch < 2; ++batch) {
            for (Index row = 0; row < 3; ++row) {
                for (Index col = 0; col < 4; ++col) {
                    const Scalar value =
                        values[batch * 12 + row * 4 + col];

                    Index out_index;
                    bool first;

                    if (dim == 0) {
                        out_index = row * 4 + col;
                        first = batch == 0;
                    } else if (dim == 1) {
                        out_index = batch * 4 + col;
                        first = row == 0;
                    } else {
                        out_index = batch * 3 + row;
                        first = col == 0;
                    }

                    sums[out_index] += value;
                    if (first)
                        maxima[out_index] = value;
                    else
                        maxima[out_index] =
                            std::max(maxima[out_index], value);
                }
            }
        }

        check_case(
            "sum dim " + std::to_string(dim),
            input.sum(dim),
            backend::sum(data, dim),
            output_shape,
            sums
        );

        check_case(
            "max dim " + std::to_string(dim),
            input.max(dim),
            backend::max(data, dim),
            output_shape,
            maxima
        );
    }

    TensorData small = make_data({2, 3}, {1, 2, 3, 4, 5, 6});
    TensorData view = small.transpose(0, 1);
    const Tensor transposed(view);

    check_case(
        "sum transpose view",
        transposed.sum(1),
        backend::sum(view, 1),
        {3},
        {5, 7, 9}
    );

    check_case(
        "max transpose view",
        transposed.max(1),
        backend::max(view, 1),
        {3},
        {4, 5, 6}
    );

    TensorData vector = make_data({4}, {1.5f, 2.5f, -1.0f, 4.0f});
    const Tensor tensor_vector(vector);

    check_case(
        "sum to 0D",
        tensor_vector.sum(0),
        backend::sum(vector, 0),
        {},
        {7}
    );

    check_case(
        "max to 0D",
        tensor_vector.max(0),
        backend::max(vector, 0),
        {},
        {4}
    );

    check_case(
        "chained sum to scalar",
        input.sum(2).sum(1).sum(0),
        backend::sum(backend::sum(backend::sum(data, 2), 1), 0),
        {},
        {12}
    );
}

// ============================================================
// All view methods and contiguous
// ============================================================

void test_views() {
    TensorData data = make_data({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor input(data);

    check_case(
        "reshape",
        input.reshape({3, 2}),
        backend::reshape(data, {3, 2}),
        {3, 2},
        {1, 2, 3, 4, 5, 6}
    );

    check_case(
        "transpose",
        input.transpose(0, 1),
        backend::transpose(data, 0, 1),
        {3, 2},
        {1, 4, 2, 5, 3, 6}
    );

    check_case(
        "permute",
        input.permute({1, 0}),
        backend::permute(data, {1, 0}),
        {3, 2},
        {1, 4, 2, 5, 3, 6}
    );

    check_case(
        "unsqueeze",
        input.unsqueeze(1),
        backend::unsqueeze(data, 1),
        {2, 1, 3},
        {1, 2, 3, 4, 5, 6}
    );

    check_case(
        "squeeze",
        input.unsqueeze(1).squeeze(1),
        backend::squeeze(backend::unsqueeze(data, 1), 1),
        {2, 3},
        {1, 2, 3, 4, 5, 6}
    );

    TensorData row = make_data({1, 3}, {2, 4, 8});
    const Tensor tensor_row(row);

    check_case(
        "broadcast_to",
        tensor_row.broadcast_to({2, 3}),
        backend::broadcast_to(row, {2, 3}),
        {2, 3},
        {2, 4, 8, 2, 4, 8}
    );

    check_case(
        "narrow",
        input.narrow(0, 1, 1),
        backend::narrow(data, 0, 1, 1),
        {1, 3},
        {4, 5, 6}
    );

    check_case(
        "slice with step",
        input.slice(1, 0, 3, 2),
        backend::slice(data, 1, 0, 3, 2),
        {2, 2},
        {1, 3, 4, 6}
    );

    check_case(
        "select",
        input.select(1, 1),
        backend::select(data, 1, 1),
        {2},
        {2, 5}
    );

    check_case(
        "flatten",
        input.flatten(),
        backend::flatten(data),
        {6},
        {1, 2, 3, 4, 5, 6}
    );

    check_case(
        "chained views",
        input.slice(0, 1, 2).select(1, 1).unsqueeze(0),
        backend::unsqueeze(
            backend::select(backend::slice(data, 0, 1, 2), 1, 1),
            0
        ),
        {1, 1},
        {5}
    );

    TensorData already_contiguous = backend::contiguous(data);

    require(
        &already_contiguous.storage() == &data.storage(),
        "contiguous input should preserve storage"
    );

    check_case(
        "contiguous fast path",
        input.contiguous(),
        already_contiguous,
        {2, 3},
        {1, 2, 3, 4, 5, 6}
    );

    TensorData transposed = backend::transpose(data, 0, 1);
    TensorData copied = backend::contiguous(transposed);

    require(copied.is_contiguous(), "contiguous conversion failed");
    require(
        &copied.storage() != &transposed.storage(),
        "non-contiguous conversion should allocate storage"
    );

    check_case(
        "transpose -> contiguous -> reshape",
        input.transpose(0, 1).contiguous().reshape({2, 3}),
        backend::reshape(copied, {2, 3}),
        {2, 3},
        {1, 4, 2, 5, 3, 6}
    );

    check_case(
        "broadcast -> contiguous",
        tensor_row.broadcast_to({2, 3}).contiguous(),
        backend::contiguous(backend::broadcast_to(row, {2, 3})),
        {2, 3},
        {2, 4, 8, 2, 4, 8}
    );

    check_metadata("source shape unchanged", input, {2, 3});
    check_values(
        "source values unchanged after views",
        read_data(data),
        {1, 2, 3, 4, 5, 6}
    );
}

// ============================================================
// Matrix multiplication
// ============================================================

std::vector<Scalar> cpu_matmul(
    const std::vector<Scalar>& a,
    const std::vector<Scalar>& b,
    Index batches,
    Index m,
    Index k,
    Index n
) {
    std::vector<Scalar> result(batches * m * n);

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

                result[(batch * m + row) * n + col] =
                    static_cast<Scalar>(value);
            }
        }
    }

    return result;
}

void test_matmul_batch(const Shape& batch_shape) {
    constexpr Index m = 4;
    constexpr Index k = 8;
    constexpr Index n = 4;

    const Index batches = numel(batch_shape);

    Shape a_shape = batch_shape;
    a_shape.push_back(m);
    a_shape.push_back(k);

    Shape b_shape = batch_shape;
    b_shape.push_back(k);
    b_shape.push_back(n);

    Shape output_shape = batch_shape;
    output_shape.push_back(m);
    output_shape.push_back(n);

    std::vector<Scalar> av(batches * m * k);
    std::vector<Scalar> bv(batches * k * n);

    for (Index i = 0; i < av.size(); ++i) {
        av[i] = static_cast<Scalar>(
            static_cast<int>((i * 7) % 31) - 15
        ) / 8.0f;
    }

    for (Index i = 0; i < bv.size(); ++i) {
        bv[i] = static_cast<Scalar>(
            static_cast<int>((i * 11) % 29) - 14
        ) / 8.0f;
    }

    TensorData ad = make_data(a_shape, av);
    TensorData bd = make_data(b_shape, bv);
    const Tensor a(ad);
    const Tensor b(bd);

    check_case(
        std::to_string(a_shape.size()) + "D matmul, batches="
            + std::to_string(batches),
        a.matmul(b),
        backend::matmul(ad, bd),
        output_shape,
        cpu_matmul(av, bv, batches, m, k, n),
        1e-4f,
        1e-4f
    );
}

void test_matmul() {
    test_matmul_batch({});
    test_matmul_batch({3});
    test_matmul_batch({2, 3});
    test_matmul_batch({2, 1, 3});
    test_matmul_batch({1, 1, 1});

    // Both inputs are non-contiguous. backend::matmul should
    // materialize contiguous copies before calling CUDA matmul.
    TensorData a_base = make_data(
        {4, 3},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    );
    TensorData b_base = make_data(
        {2, 4},
        {1, 2, 3, 4, -1, 0, 1, 2}
    );

    TensorData ad = a_base.transpose(0, 1);
    TensorData bd = b_base.transpose(0, 1);

    const Tensor a(ad);
    const Tensor b(bd);

    const std::vector<Scalar> av{
        1, 4, 7, 10,
        2, 5, 8, 11,
        3, 6, 9, 12
    };
    const std::vector<Scalar> bv{
        1, -1,
        2, 0,
        3, 1,
        4, 2
    };

    check_case(
        "matmul automatically materializes transpose inputs",
        a.matmul(b),
        backend::matmul(ad, bd),
        {3, 2},
        cpu_matmul(av, bv, 1, 3, 4, 2),
        1e-4f,
        1e-4f
    );

    TensorData padded = make_data(
        {4, 4},
        {
            -9, -9, -9, -9,
            1, 2, 3, 4,
            5, 6, 7, 8,
            -9, -9, -9, -9
        }
    );
    TensorData offset = padded.narrow(0, 1, 2);
    TensorData identity = make_data(
        {4, 4},
        {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        }
    );

    const Tensor offset_tensor(offset);
    const Tensor identity_tensor(identity);

    check_case(
        "matmul contiguous input with nonzero offset",
        offset_tensor.matmul(identity_tensor),
        backend::matmul(offset, identity),
        {2, 4},
        {1, 2, 3, 4, 5, 6, 7, 8},
        1e-4f,
        1e-4f
    );
}

// ============================================================
// Error behavior through the Tensor API
// ============================================================

void test_errors() {
    const Tensor a(make_zeros({2, 3}));
    const Tensor incompatible(make_zeros({2, 4}));

    expect_throw<std::invalid_argument>(
        "reject incompatible binary shapes",
        [&] { (void)(a + incompatible); }
    );

    expect_throw<std::out_of_range>(
        "reject sum dimension",
        [&] { (void)a.sum(2); }
    );

    expect_throw<std::out_of_range>(
        "reject max dimension",
        [&] { (void)a.max(2); }
    );

    expect_throw<std::invalid_argument>(
        "reject reshape size mismatch",
        [&] { (void)a.reshape({7}); }
    );

    expect_throw<std::invalid_argument>(
        "reject non-contiguous reshape",
        [&] { (void)a.transpose(0, 1).reshape({6}); }
    );

    expect_throw<std::invalid_argument>(
        "reject non-contiguous flatten",
        [&] { (void)a.transpose(0, 1).flatten(); }
    );

    expect_throw<std::out_of_range>(
        "reject transpose dimension",
        [&] { (void)a.transpose(0, 2); }
    );

    expect_throw<std::invalid_argument>(
        "reject duplicate permute dimension",
        [&] { (void)a.permute({0, 0}); }
    );

    expect_throw<std::out_of_range>(
        "reject unsqueeze dimension",
        [&] { (void)a.unsqueeze(3); }
    );

    expect_throw<std::invalid_argument>(
        "reject squeeze of non-singleton dimension",
        [&] { (void)a.squeeze(0); }
    );

    expect_throw<std::invalid_argument>(
        "reject broadcast target",
        [&] { (void)a.broadcast_to({2, 4}); }
    );

    expect_throw<std::out_of_range>(
        "reject narrow range",
        [&] { (void)a.narrow(0, 1, 2); }
    );

    expect_throw<std::invalid_argument>(
        "reject slice step zero",
        [&] { (void)a.slice(1, 0, 3, 0); }
    );

    expect_throw<std::out_of_range>(
        "reject select index",
        [&] { (void)a.select(1, 3); }
    );

    const Tensor vector(make_zeros({3}));
    const Tensor matrix(make_zeros({3, 4}));

    expect_throw<std::invalid_argument>(
        "reject vector matmul",
        [&] { (void)vector.matmul(matrix); }
    );

    const Tensor rank3(make_zeros({2, 3, 4}));

    expect_throw<std::invalid_argument>(
        "reject matmul rank mismatch",
        [&] { (void)rank3.matmul(matrix); }
    );

    const Tensor wrong_inner(make_zeros({4, 2}));

    expect_throw<std::invalid_argument>(
        "reject matmul inner dimension mismatch",
        [&] { (void)a.matmul(wrong_inner); }
    );

    const Tensor batch_a(make_zeros({2, 3, 4, 8}));
    const Tensor batch_b(make_zeros({3, 2, 8, 4}));

    expect_throw<std::invalid_argument>(
        "reject different batch shapes with equal batch count",
        [&] { (void)batch_a.matmul(batch_b); }
    );
}

} // namespace

int main() {
    try {
        check_cuda(cudaSetDevice(0), "Select GPU");

        std::cout << "Tensor API / backend integration tests\n"
                  << "Tensor: metadata and exceptions\n"
                  << "Backend: numerical results\n\n";

        test_metadata();
        test_binary();
        test_scalar_operators();
        test_unary();
        test_reductions();
        test_views();
        test_matmul();
        test_errors();

        sync_cuda();

        std::cout << "\nAll Tensor API / backend integration tests passed!\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "\nFAIL: " << error.what() << '\n';
        return 1;
    }
}
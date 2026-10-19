#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ad_value.h"
#include "engine.h"

class TensorTestAccess{
public:
    static const TensorData& data(const Tensor& tensor){
        return tensor.data_;
    }
};

namespace {

void require(bool condition, const std::string& message){
    if(!condition)
        throw std::runtime_error(message);
}

void check_cuda(cudaError_t error, const char* operation){
    if(error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

void sync_cuda(){
    check_cuda(cudaGetLastError(), "CUDA launch");
    check_cuda(cudaDeviceSynchronize(), "CUDA synchronize");
}

Numel numel(const Shape& shape){
    Numel result = 1;
    for(Index size : shape)
        result *= size;
    return result;
}

std::vector<Scalar> sequence(Numel size){
    std::vector<Scalar> values(size);
    for(Index i = 0; i < size; ++i)
        values[i] = static_cast<Scalar>(i + 1);
    return values;
}

Tensor make_tensor(const Shape& shape, const std::vector<Scalar>& values){
    require(numel(shape) == values.size(), "Test input shape/value count mismatch");
    auto storage = std::make_shared<GPUStorage>(values.size());
    if(!values.empty()){
        check_cuda(cudaMemcpy(storage->raw_data(), values.data(), values.size() * sizeof(Scalar),
            cudaMemcpyHostToDevice), "Upload input");
    }
    return Tensor(TensorData(std::move(storage), shape));
}

std::vector<Scalar> read_tensor(const Tensor& tensor){
    sync_cuda();
    const auto& data = TensorTestAccess::data(tensor);
    std::vector<Scalar> result(data.numel());
    if(result.empty())
        return result;

    std::vector<Scalar> storage(data.storage().size());
    check_cuda(cudaMemcpy(storage.data(), data.storage().raw_data(), storage.size() * sizeof(Scalar),
        cudaMemcpyDeviceToHost), "Download storage");

    for(Index flat = 0; flat < data.numel(); ++flat){
        Index remaining = flat;
        Index position = data.offset();
        for(std::size_t dim = data.ndim(); dim > 0; --dim){
            position += (remaining % data.shape()[dim - 1]) * data.strides()[dim - 1];
            remaining /= data.shape()[dim - 1];
        }
        require(position < storage.size(), "Tensor view exceeds storage");
        result[flat] = storage[position];
    }
    return result;
}

void check_values(const std::string& name, const std::vector<Scalar>& actual,
    const std::vector<Scalar>& expected, Scalar atol = 1e-5f, Scalar rtol = 1e-5f){
    require(actual.size() == expected.size(), name + ": value count mismatch");
    for(Index i = 0; i < actual.size(); ++i){
        if(std::isnan(expected[i]) && std::isnan(actual[i]))
            continue;
        if(std::isinf(expected[i]) && actual[i] == expected[i])
            continue;
        if(!std::isfinite(actual[i]) || !std::isfinite(expected[i])
            || std::abs(actual[i] - expected[i]) > atol + rtol * std::abs(expected[i])){
            throw std::runtime_error(name + ": index " + std::to_string(i)
                + ", actual=" + std::to_string(actual[i])
                + ", expected=" + std::to_string(expected[i]));
        }
    }
    std::cout << name << " PASS\n";
}

void check_tensor(const std::string& name, const Tensor& tensor, const Shape& shape,
    const std::vector<Scalar>& expected){
    require(tensor.shape() == shape, name + ": shape mismatch");
    require(tensor.ndim() == shape.size(), name + ": rank mismatch");
    require(tensor.size() == numel(shape), name + ": size mismatch");
    check_values(name, read_tensor(tensor), expected);
}

void check_grad(const std::string& name, const ADvalue& value, const std::vector<Scalar>& expected){
    require(value.grad().has_value(), name + ": gradient is missing");
    check_tensor(name, *value.grad(), value.shape(), expected);
}

template <typename Exception, typename Fn>
void expect_throw(const std::string& name, Fn operation){
    try{
        operation();
    }
    catch(const Exception&){
        std::cout << name << " PASS\n";
        return;
    }
    catch(const std::exception& error){
        throw std::runtime_error(name + ": wrong exception: " + error.what());
    }
    throw std::runtime_error(name + ": expected exception");
}

ADvalue sum_all(ADvalue value){
    while(value.ndim() != 0)
        value = value.sum(0);
    return value;
}

ADvalue weighted_loss(const ADvalue& output){
    ADvalue weights(make_tensor(output.shape(), sequence(output.size())));
    return sum_all(output * weights);
}

template <typename Fn>
void check_finite_difference(const std::string& name, const Shape& shape,
    const std::vector<Scalar>& values, Fn operation, const Tensor& grad){
    const Scalar epsilon = 1e-2f;
    std::vector<Scalar> expected(values.size());
    for(Index i = 0; i < values.size(); ++i){
        auto plus = values;
        auto minus = values;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        auto positive = read_tensor(operation(ADvalue(make_tensor(shape, plus))).value());
        auto negative = read_tensor(operation(ADvalue(make_tensor(shape, minus))).value());
        require(positive.size() == negative.size(), name + ": output size changed");

        double difference = 0;
        for(Index j = 0; j < positive.size(); ++j)
            difference += static_cast<double>(j + 1)
                * (static_cast<double>(positive[j]) - negative[j]);
        expected[i] = static_cast<Scalar>(difference / (plus[i] - minus[i]));
    }
    check_values(name + " finite difference", read_tensor(grad), expected, 3e-3f, 3e-3f);
}

void test_binary(){
    struct Case{
        const char* name;
        std::function<ADvalue(const ADvalue&, const ADvalue&)> operation;
        std::function<Scalar(Scalar, Scalar)> forward;
        std::function<Scalar(Scalar, Scalar)> derivative_a;
        std::function<Scalar(Scalar, Scalar)> derivative_b;
    };
    const std::vector<Case> cases{
        {"add", [](const ADvalue& a, const ADvalue& b){ return a + b; },
            [](Scalar a, Scalar b){ return a + b; },
            [](Scalar, Scalar){ return 1.0f; }, [](Scalar, Scalar){ return 1.0f; }},
        {"sub", [](const ADvalue& a, const ADvalue& b){ return a - b; },
            [](Scalar a, Scalar b){ return a - b; },
            [](Scalar, Scalar){ return 1.0f; }, [](Scalar, Scalar){ return -1.0f; }},
        {"mul", [](const ADvalue& a, const ADvalue& b){ return a * b; },
            [](Scalar a, Scalar b){ return a * b; },
            [](Scalar, Scalar b){ return b; }, [](Scalar a, Scalar){ return a; }},
        {"div", [](const ADvalue& a, const ADvalue& b){ return a / b; },
            [](Scalar a, Scalar b){ return a / b; },
            [](Scalar, Scalar b){ return 1.0f / b; },
            [](Scalar a, Scalar b){ return -a / (b * b); }}
    };
    const std::vector<Scalar> a_values{1.5f, 2, 3.5f, 4, 5, 6};
    const std::vector<Scalar> b_values{2, 3, 4};

    for(const auto& test : cases){
        const std::string name = std::string("broadcast ") + test.name;
        ADvalue a(make_tensor({2, 3}, a_values), true);
        ADvalue b(make_tensor({3}, b_values), true);
        auto output = test.operation(a, b);
        std::vector<Scalar> forward(6), grad_a(6), grad_b(3, 0);
        for(Index i = 0; i < 6; ++i){
            const Index j = i % 3;
            const Scalar weight = static_cast<Scalar>(i + 1);
            forward[i] = test.forward(a_values[i], b_values[j]);
            grad_a[i] = weight * test.derivative_a(a_values[i], b_values[j]);
            grad_b[j] += weight * test.derivative_b(a_values[i], b_values[j]);
        }
        check_tensor(name + " forward", output.value(), {2, 3}, forward);
        auto loss = weighted_loss(output);
        Engine::backward(loss);
        check_grad(name + " lhs grad", a, grad_a);
        check_grad(name + " rhs grad", b, grad_b);
        check_grad(name + " output grad", output, sequence(6));
        check_grad(name + " loss seed", loss, {1});

        ADvalue constant_a(make_tensor({2, 3}, a_values));
        ADvalue constant_b(make_tensor({3}, b_values));
        check_finite_difference(name + " lhs", {2, 3}, a_values,
            [&](const ADvalue& value){ return test.operation(value, constant_b); }, *a.grad());
        check_finite_difference(name + " rhs", {3}, b_values,
            [&](const ADvalue& value){ return test.operation(constant_a, value); }, *b.grad());
        check_tensor(name + " lhs unchanged", a.value(), {2, 3}, a_values);
        check_tensor(name + " rhs unchanged", b.value(), {3}, b_values);

        for(bool train_left : {false, true}){
            ADvalue lhs(make_tensor({2, 3}, a_values), train_left);
            ADvalue rhs(make_tensor({3}, b_values), !train_left);
            Engine::backward(weighted_loss(test.operation(lhs, rhs)));
            if(train_left){
                check_grad(name + " trainable lhs only", lhs, grad_a);
                require(!rhs.grad(), name + ": constant rhs received a gradient");
            }
            else{
                check_grad(name + " trainable rhs only", rhs, grad_b);
                require(!lhs.grad(), name + ": constant lhs received a gradient");
            }
        }
    }

    ADvalue a(make_tensor({2, 1, 3}, sequence(6)), true);
    ADvalue b(make_tensor({1, 4, 1}, sequence(4)), true);
    Engine::backward(weighted_loss(a + b));
    check_grad("multi axis broadcast lhs grad", a, {22, 26, 30, 70, 74, 78});
    check_grad("multi axis broadcast rhs grad", b, {48, 66, 84, 102});
}

void test_unary_and_scalar(){
    struct Case{
        const char* name;
        std::function<ADvalue(const ADvalue&)> operation;
        std::function<Scalar(Scalar)> forward;
        std::function<Scalar(Scalar)> derivative;
    };
    const std::vector<Case> cases{
        {"scalar add", [](const ADvalue& x){ return x + 2.0f; },
            [](Scalar x){ return x + 2; }, [](Scalar){ return 1.0f; }},
        {"scalar sub", [](const ADvalue& x){ return x - 2.0f; },
            [](Scalar x){ return x - 2; }, [](Scalar){ return 1.0f; }},
        {"scalar mul", [](const ADvalue& x){ return x * 2.0f; },
            [](Scalar x){ return x * 2; }, [](Scalar){ return 2.0f; }},
        {"scalar div", [](const ADvalue& x){ return x / 2.0f; },
            [](Scalar x){ return x / 2; }, [](Scalar){ return 0.5f; }},
        {"scalar lhs add", [](const ADvalue& x){ return 2.0f + x; },
            [](Scalar x){ return 2 + x; }, [](Scalar){ return 1.0f; }},
        {"scalar lhs mul", [](const ADvalue& x){ return 2.0f * x; },
            [](Scalar x){ return 2 * x; }, [](Scalar){ return 2.0f; }},
        {"exp", [](const ADvalue& x){ return x.exp(); },
            [](Scalar x){ return std::exp(x); }, [](Scalar x){ return std::exp(x); }},
        {"log", [](const ADvalue& x){ return x.log(); },
            [](Scalar x){ return std::log(x); }, [](Scalar x){ return 1.0f / x; }},
        {"sqrt", [](const ADvalue& x){ return x.sqrt(); },
            [](Scalar x){ return std::sqrt(x); }, [](Scalar x){ return 0.5f / std::sqrt(x); }},
        {"tanh", [](const ADvalue& x){ return x.tanh(); },
            [](Scalar x){ return std::tanh(x); },
            [](Scalar x){ const Scalar y = std::tanh(x); return 1 - y * y; }}
    };
    const std::vector<Scalar> values{0.5f, 0.75f, 1.25f, 1.5f};
    for(const auto& test : cases){
        ADvalue input(make_tensor({2, 2}, values), true);
        auto output = test.operation(input);
        std::vector<Scalar> forward(4), grad(4);
        for(Index i = 0; i < 4; ++i){
            forward[i] = test.forward(values[i]);
            grad[i] = static_cast<Scalar>(i + 1) * test.derivative(values[i]);
        }
        check_tensor(std::string(test.name) + " forward", output.value(), {2, 2}, forward);
        Engine::backward(weighted_loss(output));
        check_grad(std::string(test.name) + " grad", input, grad);
        check_finite_difference(test.name, {2, 2}, values, test.operation, *input.grad());
    }
}

template <typename Fn>
void check_mapping(const std::string& name, const Shape& input_shape, Fn operation,
    const Shape& output_shape, const std::vector<Index>& mapping){
    const auto values = sequence(numel(input_shape));
    require(mapping.size() == numel(output_shape), name + ": test mapping size");
    std::vector<Scalar> forward(mapping.size()), grad(values.size(), 0);
    for(Index i = 0; i < mapping.size(); ++i){
        require(mapping[i] < values.size(), name + ": test mapping index");
        forward[i] = values[mapping[i]];
        grad[mapping[i]] += static_cast<Scalar>(i + 1);
    }
    ADvalue input(make_tensor(input_shape, values), true);
    auto output = operation(input);
    check_tensor(name + " forward", output.value(), output_shape, forward);
    Engine::backward(weighted_loss(output));
    check_grad(name + " grad", input, grad);
    check_tensor(name + " input unchanged", input.value(), input_shape, values);
}

void test_views(){
    check_mapping("contiguous", {2, 3}, [](const ADvalue& x){ return x.contiguous(); },
        {2, 3}, {0, 1, 2, 3, 4, 5});
    check_mapping("transpose contiguous", {2, 3},
        [](const ADvalue& x){ return x.transpose(0, 1).contiguous(); },
        {3, 2}, {0, 3, 1, 4, 2, 5});
    check_mapping("reshape", {2, 3}, [](const ADvalue& x){ return x.reshape({3, 2}); },
        {3, 2}, {0, 1, 2, 3, 4, 5});
    check_mapping("transpose", {2, 3}, [](const ADvalue& x){ return x.transpose(0, 1); },
        {3, 2}, {0, 3, 1, 4, 2, 5});
    check_mapping("permute", {2, 2, 3}, [](const ADvalue& x){ return x.permute({2, 0, 1}); },
        {3, 2, 2}, {0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11});
    check_mapping("unsqueeze", {2, 3}, [](const ADvalue& x){ return x.unsqueeze(1); },
        {2, 1, 3}, {0, 1, 2, 3, 4, 5});
    check_mapping("squeeze", {2, 1, 3}, [](const ADvalue& x){ return x.squeeze(1); },
        {2, 3}, {0, 1, 2, 3, 4, 5});
    check_mapping("broadcast", {1, 3}, [](const ADvalue& x){ return x.broadcast_to({2, 3}); },
        {2, 3}, {0, 1, 2, 0, 1, 2});
    check_mapping("scalar broadcast", {}, [](const ADvalue& x){ return x.broadcast_to({2, 2}); },
        {2, 2}, {0, 0, 0, 0});
    check_mapping("flatten", {2, 3}, [](const ADvalue& x){ return x.flatten(); },
        {6}, {0, 1, 2, 3, 4, 5});
    check_mapping("narrow offset", {2, 3}, [](const ADvalue& x){ return x.narrow(1, 1, 2); },
        {2, 2}, {1, 2, 4, 5});
    check_mapping("slice step", {2, 4}, [](const ADvalue& x){ return x.slice(1, 0, 4, 2); },
        {2, 2}, {0, 2, 4, 6});
    check_mapping("select", {2, 3}, [](const ADvalue& x){ return x.select(1, 1); },
        {2}, {1, 4});
    check_mapping("transpose slice", {2, 3},
        [](const ADvalue& x){ return x.transpose(0, 1).slice(0, 1, 3); },
        {2, 2}, {1, 4, 2, 5});
    check_mapping("empty slice", {2, 3}, [](const ADvalue& x){ return x.slice(1, 1, 1); },
        {2, 0}, {});
}

template <typename Fn>
void check_reduction(const std::string& name, const Shape& shape, const std::vector<Scalar>& values,
    Fn operation, const Shape& output_shape, const std::vector<Scalar>& forward,
    const std::vector<Scalar>& grad){
    ADvalue input(make_tensor(shape, values), true);
    auto output = operation(input);
    check_tensor(name + " forward", output.value(), output_shape, forward);
    Engine::backward(weighted_loss(output));
    check_grad(name + " grad", input, grad);
}

void test_reductions(){
    check_reduction("sum first dim", {2, 3}, {1, 2, 3, 4, 5, 6},
        [](const ADvalue& x){ return x.sum(0); }, {3}, {5, 7, 9}, {1, 2, 3, 1, 2, 3});
    check_reduction("sum last dim", {2, 3}, {1, 2, 3, 4, 5, 6},
        [](const ADvalue& x){ return x.sum(1); }, {2}, {6, 15}, {1, 1, 1, 2, 2, 2});
    check_reduction("sum middle dim", {2, 2, 2}, sequence(8),
        [](const ADvalue& x){ return x.sum(1); }, {2, 2}, {4, 6, 12, 14},
        {1, 2, 1, 2, 3, 4, 3, 4});
    check_reduction("sum empty axis", {0, 3}, {},
        [](const ADvalue& x){ return x.sum(0); }, {3}, {0, 0, 0}, {});
    check_reduction("max first tie", {2, 3}, {4, 4, 2, 1, 5, 5},
        [](const ADvalue& x){ return x.max(1); }, {2}, {4, 5}, {1, 0, 0, 0, 2, 0});
    check_reduction("max middle dim", {2, 2, 2}, sequence(8),
        [](const ADvalue& x){ return x.max(1); }, {2, 2}, {3, 4, 7, 8},
        {0, 0, 1, 2, 0, 0, 3, 4});
    const Scalar nan = std::numeric_limits<Scalar>::quiet_NaN();
    const Scalar inf = std::numeric_limits<Scalar>::infinity();
    check_reduction("max ignores NaN and routes infinity", {2, 3}, {nan, 4, 4, -inf, -inf, -inf},
        [](const ADvalue& x){ return x.max(1); }, {2}, {4, -inf}, {0, 1, 0, 2, 0, 0});
    check_reduction("max all NaN", {3}, {nan, nan, nan},
        [](const ADvalue& x){ return x.max(0); }, {}, {-inf}, {0, 0, 0});
    check_reduction("max empty axis", {0, 3}, {},
        [](const ADvalue& x){ return x.max(0); }, {3}, {-inf, -inf, -inf}, {});
}

void check_matmul(const std::string& name, bool batched){
    const Index batches = batched ? 2 : 1;
    const Index m = 2, k = 3, n = 2;
    const Shape a_shape = batched ? Shape{batches, m, k} : Shape{m, k};
    const Shape b_shape = batched ? Shape{batches, k, n} : Shape{k, n};
    const Shape output_shape = batched ? Shape{batches, m, n} : Shape{m, n};
    const auto a_values = sequence(numel(a_shape));
    const auto b_values = sequence(numel(b_shape));
    std::vector<Scalar> forward(numel(output_shape), 0);
    std::vector<Scalar> grad_a(a_values.size(), 0), grad_b(b_values.size(), 0);
    for(Index batch = 0; batch < batches; ++batch){
        for(Index row = 0; row < m; ++row){
            for(Index col = 0; col < n; ++col){
                const Index out = batch * m * n + row * n + col;
                const Scalar weight = static_cast<Scalar>(out + 1);
                for(Index inner = 0; inner < k; ++inner){
                    const Index ai = batch * m * k + row * k + inner;
                    const Index bi = batch * k * n + inner * n + col;
                    forward[out] += a_values[ai] * b_values[bi];
                    grad_a[ai] += weight * b_values[bi];
                    grad_b[bi] += weight * a_values[ai];
                }
            }
        }
    }

    ADvalue a(make_tensor(a_shape, a_values), true);
    ADvalue b(make_tensor(b_shape, b_values), true);
    auto output = a.matmul(b);
    check_tensor(name + " forward", output.value(), output_shape, forward);
    Engine::backward(weighted_loss(output));
    check_grad(name + " lhs grad", a, grad_a);
    check_grad(name + " rhs grad", b, grad_b);
    check_tensor(name + " lhs unchanged", a.value(), a_shape, a_values);
    check_tensor(name + " rhs unchanged", b.value(), b_shape, b_values);
}

void test_matmul(){
    check_matmul("matmul 2D", false);
    check_matmul("matmul batched", true);

    ADvalue a(make_tensor({3, 2}, sequence(6)), true);
    ADvalue b(make_tensor({3, 2}, sequence(6)), true);
    auto output = a.transpose(0, 1).matmul(b);
    check_tensor("matmul strided forward", output.value(), {2, 2}, {35, 44, 44, 56});
    Engine::backward(sum_all(output));
    check_grad("matmul strided lhs grad", a, {3, 3, 7, 7, 11, 11});
    check_grad("matmul strided rhs grad", b, {3, 3, 7, 7, 11, 11});
}

void test_graph_scheduling(){
    ADvalue x(make_tensor({}, {3}), true);
    require(!x.grad(), "Leaf gradient must initially be empty");
    auto square = x * x;
    Engine::backward(square);
    check_grad("repeated input slots", x, {6});
    check_tensor("backward preserves parameter value", x.value(), {}, {3});

    Engine::zero_grad({x});
    auto u = x * 2.0f;
    auto a = u * 3.0f;
    auto b = u * 4.0f;
    auto loss = a + b;
    Engine::backward(loss);
    check_grad("diamond shared intermediate", u, {7});
    check_grad("diamond leaf", x, {14});
    check_grad("diamond branch a", a, {1});
    check_grad("diamond branch b", b, {1});
    check_grad("diamond loss seed", loss, {1});

    Engine::zero_grad({x});
    auto shared = x * 2.0f;
    auto left = shared + 0.0f;
    auto right = shared + 0.0f;
    auto aliased_loss = left + right;
    Engine::backward(aliased_loss);
    check_grad("shared gradient storage accumulation", shared, {2});
    check_grad("shared gradient storage leaf", x, {4});
    check_grad("left gradient remains unchanged", left, {1});
    check_grad("right gradient remains unchanged", right, {1});
    check_grad("loss seed remains unchanged", aliased_loss, {1});

    ADvalue small(make_tensor({}, {0.5f}), true);
    auto temporary_loss = (small * 2.0f).exp();
    Engine::backward(temporary_loss);
    check_grad("temporary input lifetime", small, {2.0f * std::exp(1.0f)});
}

void test_constants(){
    for(bool train_left : {false, true}){
        ADvalue a(make_tensor({2}, {2, 3}), train_left);
        ADvalue b(make_tensor({2}, {4, 5}), !train_left);
        Engine::backward(sum_all(a * b));
        if(train_left){
            check_grad("constant rhs still used in derivative", a, {4, 5});
            require(!b.grad(), "Constant rhs must not receive a gradient");
        }
        else{
            check_grad("constant lhs still used in derivative", b, {2, 3});
            require(!a.grad(), "Constant lhs must not receive a gradient");
        }
    }

    ADvalue a(make_tensor({}, {2}));
    ADvalue b(make_tensor({}, {3}));
    auto constant_loss = a * b;
    require(!constant_loss.requires_grad(), "Constant expression must not require gradients");
    expect_throw<std::invalid_argument>("reject constant loss", [&]{ Engine::backward(constant_loss); });
    require(!a.grad() && !b.grad() && !constant_loss.grad(), "Constant graph must keep empty gradients");

    ADvalue x(make_tensor({}, {2}), true);
    ADvalue unused(make_tensor({}, {7}), true);
    auto unrelated = unused * 5.0f;
    Engine::backward(x * 3.0f);
    require(!unused.grad() && !unrelated.grad(), "Unreachable values must not receive gradients");
    Engine::backward(unrelated);
    check_grad("unreachable graph remains usable", unused, {5});
}

void test_accumulation_and_zero_grad(){
    ADvalue x(make_tensor({}, {3}), true);
    ADvalue alias = x;
    Engine::backward(x * x);
    check_grad("first graph leaf grad", x, {6});
    Engine::backward(x * 2.0f);
    check_grad("fresh graph accumulates leaf grad", x, {8});
    check_grad("copied handle shares leaf grad", alias, {8});

    ADvalue other(make_tensor({}, {4}), true);
    Engine::backward(other * 3.0f);
    Engine::zero_grad({x, alias, x});
    require(!x.grad() && !alias.grad(), "zero_grad must reset all shared handles");
    check_grad("zero_grad preserves unlisted leaf", other, {3});
    Engine::zero_grad({});
    Engine::zero_grad({x});
    Engine::backward(x * x);
    check_grad("fresh graph after zero_grad", x, {6});

    ADvalue scalar(make_tensor({}, {9}), true);
    Engine::backward(scalar);
    Engine::backward(scalar);
    check_grad("scalar leaf seed accumulates", scalar, {2});
    Engine::zero_grad({scalar});
    Engine::backward(scalar);
    check_grad("scalar leaf seed after reset", scalar, {1});
}

void test_errors(){
    ADvalue x(make_tensor({}, {3}), true);
    auto shared = x * x;
    auto first = shared * 2.0f;
    auto second = shared * 3.0f;
    Engine::backward(first);
    expect_throw<std::logic_error>("reject repeated backward", [&]{ Engine::backward(first); });
    expect_throw<std::logic_error>("reject shared used intermediate", [&]{ Engine::backward(second); });
    check_grad("rejected backward preserves leaf grad", x, {12});
    require(!second.grad(), "Old graph rejection must happen before seeding new loss");

    auto extended = shared + 1.0f;
    expect_throw<std::logic_error>("reject extending a used graph", [&]{ Engine::backward(extended); });
    require(!extended.grad(), "Extended old graph must remain unseeded");
    Engine::zero_grad({x});
    expect_throw<std::logic_error>("zero_grad does not reenable old graph", [&]{ Engine::backward(first); });
    require(!x.grad(), "Old graph rejection must preserve cleared leaf grad");
    Engine::backward(x * 4.0f);
    check_grad("leaf reusable in fresh graph", x, {4});

    ADvalue vector(make_tensor({1}, {2}), true);
    expect_throw<std::invalid_argument>("reject rank one loss", [&]{ Engine::backward(vector); });
    require(!vector.grad(), "Invalid loss must not receive a gradient");

    ADvalue constant(make_tensor({}, {1}));
    auto nonleaf = x + 1.0f;
    expect_throw<std::invalid_argument>("zero_grad rejects nonleaf", [&]{ Engine::zero_grad({x, nonleaf}); });
    check_grad("zero_grad validates complete list", x, {4});
    expect_throw<std::invalid_argument>("zero_grad rejects constant", [&]{ Engine::zero_grad({x, constant}); });
    check_grad("invalid constant preserves earlier parameter grad", x, {4});

    ADvalue moved(make_tensor({}, {5}), true);
    ADvalue owner = std::move(moved);
    expect_throw<std::invalid_argument>("reject moved from loss", [&]{ Engine::backward(moved); });
    expect_throw<std::invalid_argument>("zero_grad rejects moved from parameter",
        [&]{ Engine::zero_grad({x, moved}); });
    check_grad("invalid handle preserves earlier parameter grad", x, {4});
    Engine::backward(owner);
    check_grad("moved to handle remains valid", owner, {1});
}

}

int main(){
    try{
        check_cuda(cudaSetDevice(0), "Select GPU");
        std::cout << "FarfieldLLM Autograd tests\n\n";
        test_binary();
        test_unary_and_scalar();
        test_views();
        test_reductions();
        test_matmul();
        test_graph_scheduling();
        test_constants();
        test_accumulation_and_zero_grad();
        test_errors();
        sync_cuda();
        std::cout << "\nAll Autograd tests passed!\n";
        return 0;
    }
    catch(const std::exception& error){
        std::cerr << "\nFAIL: " << error.what() << '\n';
        return 1;
    }
}

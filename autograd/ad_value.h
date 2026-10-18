#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "tensor.h"

class Node;

class ADimpl{
private:
    Tensor value_;
    std::optional<Tensor> grad_;
    std::shared_ptr<Node> creator_;
    bool requires_grad_;

public:
    explicit ADimpl(Tensor value, bool requires_grad = false, std::shared_ptr<Node> creator = nullptr);
    const Tensor& value() const noexcept;
    const std::optional<Tensor>& grad() const noexcept;
    const std::shared_ptr<Node>& creator() const noexcept;
    bool requires_grad() const noexcept;
};

class ADvalue{
private:
    std::shared_ptr<ADimpl> impl_;
    explicit ADvalue(std::shared_ptr<ADimpl> impl);

public:
    explicit ADvalue(Tensor value, bool requires_grad = false);

    const Tensor& value() const noexcept;
    const std::optional<Tensor>& grad() const noexcept;
    bool requires_grad() const noexcept;

    std::size_t size() const;
    std::size_t ndim() const;
    const Shape& shape() const;

    ADvalue operator+(const ADvalue& other) const;
    ADvalue operator-(const ADvalue& other) const;
    ADvalue operator*(const ADvalue& other) const;
    ADvalue operator/(const ADvalue& other) const;
    ADvalue operator+(Scalar scalar) const;
    ADvalue operator-(Scalar scalar) const;
    ADvalue operator*(Scalar scalar) const;
    ADvalue operator/(Scalar scalar) const;

    ADvalue exp() const;
    ADvalue log() const;
    ADvalue sqrt() const;
    ADvalue tanh() const;
    ADvalue sum(Dim dim) const;
    ADvalue max(Dim dim) const;
    ADvalue matmul(const ADvalue& other) const;

    ADvalue contiguous() const;
    ADvalue reshape(const Shape& new_shape) const;
    ADvalue transpose(Dim dim0, Dim dim1) const;
    ADvalue permute(const Dims& dims) const;
    ADvalue unsqueeze(Dim dim) const;
    ADvalue squeeze(Dim dim) const;
    ADvalue broadcast_to(const Shape& new_shape) const;
    ADvalue narrow(Dim dim, Index start, std::size_t length) const;
    ADvalue slice(Dim dim, Index start, Index end, std::size_t step = 1) const;
    ADvalue select(Dim dim, Index index) const;
    ADvalue flatten() const;
};

ADvalue operator*(Scalar scalar, const ADvalue& other);
ADvalue operator+(Scalar scalar, const ADvalue& other);

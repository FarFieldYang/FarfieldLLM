#pragma once

#include <vector>
#include <cstddef>
#include <memory>

#include "storage.h"

using Shape = std::vector<std::size_t>;
using Strides = std::vector<std::size_t>;
using Offset = std::size_t;
using Numel = std::size_t;
using Index = std::size_t;
using Dim = std::size_t;
using Dims = std::vector<std::size_t>;

class TensorData {
private:
    std::shared_ptr<Storage> storage_;
    Shape shape_;
    Strides strides_;
    Offset offset_ = 0;
    Numel numel_;

    static Strides compute_contiguous_strides(const Shape& shape);
    void compute_numel();

    static Shape broadcast_shape(
        const Shape& a,
        const Shape& b
    );
public:
    TensorData(std::shared_ptr<Storage> storage, Shape shape);
    TensorData(
    std::shared_ptr<Storage> storage,
    Shape shape,
    Strides strides,
    Offset offset = 0
    );
    Device device() const;

    Numel numel() const;
    std::size_t ndim() const;
    const Shape& shape() const;
    const Strides& strides() const;
    Offset offset() const;
    
    Storage& storage();
    const Storage& storage() const;

    bool is_contiguous() const;
    TensorData reshape(const Shape& new_shape) const;
    TensorData transpose(Dim dim0, Dim dim1) const;
    TensorData permute(const Dims& dims) const;
    TensorData unsqueeze(Dim dim) const;
    TensorData squeeze(Dim dim) const;
    TensorData broadcast_to(const Shape& new_shape) const;
    TensorData narrow(Dim dim, Index start, std::size_t length) const;
    TensorData slice(Dim dim, Index start, Index end, std::size_t step = 1) const;
    TensorData select(Dim dim, Index index) const;
    TensorData flatten() const;
};
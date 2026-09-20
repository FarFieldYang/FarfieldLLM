#pragma once

#include <vector>
#include <memory>

#include "storage.h"

using Shape = std::vector<std::size_t>;
using Strides = std::vector<std::size_t>;
using Index = std::vector<std::size_t>;
using Offset = std::size_t;

class TensorData {
public:
    TensorData(std::shared_ptr<Storage> storage, Shape shape);
    TensorData(
    std::shared_ptr<Storage> storage,
    Shape shape,
    Strides strides,
    Offset offset = 0
    );
    Device device() const;

    std::size_t numel() const;
    std::size_t ndim() const;
    const Shape& shape() const;
    const Strides& strides() const;
    Offset offset() const;
    bool is_contiguous() const;
    
    Storage& storage();
    const Storage& storage() const;

private:
    std::shared_ptr<Storage> storage_;
    Shape shape_;
    Strides strides_;
    std::size_t offset_ = 0;
    std::size_t numel_;

    
    void compute_strides();
    void compute_size();

    static Shape broadcast_shape(
        const Shape& a,
        const Shape& b
    );

    static Index broadcast_index(
        const Index& out_index,
        const Index& in_shape
    );
};


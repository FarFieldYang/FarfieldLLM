#pragma once

#include <vector>
#include <cstddef>
#include <memory>

using Scalar = float;
using Scalars = std::vector<Scalar>;

class Storage {
private:
    std::vector<Scalar> data_;

public:
    explicit Storage(Scalars data);

    Scalar& at(std::size_t index);
    const Scalar& at(std::size_t index) const;

    std::size_t size() const;
};

class TensorData {
public:
    TensorData(std::vector<std::size_t> shape, Scalars scalars);

    
    std::size_t size() const;
    std::size_t ndim() const;
    const std::vector<std::size_t>& shape() const;
    
    const Scalar& at(const std::vector<std::size_t>& position) const;
    Scalar& at(const std::vector<std::size_t>& position);

private:
    std::shared_ptr<Storage> storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t offset_ = 0;
    std::size_t numel_;

    TensorData(
    std::shared_ptr<Storage> storage,
    std::vector<std::size_t> shape,
    std::vector<std::size_t> strides,
    std::size_t offset = 0
    );
    
    void compute_strides();
    void compute_size();

    static std::vector<std::size_t> broadcast_shape(
        const std::vector<std::size_t>& a,
        const std::vector<std::size_t>& b
    );

    static std::vector<std::size_t> broadcast_index(
        const std::vector<std::size_t>& out_index,
        const std::vector<std::size_t>& in_shape
    );
};


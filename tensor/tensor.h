#pragma once

#include <vector>
#include <cstddef>

using Scalar = float;
using Scalars = std::vector<Scalar>;
class Tensor {
public:
    Tensor(std::vector<std::size_t> shape, Scalars scalars);
    std::size_t size() const;
    std::size_t ndim() const;
    const std::vector<std::size_t>& shape() const;
    
    const Scalar& at(const std::vector<std::size_t>& position) const;
    Scalar& at(const std::vector<std::size_t>& position);

    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator+(Scalar scalar) const;
    Tensor operator*(Scalar scalar) const;
    Tensor sum() const;

private:
    Scalars storage_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;

    void compute_strides();

    static std::vector<std::size_t> broadcast_shape(
        const std::vector<std::size_t>& a,
        const std::vector<std::size_t>& b
    );

    static std::vector<std::size_t> broadcast_index(
        const std::vector<std::size_t>& out_index,
        const std::vector<std::size_t>& in_shape
    );
};

Tensor operator*(const Scalar& scalar, const Tensor& other);
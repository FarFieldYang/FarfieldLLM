#include <stdexcept>
#include <utility>
#include <algorithm>

#include "tensor.h"

Tensor::Tensor(std::vector<std::size_t> shape, Scalars scalars){
    if (shape.empty()) throw std::invalid_argument("Tensor shape cannot be empty");
    
    std::size_t expected_size = 1;
    for (std::size_t dim : shape) {
        expected_size *= dim;
    }
    if(expected_size != scalars.size()){
        throw std::invalid_argument("Tensor shape does not match data size");
    }

    shape_ = std::move(shape);
    storage_ = std::move(scalars);
    compute_strides();
}

std::size_t Tensor::size() const{
    return storage_.size();
}

std::size_t Tensor::ndim() const{
    return shape_.size();
}

const Scalar& Tensor::at(const std::vector<std::size_t>& position) const{
    if (position.size() != shape_.size()) {
        throw std::invalid_argument("Tensor index dimension mismatch");
    }

    std::size_t offset = 0;
    for(std::size_t i = 0; i < shape_.size(); ++i){
        if (position[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of range");
        }

        offset += position[i] * strides_[i];
    }
    return storage_[offset];
}

Scalar& Tensor::at(const std::vector<std::size_t>& position){
    if (position.size() != shape_.size()) {
        throw std::invalid_argument("Tensor index dimension mismatch");
    }

    std::size_t offset = 0;
    for(std::size_t i = 0; i < shape_.size(); ++i){
        if (position[i] >= shape_[i]) {
            throw std::out_of_range("Tensor index out of range");
        }

        offset += position[i] * strides_[i];
    }
    return storage_[offset];
}

Tensor Tensor::operator+(const Tensor& other) const{

}

Tensor Tensor::operator*(const Tensor& other) const{

}

Tensor Tensor::operator+(Scalar scalar) const{

}

Tensor Tensor::operator*(Scalar scalar) const{

}

Tensor Tensor::sum() const{

}

Tensor operator*(const Scalar& scalar, const Tensor& other){

}

void Tensor::compute_strides(){
    strides_.resize(shape_.size());
    
    int last = static_cast<int>(shape_.size()) - 1;

    for(int i = last; i >= 0; --i){
        if(i == last) strides_[i] = 1;
        else strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
}

std::vector<std::size_t> Tensor::broadcast_shape(
    const std::vector<std::size_t>& a,
    const std::vector<std::size_t>& b
){
    std::size_t na = a.size(), nb = b.size();
    std::vector<std::size_t> result;
    result.resize(std::max(na, nb));
    int imax = result.size() - 1;

    for(int ia = static_cast<int>(na) - 1, ib = static_cast<int>(nb) - 1; ia >= 0 || ib >= 0; --ia, --ib, --imax){

        if(ia < 0 || ib < 0){
            const std::vector<std::size_t>& longer_one = (ia < 0) ? b : a;
            for( ; imax >= 0; --imax){
                result[imax] = longer_one[imax];
            }
            return result;
        }

        else if(a[ia] == b[ib]) result[imax] = a[ia];
        else if(a[ia] == 1 || b[ib] == 1) result[imax] = a[ia] * b[ib];

        else throw std::invalid_argument("Shapes cannot be broadcast");
    }
    return result;
}

std::vector<std::size_t> Tensor::broadcast_index(
    const std::vector<std::size_t>& out_index,
    const std::vector<std::size_t>& in_shape
){
    if (out_index.size() < in_shape.size()) {
        throw std::invalid_argument("Output index has fewer dimensions than input shape");
    }
    
    std::vector<std::size_t> in_index = out_index;
    std::size_t diff = out_index.size() - in_shape.size();
    in_index.erase(in_index.begin(), in_index.begin() + diff);
    for(std::size_t i = 0; i < in_shape.size(); ++i){
        if(in_shape[i] == 1){
            in_index[i] = 0;
        }
    }
    return in_index;
}
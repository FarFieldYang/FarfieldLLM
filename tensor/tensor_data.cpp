#include <stdexcept>
#include <utility>
#include <algorithm>

#include "tensor_data.h"

//public
TensorData::TensorData(std::vector<std::size_t> shape, Scalars scalars){
    shape_ = std::move(shape);
    storage_ = std::make_shared<Storage>(std::move(scalars));
    compute_strides();
    compute_size();

    if(numel_ != storage_->size()){
        throw std::invalid_argument("TensorData shape does not match data size");
    }
}


std::size_t TensorData::size() const{
    return numel_;
}

std::size_t TensorData::ndim() const{
    return shape_.size();
}

const std::vector<std::size_t>& TensorData::shape() const{
    return shape_;
}

const Scalar& TensorData::at(const std::vector<std::size_t>& position) const{
    if (position.size() != shape_.size()) {
        throw std::invalid_argument("TensorData index dimension mismatch");
    }

    std::size_t offset = 0;
    for(std::size_t i = 0; i < shape_.size(); ++i){
        if (position[i] >= shape_[i]) {
            throw std::out_of_range("TensorData index out of range");
        }

        offset += position[i] * strides_[i];
    }
    return storage_->at(offset + offset_);
}

Scalar& TensorData::at(const std::vector<std::size_t>& position){
    if (position.size() != shape_.size()) {
        throw std::invalid_argument("TensorData index dimension mismatch");
    }

    std::size_t offset = 0;
    for(std::size_t i = 0; i < shape_.size(); ++i){
        if (position[i] >= shape_[i]) {
            throw std::out_of_range("TensorData index out of range");
        }

        offset += position[i] * strides_[i];
    }
    return storage_->at(offset + offset_);
}

//private
TensorData::TensorData(
    std::shared_ptr<Storage> storage,
    std::vector<std::size_t> shape,
    std::vector<std::size_t> strides,
    std::size_t offset
):storage_(std::move(storage)), shape_(std::move(shape)), strides_(std::move(strides)), offset_(offset){
    if (!storage_) {
        throw std::invalid_argument("TensorData storage cannot be null");
    }

    if (shape_.size() != strides_.size()) {
        throw std::invalid_argument("TensorData shape and strides dimension mismatch");
    }    
    compute_size();
}

void TensorData::compute_strides(){
    strides_.resize(shape_.size());

    std::size_t stride = 1;

    for(std::size_t i = shape_.size(); i > 0; --i){
        strides_[i - 1] = stride;
        stride *= shape_[i - 1];
    }
}

void TensorData::compute_size(){
    numel_ = 1;
    for(std::size_t dim : shape_){
        numel_ *= dim;
    }
}

std::vector<std::size_t> TensorData::broadcast_shape(
    const std::vector<std::size_t>& a,
    const std::vector<std::size_t>& b
){
    std::size_t na = a.size(), nb = b.size();
    std::vector<std::size_t> result;
    result.resize(std::max(na, nb));
    int imax = static_cast<int>(result.size()) - 1;

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

std::vector<std::size_t> TensorData::broadcast_index(
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
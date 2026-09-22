#include <stdexcept>
#include <utility>
#include <algorithm>

#include "tensor_data.h"

//public
TensorData::TensorData(std::shared_ptr<Storage> storage, Shape shape):shape_(std::move(shape)), storage_(storage){
    if (!storage_)
        throw std::invalid_argument("TensorData storage cannot be null");
    
    strides_ = compute_contiguous_strides(shape_);
    compute_numel();

    if(numel_ != storage_->size())
        throw std::invalid_argument("TensorData shape does not match data size");
}

TensorData::TensorData(
    std::shared_ptr<Storage> storage,
    Shape shape,
    Strides strides,
    Offset offset
):
    storage_(std::move(storage)),
    shape_(std::move(shape)),
    strides_(std::move(strides)),
    offset_(offset){
    if (!storage_)
        throw std::invalid_argument("TensorData storage cannot be null");

    if (shape_.size() != strides_.size())
        throw std::invalid_argument("TensorData shape and strides dimension mismatch");
    compute_numel();
}

Device TensorData::device() const {
    return storage_->device();
}

Numel TensorData::numel() const{
    return numel_;
}

std::size_t TensorData::ndim() const{
    return shape_.size();
}

const Shape& TensorData::shape() const{
    return shape_;
}

const Strides& TensorData::strides() const{
    return strides_;
}

Offset TensorData::offset() const {
    return offset_;
}

Storage& TensorData::storage(){
    return *storage_;
}
const Storage& TensorData::storage() const{
    return *storage_;
}

bool TensorData::is_contiguous() const{
    Strides strides = compute_contiguous_strides(shape_);
    for(std::size_t i = 0; i < shape_.size(); ++i)
        if(strides[i] != strides_[i] && shape_[i] != 1)
            return false;
    return true;
}

TensorData TensorData::reshape(const Shape& new_shape) const{
    Numel new_numel = 1;
    for(Index i : new_shape) new_numel *= i;
    if (new_numel != numel_)
        throw std::invalid_argument("reshape changes number of elements");

    if (!is_contiguous())
        throw std::invalid_argument("reshape requires contiguous TensorData");

    return TensorData(
        storage_,
        new_shape,
        compute_contiguous_strides(new_shape),
        offset_
    );    
} 

TensorData TensorData::transpose(Dim dim0, Dim dim1) const{
    if (dim0 >= shape_.size() || dim1 >= shape_.size())
        throw std::out_of_range("transpose dimension out of range");

    Dims dims(shape_.size());
    for(std::size_t i = 0; i < shape_.size(); ++i) dims[i] = i;
    std::swap(dims[dim0], dims[dim1]);
    return permute(dims);
}

TensorData TensorData::permute(const Dims& dims) const{
    if(dims.size() != shape_.size())
        throw std::invalid_argument("permute dimension mismatch");

    Shape shape(dims.size());
    Strides strides(dims.size());
    Dim i = 0;
    std::vector<bool> used(shape_.size(), false);

    for(Dim j : dims){
        if (j >= shape_.size())
            throw std::out_of_range("permute dimension out of range");

        if (used[j])
            throw std::invalid_argument("permute dimensions must be unique");
        
        used[j] = true;
        shape[i] = shape_[j];
        strides[i] = strides_[j];
        ++i; 
    }
    return TensorData(
        storage_,
        shape,
        strides,
        offset_
    );
}

TensorData TensorData::unsqueeze(Dim dim) const{
    if(dim > shape_.size())
        throw std::out_of_range("unsqueeze dimension out of range");

    Shape shape = shape_;
    Strides strides = strides_;
    
    std::size_t new_stride;
    if(dim == shape_.size()) new_stride = 1;
    else new_stride = strides_[dim] * (shape_[dim] == 0 ? 1 : shape_[dim]);
    shape.insert(shape.begin() + dim, 1);
    strides.insert(strides.begin() + dim, new_stride);
    return TensorData(
        storage_,
        shape,
        strides,
        offset_
    );
}

TensorData TensorData::squeeze(Dim dim) const{

}
TensorData TensorData::broadcast_to(const Shape& new_shape) const{

}
TensorData TensorData::narrow(Dim dim, Index start, std::size_t length) const{

}
TensorData TensorData::slice(Dim dim, Index start, Index end, std::size_t step) const{

}
TensorData TensorData::select(Dim dim, Index index) const{

}
TensorData TensorData::flatten() const{

}
//private

Strides TensorData::compute_contiguous_strides(const Shape& shape){
    Strides strides(shape.size());

    std::size_t stride = 1;

    for(std::size_t i = shape.size(); i > 0; --i){
        strides[i - 1] = stride;
        if(shape[i - 1] == 0) stride *= 1;
        else stride *= shape[i - 1];
    }
    return strides;
}

void TensorData::compute_numel(){
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


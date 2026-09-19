#include "storage.h"
//CPUStorage
CPUStorage::CPUStorage(Scalars data): data_(std::move(data)){}

Scalar& CPUStorage::at(std::size_t index){
    return data_.at(index);
}
const Scalar& CPUStorage::at(std::size_t index) const{
    return data_.at(index);
}

std::size_t CPUStorage::size() const{
    return data_.size();
}
#include <stdexcept>

#include "storage.h"
//CPUStorage
CPUStorage::CPUStorage(Size size, Device device): data_(size), device_(device){
    if (device_.type != DeviceType::CPU)
        throw std::invalid_argument("CPUStorage requires CPU device");
}

Size CPUStorage::size() const{
    return data_.size();
}

Device CPUStorage::device() const{
    return device_;
}

void* CPUStorage::raw_data(){
    return data_.data();
}
const void* CPUStorage::raw_data() const {
    return data_.data();
}
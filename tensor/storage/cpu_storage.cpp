#include <stdexcept>

#include "storage.h"
//CPUStorage
CPUStorage::CPUStorage(Size size): data_(size){}

Size CPUStorage::size() const{
    return data_.size();
}

void* CPUStorage::raw_data(){
    return data_.data();
}
const void* CPUStorage::raw_data() const {
    return data_.data();
}
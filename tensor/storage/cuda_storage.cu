#include "storage.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

GPUStorage::GPUStorage(Size size, Device device): data_(nullptr), size_(size), device_(device){
    if (device_.type != DeviceType::GPU) throw std::invalid_argument("GPUStorage requires GPU device");
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(Scalar));
    if(err != cudaSuccess) throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
}

GPUStorage::~GPUStorage(){
    if (data_ != nullptr) cudaFree(data_);
}

Size GPUStorage::size() const{
    return size_;
}
Device GPUStorage::device() const{
    return device_;
}

void* GPUStorage::raw_data(){
    return data_;
}
const void* GPUStorage::raw_data() const{
    return data_;
}
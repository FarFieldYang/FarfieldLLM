#pragma once

#include <vector>
#include <cstddef>

#include "device.h"

using Scalar = float;
using Scalars = std::vector<Scalar>;
using Size = std::size_t;

class Storage {
public:
    virtual ~Storage() = default;

    virtual Size size() const = 0;
    virtual Device device() const = 0;

    virtual void* raw_data() = 0;
    virtual const void* raw_data() const = 0;
};

class CPUStorage : public Storage{
private:
    Scalars data_;
    Device device_;

public:
    explicit CPUStorage(Size size, Device device);

    Size size() const override;
    Device device() const override;

    void* raw_data() override;
    const void* raw_data() const override;
};

class GPUStorage : public Storage {
private:
    Scalar* data_;
    Size size_;
    Device device_;

public:
    GPUStorage(Size size, Device device);
    ~GPUStorage() override;

    GPUStorage(const GPUStorage&) = delete;
    GPUStorage& operator=(const GPUStorage&) = delete;

    Size size() const override;
    Device device() const override;

    void* raw_data() override;
    const void* raw_data() const override;
};
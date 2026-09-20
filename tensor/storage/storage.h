#pragma once

#include <vector>
#include <cstddef>

#include "device.h"

using Scalar = float;
using Scalars = std::vector<Scalar>;

class Storage {
public:
    virtual ~Storage() = default;

    virtual std::size_t size() const = 0;
    virtual Device device() const = 0;

    virtual void* raw_data() = 0;
    virtual const void* raw_data() const = 0;
};

class CPUStorage : public Storage{
private:
    std::vector<Scalar> data_;
    Device device_;

public:
    explicit CPUStorage(std::size_t size, Device device);

    std::size_t size() const override;
    Device device() const override;

    void* raw_data() override;
    const void* raw_data() const override;
};

class GPUStorage : public Storage {
private:
    Scalar* data_;
    std::size_t size_;
    Device device_;

public:
    GPUStorage(std::size_t size, Device device);
    ~GPUStorage() override;

    std::size_t size() const override;
    Device device() const override;

    void* raw_data() override;
    const void* raw_data() const override;
};
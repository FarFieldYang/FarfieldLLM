#pragma once

#include <vector>
#include <cstddef>

using Scalar = float;
using Scalars = std::vector<Scalar>;
using Size = std::size_t;

class Storage {
public:
    virtual ~Storage() = default;

    virtual Size size() const = 0;

    virtual void* raw_data() = 0;
    virtual const void* raw_data() const = 0;
};

class CPUStorage : public Storage{
private:
    Scalars data_;

public:
    explicit CPUStorage(Size size);

    Size size() const override;

    void* raw_data() override;
    const void* raw_data() const override;
};

class GPUStorage : public Storage {
private:
    Scalar* data_;
    Size size_;

public:
    GPUStorage(Size size);
    ~GPUStorage() override;

    GPUStorage(const GPUStorage&) = delete;
    GPUStorage& operator=(const GPUStorage&) = delete;

    Size size() const override;

    void* raw_data() override;
    const void* raw_data() const override;
};
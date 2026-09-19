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
};

class CPUStorage : public Storage{
private:
    std::vector<Scalar> data_;

public:
    explicit CPUStorage(Scalars data);

    std::size_t size() const override;
    Device device() const override;

    Scalar& at(std::size_t index);
    const Scalar& at(std::size_t index) const;

    Scalar* data();
    const Scalar* data() const;
};

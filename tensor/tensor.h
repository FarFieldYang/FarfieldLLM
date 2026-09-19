#pragma once

#include <memory>

#include "tensor_data.h"

class AutogradMeta;

class Tensor{
//主要负责暴露接口
private:
    TensorData data_;
    std::shared_ptr<AutogradMeta> autograd_;
public:
    explicit Tensor(TensorData data);

    //数据层
    std::size_t size() const;
    std::size_t ndim() const;
    const std::vector<std::size_t>& shape() const;
    //autogard层
    bool requires_grad() const;
    void set_requires_grad(bool);
    const std::shared_ptr<Tensor>& grad() const;    
    //算子层
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator+(Scalar scalar) const;
    Tensor operator*(Scalar scalar) const;
    Tensor sum() const;

    void backward();
};

Tensor operator*(Scalar scalar, const Tensor& other);
Tensor operator+(Scalar scalar, const Tensor& other);
#include <utility>

#include "ad_value.h"
#include "nodes/binary.h"
#include "nodes/matmul.h"
#include "nodes/reduction.h"
#include "nodes/scalar.h"
#include "nodes/unary.h"
#include "nodes/views.h"

ADimpl::ADimpl(Tensor value, bool requires_grad, std::shared_ptr<Node> creator):
    value_(std::move(value)), creator_(std::move(creator)), requires_grad_(requires_grad) {}

const Tensor& ADimpl::value() const noexcept {
    return value_;
}

const std::optional<Tensor>& ADimpl::grad() const noexcept {
    return grad_;
}

const std::shared_ptr<Node>& ADimpl::creator() const noexcept {
    return creator_;
}

bool ADimpl::requires_grad() const noexcept {
    return requires_grad_;
}

ADvalue::ADvalue(std::shared_ptr<ADimpl> impl): impl_(std::move(impl)) {}

ADvalue::ADvalue(Tensor value, bool requires_grad):
    impl_(std::make_shared<ADimpl>(std::move(value), requires_grad)) {}

const Tensor& ADvalue::value() const noexcept {
    return impl_->value();
}

const std::optional<Tensor>& ADvalue::grad() const noexcept {
    return impl_->grad();
}

bool ADvalue::requires_grad() const noexcept {
    return impl_->requires_grad();
}

std::size_t ADvalue::size() const{
    return value().size();
}

std::size_t ADvalue::ndim() const{
    return value().ndim();
}

const Shape& ADvalue::shape() const{
    return value().shape();
}

ADvalue ADvalue::operator+(const ADvalue& other) const{
    Tensor output = value() + other.value();
    bool requires_grad = this->requires_grad() || other.requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<AddNode>(impl_, other.impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::operator-(const ADvalue& other) const{
    Tensor output = value() - other.value();
    bool requires_grad = this->requires_grad() || other.requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<SubNode>(impl_, other.impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::operator*(const ADvalue& other) const{
    Tensor output = value() * other.value();
    bool requires_grad = this->requires_grad() || other.requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<MulNode>(impl_, other.impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::operator/(const ADvalue& other) const{
    Tensor output = value() / other.value();
    bool requires_grad = this->requires_grad() || other.requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<DivNode>(impl_, other.impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::operator+(Scalar scalar) const{
    Tensor output = value() + scalar;
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<ScalarAddNode>(impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::operator-(Scalar scalar) const{
    Tensor output = value() - scalar;
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<ScalarSubNode>(impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::operator*(Scalar scalar) const{
    Tensor output = value() * scalar;
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<ScalarMulNode>(impl_, scalar);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::operator/(Scalar scalar) const{
    Tensor output = value() / scalar;
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<ScalarDivNode>(impl_, scalar);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::exp() const{
    Tensor output = value().exp();
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<ExpNode>(impl_, output);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::log() const{
    Tensor output = value().log();
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<LogNode>(impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::sqrt() const{
    Tensor output = value().sqrt();
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<SqrtNode>(impl_, output);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::tanh() const{
    Tensor output = value().tanh();
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<TanhNode>(impl_, output);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::sum(Dim dim) const{
    Tensor output = value().sum(dim);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<SumNode>(impl_, dim);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::max(Dim dim) const{
    Tensor output = value().max(dim);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<MaxNode>(impl_, dim);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::matmul(const ADvalue& other) const{
    Tensor output = value().matmul(other.value());
    bool requires_grad = this->requires_grad() || other.requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<MatmulNode>(impl_, other.impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::contiguous() const{
    Tensor output = value().contiguous();
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<ContiguousNode>(impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::reshape(const Shape& new_shape) const{
    Tensor output = value().reshape(new_shape);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<ReshapeNode>(impl_, new_shape);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::transpose(Dim dim0, Dim dim1) const{
    Tensor output = value().transpose(dim0, dim1);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<TransposeNode>(impl_, dim0, dim1);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::permute(const Dims& dims) const{
    Tensor output = value().permute(dims);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<PermuteNode>(impl_, dims);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::unsqueeze(Dim dim) const{
    Tensor output = value().unsqueeze(dim);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<UnsqueezeNode>(impl_, dim);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::squeeze(Dim dim) const{
    Tensor output = value().squeeze(dim);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<SqueezeNode>(impl_, dim);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::broadcast_to(const Shape& new_shape) const{
    Tensor output = value().broadcast_to(new_shape);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<BroadcastNode>(impl_, new_shape);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::narrow(Dim dim, Index start, std::size_t length) const{
    Tensor output = value().narrow(dim, start, length);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<NarrowNode>(impl_, dim, start, length);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::slice(Dim dim, Index start, Index end, std::size_t step) const{
    Tensor output = value().slice(dim, start, end, step);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<SliceNode>(impl_, dim, start, end, step);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::select(Dim dim, Index index) const{
    Tensor output = value().select(dim, index);
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<SelectNode>(impl_, dim, index);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue ADvalue::flatten() const{
    Tensor output = value().flatten();
    bool requires_grad = this->requires_grad();

    std::shared_ptr<Node> creator;
    if(requires_grad)
        creator = std::make_shared<FlattenNode>(impl_);

    return ADvalue(std::make_shared<ADimpl>(std::move(output), requires_grad, std::move(creator)));
}

ADvalue operator*(Scalar scalar, const ADvalue& other){
    return other * scalar;
}

ADvalue operator+(Scalar scalar, const ADvalue& other){
    return other + scalar;
}

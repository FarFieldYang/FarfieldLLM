#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "engine.h"
#include "ad_value.h"
#include "node.h"

std::vector<std::shared_ptr<ADimpl>> Engine::collect_graph(const std::shared_ptr<ADimpl>& root){
    struct Frame{
        std::shared_ptr<ADimpl> value;
        std::size_t next_input;
    };

    std::vector<std::shared_ptr<ADimpl>> order;
    std::unordered_set<const ADimpl*> visited;
    std::vector<Frame> stack;
    visited.insert(root.get());
    stack.push_back({root, 0});

    while(!stack.empty()){
        auto& frame = stack.back();
        const auto& creator = frame.value->creator_;

        if(creator && frame.next_input < creator->inputs().size()){
            const auto& input = creator->inputs()[frame.next_input++];
            if(input->requires_grad_ && visited.insert(input.get()).second)
                stack.push_back({input, 0});
            continue;
        }

        order.push_back(frame.value);
        stack.pop_back();
    }

    return order;
}

void Engine::accumulate_grad(ADimpl& value, Tensor contribution){
    if(value.grad_)
        value.grad_ = *value.grad_ + contribution;
    else
        value.grad_ = std::move(contribution);
}

void Engine::backward(const ADvalue& loss){
    if(!loss.impl_)
        throw std::invalid_argument("Loss cannot be null");
    if(!loss.impl_->requires_grad_)
        throw std::invalid_argument("Loss must require gradients");
    if(loss.impl_->value_.ndim() != 0)
        throw std::invalid_argument("Loss must be a 0D scalar");

    auto order = collect_graph(loss.impl_);
    for(const auto& value : order){
        if(value->creator_ && value->backward_used_)
            throw std::logic_error("Cannot backward through a graph that has already been used");
    }

    Tensor seed = Tensor::ones({});
    for(const auto& value : order){
        if(value->creator_){
            value->backward_used_ = true;
            value->grad_.reset();
        }
    }
    accumulate_grad(*loss.impl_, std::move(seed));

    for(auto it = order.rbegin(); it != order.rend(); ++it){
        const auto& value = *it;
        if(!value->creator_)
            continue;
        if(!value->grad_)
            throw std::logic_error("Graph value did not receive a gradient");

        const auto& inputs = value->creator_->inputs();
        auto grads = value->creator_->backward(*value->grad_);
        if(grads.size() != inputs.size())
            throw std::logic_error("Node gradient count does not match input count");

        for(std::size_t i = 0; i < inputs.size(); ++i){
            if(inputs[i]->requires_grad_){
                if(!grads[i])
                    throw std::logic_error("Node did not return a required input gradient");
                if(grads[i]->shape() != inputs[i]->value_.shape())
                    throw std::logic_error("Node gradient shape does not match input shape");
            }
            else if(grads[i])
                throw std::logic_error("Node returned a gradient for an input that does not require gradients");
        }

        for(std::size_t i = 0; i < inputs.size(); ++i){
            if(inputs[i]->requires_grad_)
                accumulate_grad(*inputs[i], std::move(*grads[i]));
        }
    }
}

void Engine::zero_grad(const std::vector<ADvalue>& parameters){
    for(const auto& parameter : parameters){
        if(!parameter.impl_)
            throw std::invalid_argument("Parameter cannot be null");
        if(!parameter.impl_->requires_grad_)
            throw std::invalid_argument("Parameter must require gradients");
        if(parameter.impl_->creator_)
            throw std::invalid_argument("Parameter must be a leaf");
    }

    for(const auto& parameter : parameters)
        parameter.impl_->grad_.reset();
}

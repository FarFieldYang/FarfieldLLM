#pragma once

#include <memory>
#include <vector>

class ADimpl;
class ADvalue;
class Tensor;

class Engine{
private:
    static std::vector<std::shared_ptr<ADimpl>> collect_graph(const std::shared_ptr<ADimpl>& root);
    static void accumulate_grad(ADimpl& value, Tensor contribution);

public:
    static void backward(const ADvalue& loss);
    static void zero_grad(const std::vector<ADvalue>& parameters);
};

#include <cassert>
#include <iostream>
#include <stdexcept>

#include "tensor.h"

int main() {
    std::cout << "Testing Tensor V1...\n";

    // 1. 构造一个 shape = (2, 3, 4) 的 Tensor
    Tensor t(
        {2, 3, 4},
        {
             0,  1,  2,  3,
             4,  5,  6,  7,
             8,  9, 10, 11,

            12, 13, 14, 15,
            16, 17, 18, 19,
            20, 21, 22, 23
        }
    );

    // 2. size / ndim
    assert(t.size() == 24);
    assert(t.ndim() == 3);

    // 3. indexing
    assert(t.at({0, 0, 0}) == 0);
    assert(t.at({0, 1, 2}) == 6);
    assert(t.at({0, 2, 3}) == 11);

    assert(t.at({1, 0, 0}) == 12);
    assert(t.at({1, 1, 1}) == 17);
    assert(t.at({1, 2, 3}) == 23);

    // 4. 测试返回引用：直接修改 Tensor 内部元素
    t.at({1, 0, 0}) = 100;
    assert(t.at({1, 0, 0}) == 100);

    // 5. 测试 const at()
    const Tensor& const_t = t;
    assert(const_t.at({0, 1, 2}) == 6);
    assert(const_t.at({1, 0, 0}) == 100);

    // 6. position 维度数量错误
    bool caught_dimension_error = false;

    try {
        t.at({1, 2});
    }
    catch (const std::invalid_argument&) {
        caught_dimension_error = true;
    }

    assert(caught_dimension_error);

    // 7. 某个维度 index 越界
    bool caught_out_of_range = false;

    try {
        t.at({2, 0, 0});
    }
    catch (const std::out_of_range&) {
        caught_out_of_range = true;
    }

    assert(caught_out_of_range);

    // 8. shape 和 data 数量不匹配
    bool caught_size_error = false;

    try {
        Tensor bad(
            {2, 3},
            {1, 2, 3}
        );
    }
    catch (const std::invalid_argument&) {
        caught_size_error = true;
    }

    assert(caught_size_error);

    // 9. 空 shape
    bool caught_empty_shape = false;

    try {
        Tensor bad({}, {});
    }
    catch (const std::invalid_argument&) {
        caught_empty_shape = true;
    }

    assert(caught_empty_shape);

    std::cout << "All Tensor V1 tests passed!\n";

    return 0;
}
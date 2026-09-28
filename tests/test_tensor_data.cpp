#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "tensor_data.h"
#include "storage.h"


template <typename Exception, typename Fn>
void expect_throw(Fn fn) {
    bool threw = false;

    try {
        fn();
    } catch (const Exception&) {
        threw = true;
    }

    assert(threw);
}


std::shared_ptr<Storage> make_cpu_storage(std::size_t size) {
    return std::make_shared<CPUStorage>(
        size,
        Device{DeviceType::CPU, 0}
    );
}


void test_cpu_storage() {
    auto storage = make_cpu_storage(4);

    assert(storage->size() == 4);
    assert(storage->device().type == DeviceType::CPU);
    assert(storage->device().index == 0);
    assert(storage->raw_data() != nullptr);

    auto* data = static_cast<Scalar*>(storage->raw_data());

    data[0] = 1.0f;
    data[1] = 2.0f;
    data[2] = 3.0f;
    data[3] = 4.0f;

    assert(data[0] == 1.0f);
    assert(data[3] == 4.0f);
}


void test_basic_tensor_data() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    assert(t.numel() == 24);
    assert(t.ndim() == 3);

    assert(t.shape() == Shape({2, 3, 4}));
    assert(t.strides() == Strides({12, 4, 1}));

    assert(t.offset() == 0);
    assert(t.is_contiguous());

    assert(t.device().type == DeviceType::CPU);

    // TensorData should reference the same Storage.
    assert(&t.storage() == storage.get());
}


void test_scalar_tensor() {
    auto storage = make_cpu_storage(1);

    TensorData scalar(storage, {});

    assert(scalar.numel() == 1);
    assert(scalar.ndim() == 0);
    assert(scalar.shape().empty());
    assert(scalar.strides().empty());
    assert(scalar.offset() == 0);
    assert(scalar.is_contiguous());
}


void test_empty_tensor() {
    auto storage = make_cpu_storage(0);

    TensorData t(storage, {0, 3});

    assert(t.numel() == 0);
    assert(t.ndim() == 2);

    assert(t.shape() == Shape({0, 3}));
    assert(t.strides() == Strides({3, 1}));

    assert(t.is_contiguous());
}


void test_constructor_errors() {
    expect_throw<std::invalid_argument>([] {
        std::shared_ptr<Storage> null_storage;

        TensorData t(null_storage, {2, 3});
    });

    expect_throw<std::invalid_argument>([] {
        auto storage = make_cpu_storage(5);

        // Shape requires 6 elements.
        TensorData t(storage, {2, 3});
    });
}


void test_reshape() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    TensorData r = t.reshape({4, 6});

    assert(r.shape() == Shape({4, 6}));
    assert(r.strides() == Strides({6, 1}));
    assert(r.numel() == 24);
    assert(r.offset() == 0);
    assert(r.is_contiguous());

    assert(&r.storage() == &t.storage());

    expect_throw<std::invalid_argument>([&] {
        t.reshape({5, 5});
    });
}


void test_transpose() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    TensorData x = t.transpose(0, 2);

    assert(x.shape() == Shape({4, 3, 2}));
    assert(x.strides() == Strides({1, 4, 12}));
    assert(x.numel() == 24);
    assert(x.offset() == 0);

    assert(!x.is_contiguous());

    assert(&x.storage() == &t.storage());

    expect_throw<std::out_of_range>([&] {
        t.transpose(0, 3);
    });
}


void test_permute() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    TensorData p = t.permute({1, 2, 0});

    assert(p.shape() == Shape({3, 4, 2}));
    assert(p.strides() == Strides({4, 1, 12}));
    assert(p.offset() == 0);

    assert(!p.is_contiguous());

    expect_throw<std::invalid_argument>([&] {
        t.permute({0, 1});
    });

    expect_throw<std::invalid_argument>([&] {
        t.permute({0, 0, 2});
    });

    expect_throw<std::out_of_range>([&] {
        t.permute({0, 1, 3});
    });
}


void test_unsqueeze_and_squeeze() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    TensorData u = t.unsqueeze(1);

    assert(u.shape() == Shape({2, 1, 3, 4}));
    assert(u.strides() == Strides({12, 12, 4, 1}));
    assert(u.numel() == 24);
    assert(u.is_contiguous());

    TensorData s = u.squeeze(1);

    assert(s.shape() == t.shape());
    assert(s.strides() == t.strides());
    assert(s.offset() == t.offset());

    expect_throw<std::out_of_range>([&] {
        t.unsqueeze(4);
    });

    expect_throw<std::invalid_argument>([&] {
        t.squeeze(1);
    });
}


void test_broadcast() {
    auto storage = make_cpu_storage(3);

    TensorData t(storage, {1, 3, 1});

    TensorData b = t.broadcast_to({2, 3, 4});

    assert(b.shape() == Shape({2, 3, 4}));
    assert(b.strides() == Strides({0, 1, 0}));
    assert(b.numel() == 24);
    assert(b.offset() == 0);

    assert(!b.is_contiguous());

    // Broadcasting must still share storage.
    assert(&b.storage() == &t.storage());

    expect_throw<std::invalid_argument>([&] {
        t.broadcast_to({2, 4, 4});
    });
}


void test_narrow() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    TensorData n = t.narrow(0, 1, 1);

    assert(n.shape() == Shape({1, 3, 4}));
    assert(n.strides() == Strides({12, 4, 1}));
    assert(n.offset() == 12);
    assert(n.numel() == 12);

    expect_throw<std::out_of_range>([&] {
        t.narrow(0, 2, 1);
    });

    expect_throw<std::out_of_range>([&] {
        t.narrow(1, 2, 2);
    });
}


void test_slice() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    {
        TensorData s = t.slice(1, 1, 3);

        assert(s.shape() == Shape({2, 2, 4}));
        assert(s.strides() == Strides({12, 4, 1}));
        assert(s.offset() == 4);
        assert(s.numel() == 16);

        assert(!s.is_contiguous());
    }

    {
        TensorData s = t.slice(2, 0, 4, 2);

        assert(s.shape() == Shape({2, 3, 2}));
        assert(s.strides() == Strides({12, 4, 2}));
        assert(s.offset() == 0);
        assert(s.numel() == 12);

        assert(!s.is_contiguous());
    }

    expect_throw<std::invalid_argument>([&] {
        t.slice(0, 0, 2, 0);
    });

    expect_throw<std::invalid_argument>([&] {
        t.slice(1, 2, 1);
    });

    expect_throw<std::out_of_range>([&] {
        t.slice(1, 0, 4);
    });
}


void test_select() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    TensorData s = t.select(1, 2);

    assert(s.shape() == Shape({2, 4}));
    assert(s.strides() == Strides({12, 1}));
    assert(s.offset() == 8);
    assert(s.numel() == 8);

    assert(!s.is_contiguous());

    expect_throw<std::out_of_range>([&] {
        t.select(1, 3);
    });
}


void test_flatten() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    TensorData f = t.flatten();

    assert(f.shape() == Shape({24}));
    assert(f.strides() == Strides({1}));
    assert(f.offset() == 0);
    assert(f.numel() == 24);
    assert(f.is_contiguous());

    TensorData transposed = t.transpose(0, 1);

    expect_throw<std::invalid_argument>([&] {
        transposed.flatten();
    });
}


void test_chained_views() {
    auto storage = make_cpu_storage(24);

    TensorData t(storage, {2, 3, 4});

    TensorData x =
        t.slice(0, 1, 2)
         .select(1, 1)
         .unsqueeze(0);

    /*
        Original:
            shape   = {2, 3, 4}
            strides = {12, 4, 1}

        slice dim 0 [1,2):
            shape   = {1, 3, 4}
            strides = {12, 4, 1}
            offset  = 12

        select dim 1 index 1:
            shape   = {1, 4}
            strides = {12, 1}
            offset  = 16

        unsqueeze dim 0:
            shape   = {1, 1, 4}
            strides = {12, 12, 1}
            offset  = 16
    */

    assert(x.shape() == Shape({1, 1, 4}));
    assert(x.strides() == Strides({12, 12, 1}));
    assert(x.offset() == 16);
    assert(x.numel() == 4);

    assert(&x.storage() == &t.storage());
}


int main() {
    test_cpu_storage();

    test_basic_tensor_data();
    test_scalar_tensor();
    test_empty_tensor();
    test_constructor_errors();

    test_reshape();
    test_transpose();
    test_permute();
    test_unsqueeze_and_squeeze();
    test_broadcast();
    test_narrow();
    test_slice();
    test_select();
    test_flatten();

    test_chained_views();

    std::cout << "All TensorData tests passed!\n";

    return 0;
}
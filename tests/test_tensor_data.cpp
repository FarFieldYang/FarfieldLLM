#include <cassert>
#include <iostream>
#include <stdexcept>

#include "tensor_data.h"

int main() {

    // =========================
    // Storage
    // =========================
    {
        Storage storage({1.0f, 2.0f, 3.0f});

        assert(storage.size() == 3);
        assert(storage.at(0) == 1.0f);
        assert(storage.at(1) == 2.0f);
        assert(storage.at(2) == 3.0f);

        storage.at(1) = 10.0f;
        assert(storage.at(1) == 10.0f);

        bool threw = false;

        try {
            storage.at(3);
        } catch (const std::out_of_range&) {
            threw = true;
        }

        assert(threw);
    }


    // =========================
    // 1D TensorData
    // =========================
    {
        TensorData t(
            {3},
            {1.0f, 2.0f, 3.0f}
        );

        assert(t.size() == 3);
        assert(t.ndim() == 1);

        assert(t.shape().size() == 1);
        assert(t.shape()[0] == 3);

        assert(t.at({0}) == 1.0f);
        assert(t.at({1}) == 2.0f);
        assert(t.at({2}) == 3.0f);
    }


    // =========================
    // 2D TensorData
    // =========================
    {
        TensorData t(
            {2, 3},
            {
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
            }
        );

        assert(t.size() == 6);
        assert(t.ndim() == 2);

        assert(t.shape().size() == 2);
        assert(t.shape()[0] == 2);
        assert(t.shape()[1] == 3);

        assert(t.at({0, 0}) == 1.0f);
        assert(t.at({0, 1}) == 2.0f);
        assert(t.at({0, 2}) == 3.0f);

        assert(t.at({1, 0}) == 4.0f);
        assert(t.at({1, 1}) == 5.0f);
        assert(t.at({1, 2}) == 6.0f);

        // mutable at()
        t.at({1, 1}) = 99.0f;
        assert(t.at({1, 1}) == 99.0f);
    }


    // =========================
    // 3D TensorData
    // =========================
    {
        TensorData t(
            {2, 2, 2},
            {
                0.0f, 1.0f,
                2.0f, 3.0f,
                4.0f, 5.0f,
                6.0f, 7.0f
            }
        );

        assert(t.size() == 8);
        assert(t.ndim() == 3);

        assert(t.at({0, 0, 0}) == 0.0f);
        assert(t.at({0, 1, 1}) == 3.0f);
        assert(t.at({1, 0, 0}) == 4.0f);
        assert(t.at({1, 1, 1}) == 7.0f);
    }


    // =========================
    // Scalar / 0-D TensorData
    // =========================
    {
        TensorData scalar(
            {},
            {42.0f}
        );

        assert(scalar.size() == 1);
        assert(scalar.ndim() == 0);
        assert(scalar.shape().empty());

        assert(scalar.at({}) == 42.0f);

        scalar.at({}) = 7.0f;
        assert(scalar.at({}) == 7.0f);
    }


    // =========================
    // Shape/data size mismatch
    // =========================
    {
        bool threw = false;

        try {
            TensorData bad(
                {2, 3},
                {1.0f, 2.0f}
            );
        } catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }


    // =========================
    // Wrong number of indices
    // =========================
    {
        TensorData t(
            {2, 2},
            {
                1.0f, 2.0f,
                3.0f, 4.0f
            }
        );

        bool threw = false;

        try {
            t.at({0});
        } catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }


    // =========================
    // Index out of range
    // =========================
    {
        TensorData t(
            {2, 2},
            {
                1.0f, 2.0f,
                3.0f, 4.0f
            }
        );

        bool threw = false;

        try {
            t.at({2, 0});
        } catch (const std::out_of_range&) {
            threw = true;
        }

        assert(threw);
    }


    std::cout << "All TensorData tests passed!\n";

    return 0;
}
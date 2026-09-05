#include "bpe_trainer.h"

#include <iostream>

int main() {
    BPEtrainer trainer;

    BPEModel model = trainer.train("你好你好", 260);

    std::cout << "vocab size: " << model.vocab.size() << '\n';

    for (const auto& [pair, id] : model.merges) {
        std::cout
            << "(" << pair.first
            << ", " << pair.second
            << ") -> "
            << id
            << '\n';
    }

    return 0;
}
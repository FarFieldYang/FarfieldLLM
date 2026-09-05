#include "bpe_trainer.h"

#include <iostream>
#include <chrono>

int main(){

    BPEtrainer trainer;

    auto start = std::chrono::high_resolution_clock::now();

    BPEModel model = trainer.train_file(
        "data/sample.txt",
        16000
    );

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "vocab size: " << model.vocab.size() << '\n';
    std::cout << "merge count: " << model.merges.size() << '\n';
    std::cout << "training time: " << elapsed.count() << " s\n";
}
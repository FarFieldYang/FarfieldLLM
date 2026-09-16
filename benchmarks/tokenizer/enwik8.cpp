#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "bpe_trainer.h"

int main(int argc, char* argv[]) {
    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;

    const std::string path = "data/enwik8";

    // Default: vocab = 10,000
    std::size_t target_vocab = 10000;

    // Optional:
    // ./build/test_bpe_enwik8_benchmark 30000
    if (argc >= 2) {
        target_vocab = std::stoull(argv[1]);
    }

    if (!fs::exists(path)) {
        std::cerr << "Benchmark corpus not found: "
                  << path << '\n';
        return 1;
    }

    const std::uintmax_t file_bytes = fs::file_size(path);
    const double file_mb =
        static_cast<double>(file_bytes) /
        (1024.0 * 1024.0);

    std::cout
        << "========================================\n"
        << "FarfieldLLM BPE Trainer - enwik8\n"
        << "========================================\n"
        << "Corpus:       " << path << '\n'
        << "Bytes:        " << file_bytes << '\n'
        << "Size:         "
        << std::fixed << std::setprecision(3)
        << file_mb << " MiB\n"
        << "Target vocab: " << target_vocab << '\n'
        << "Target merges:"
        << ' ' << (target_vocab >= 256
                    ? target_vocab - 256
                    : 0)
        << '\n'
        << "========================================\n"
        << "Training started...\n";

    BPETrainer trainer;

    const auto start = Clock::now();

    BPEModel model =
        trainer.train_file(path, target_vocab);

    const auto end = Clock::now();

    const double seconds =
        std::chrono::duration<double>(
            end - start
        ).count();

    const std::size_t actual_vocab =
        model.vocab_size();

    const std::size_t merge_count =
        model.merges().size();

    // Basic invariants
    if (actual_vocab < 256) {
        throw std::runtime_error(
            "Vocab size below base byte vocabulary."
        );
    }

    if (actual_vocab > target_vocab) {
        throw std::runtime_error(
            "Trainer exceeded target vocab size."
        );
    }

    if (merge_count != actual_vocab - 256) {
        throw std::runtime_error(
            "Merge count does not match vocab growth."
        );
    }

    std::cout
        << "\n========================================\n"
        << "RESULT\n"
        << "========================================\n"
        << "Actual vocab: " << actual_vocab << '\n'
        << "Merge count:  " << merge_count << '\n'
        << "Training time:"
        << ' ' << std::fixed << std::setprecision(6)
        << seconds << " s\n";

    if (seconds > 0.0) {
        std::cout
            << "Throughput:   "
            << std::setprecision(3)
            << file_mb / seconds
            << " MiB/s\n";
    }

    if (actual_vocab < target_vocab) {
        std::cout
            << "Note: training stopped early because "
               "no pair with frequency >= 2 remained.\n";
    }

    std::cout
        << "PASS\n"
        << "========================================\n";

    return 0;
}
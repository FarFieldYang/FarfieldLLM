#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "bpe_trainer.h"


void run_stress_test(
    BPETrainer& trainer,
    const std::string& name,
    const std::string& path,
    std::size_t target_vocab
) {
    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;

    if (!fs::exists(path)) {
        throw std::runtime_error(
            "Test file not found: " + path
        );
    }

    const std::uintmax_t file_bytes =
        fs::file_size(path);

    const double file_mb =
        static_cast<double>(file_bytes) /
        (1024.0 * 1024.0);

    std::cout
        << "\n========================================\n"
        << name << '\n'
        << "File:         " << path << '\n'
        << "Size:         "
        << std::fixed << std::setprecision(3)
        << file_mb << " MB\n"
        << "Target vocab: " << target_vocab << '\n'
        << "========================================\n";

    auto start = Clock::now();

    BPEModel model =
        trainer.train_file(path, target_vocab);

    auto end = Clock::now();

    const double seconds =
        std::chrono::duration<double>(
            end - start
        ).count();

    const std::size_t actual_vocab =
        model.vocab_size();

    const std::size_t merge_count =
        model.merges().size();

    // ------------------------------
    // Basic invariants
    // ------------------------------

    if (actual_vocab < 256) {
        throw std::runtime_error(
            "Invalid vocab size: below base byte vocab."
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
        << "Actual vocab: " << actual_vocab << '\n'
        << "Merge count:  " << merge_count << '\n'
        << "Training time:"
        << std::fixed << std::setprecision(6)
        << ' ' << seconds << " s\n";

    if (seconds > 0.0) {
        std::cout
            << "Input rate:   "
            << std::setprecision(3)
            << file_mb / seconds
            << " MB/s\n";
    }

    if (actual_vocab < target_vocab) {
        std::cout
            << "Note: training stopped early because "
               "no pair with frequency >= 2 remained.\n";
    }

    std::cout << "PASS\n";
}


int main() {
    try {
        BPETrainer trainer;

        // Same corpus used for the old V1 benchmark.
        run_stress_test(
            trainer,
            "Jensen Huang sample",
            "data/sample.txt",
            500
        );

        // Larger corpus.
        run_stress_test(
            trainer,
            "Long corpus",
            "data/test.txt",
            10000
        );

        std::cout
            << "\n========================================\n"
            << "All BPE Trainer V2 stress tests passed!\n"
            << "========================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "\nSTRESS TEST FAILED\n"
            << "Reason: " << e.what()
            << '\n';

        return 1;
    }
}
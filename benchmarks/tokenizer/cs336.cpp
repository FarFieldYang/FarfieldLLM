#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "bpe_trainer.h"

int main() {
    namespace fs = std::filesystem;
    using Clock = std::chrono::steady_clock;

    // Put Stanford CS336 corpus.en here.
    const std::string path = "data/corpus.en";

    // Same target vocab size used by the CS336 small BPE benchmark.
    constexpr std::size_t TARGET_VOCAB = 500;

    // CS336 uses roughly 1.5 s as a generous upper bound.
    // Different hardware + different tokenizer design,
    // so treat this only as a reference budget.
    constexpr double REFERENCE_LIMIT_SECONDS = 1.5;

    if (!fs::exists(path)) {
        std::cerr
            << "Benchmark corpus not found: "
            << path << '\n';
        return 1;
    }

    const std::uintmax_t bytes = fs::file_size(path);

    const double mb =
        static_cast<double>(bytes) /
        (1024.0 * 1024.0);

    BPETrainer trainer;

    std::cout
        << "========================================\n"
        << "CS336-style BPE Trainer Benchmark\n"
        << "========================================\n"
        << "Corpus:       " << path << '\n'
        << "Size:         "
        << std::fixed << std::setprecision(3)
        << mb << " MB\n"
        << "Target vocab: " << TARGET_VOCAB << '\n';

    auto start = Clock::now();

    BPEModel model =
        trainer.train_file(path, TARGET_VOCAB);

    auto end = Clock::now();

    const double seconds =
        std::chrono::duration<double>(
            end - start
        ).count();

    const std::size_t actual_vocab =
        model.vocab_size();

    const std::size_t merge_count =
        model.merges().size();


    // ----------------------------------------
    // Design-independent correctness checks
    // ----------------------------------------

    if (actual_vocab < 256) {
        throw std::runtime_error(
            "Vocab size fell below base byte vocabulary."
        );
    }

    if (actual_vocab > TARGET_VOCAB) {
        throw std::runtime_error(
            "Trainer exceeded target vocabulary size."
        );
    }

    if (merge_count != actual_vocab - 256) {
        throw std::runtime_error(
            "Merge count does not match vocabulary growth."
        );
    }


    // ----------------------------------------
    // Results
    // ----------------------------------------

    std::cout
        << "Actual vocab: " << actual_vocab << '\n'
        << "Merge count:  " << merge_count << '\n'
        << "Training time:"
        << std::fixed << std::setprecision(6)
        << ' ' << seconds << " s\n";

    if (seconds > 0.0) {
        std::cout
            << "Throughput:   "
            << std::setprecision(3)
            << mb / seconds
            << " MB/s\n";
    }

    std::cout
        << "Reference budget: "
        << REFERENCE_LIMIT_SECONDS
        << " s\n";

    if (seconds <= REFERENCE_LIMIT_SECONDS) {
        std::cout << "Runtime result: PASS\n";
    }
    else {
        std::cout << "Runtime result: ABOVE REFERENCE BUDGET\n";
    }

    if (actual_vocab < TARGET_VOCAB) {
        std::cout
            << "Note: trainer stopped early because no pair "
               "with frequency >= 2 remained.\n";
    }

    std::cout
        << "========================================\n";

    return 0;
}
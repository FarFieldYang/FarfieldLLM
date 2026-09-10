#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

#include "bpe_trainer.h"
#include "bpe_tokenizer.h"


static std::string file_to_text(const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "Failed to open file: " + path
        );
    }

    return std::string(
        std::istreambuf_iterator<char>{file},
        std::istreambuf_iterator<char>{}
    );
}


static double to_mib(std::size_t bytes) {
    return static_cast<double>(bytes) /
           (1024.0 * 1024.0);
}


static void print_stats(
    const std::string& name,
    const TokenizerStats& stats
) {
    const std::size_t cache_lookups =
        stats.cache_hits +
        stats.cache_misses;

    const std::size_t total_pieces =
        stats.vocab_hits +
        stats.cache_hits +
        stats.cache_misses;

    double cache_hit_rate = 0.0;

    if (cache_lookups > 0) {
        cache_hit_rate =
            100.0 *
            static_cast<double>(stats.cache_hits) /
            static_cast<double>(cache_lookups);
    }

    double vocab_hit_rate = 0.0;

    if (total_pieces > 0) {
        vocab_hit_rate =
            100.0 *
            static_cast<double>(stats.vocab_hits) /
            static_cast<double>(total_pieces);
    }

    double merge_rate = 0.0;

    if (total_pieces > 0) {
        merge_rate =
            100.0 *
            static_cast<double>(stats.merge_calls) /
            static_cast<double>(total_pieces);
    }


    std::cout << name << '\n';

    std::cout << "  pieces:         "
              << total_pieces << '\n';

    std::cout << "  vocab hits:     "
              << stats.vocab_hits
              << "  (" << vocab_hit_rate << "%)\n";

    std::cout << "  cache hits:     "
              << stats.cache_hits << '\n';

    std::cout << "  cache misses:   "
              << stats.cache_misses << '\n';

    std::cout << "  merge calls:    "
              << stats.merge_calls
              << "  (" << merge_rate << "%)\n";

    std::cout << "  cache hit rate: "
              << cache_hit_rate << "%\n";
}


int main(int argc, char* argv[]) {

    // Default held-out benchmark:
    // train on test.txt, evaluate on corpus.en
    std::string train_path = "data/test.txt";
    std::string eval_path  = "data/corpus.en";
    std::size_t target_vocab_size = 2000;


    if (argc >= 2) {
        train_path = argv[1];
    }

    if (argc >= 3) {
        eval_path = argv[2];
    }

    if (argc >= 4) {
        target_vocab_size =
            static_cast<std::size_t>(
                std::stoull(argv[3])
            );
    }


    // --------------------
    // Read corpora
    // --------------------

    std::string train_text =
        file_to_text(train_path);

    std::string eval_text =
        file_to_text(eval_path);


    const std::size_t train_bytes =
        train_text.size();

    const std::size_t eval_bytes =
        eval_text.size();

    const double train_mib =
        to_mib(train_bytes);

    const double eval_mib =
        to_mib(eval_bytes);


    std::cout << std::fixed
              << std::setprecision(6);


    std::cout << "BPE Held-Out Encode Benchmark\n";
    std::cout << "-----------------------------\n";

    std::cout << "train file:    "
              << train_path << '\n';

    std::cout << "train size:    "
              << train_bytes
              << " bytes ("
              << train_mib
              << " MiB)\n";

    std::cout << "eval file:     "
              << eval_path << '\n';

    std::cout << "eval size:     "
              << eval_bytes
              << " bytes ("
              << eval_mib
              << " MiB)\n";

    std::cout << "target vocab:  "
              << target_vocab_size
              << "\n\n";


    // --------------------
    // Train model
    // ONLY train_text is used here
    // --------------------

    BPETrainer trainer;

    auto train_start =
        std::chrono::steady_clock::now();

    BPEModel model =
        trainer.train(
            train_text,
            target_vocab_size
        );

    auto train_end =
        std::chrono::steady_clock::now();


    const double train_seconds =
        std::chrono::duration<double>(
            train_end - train_start
        ).count();


    const std::size_t actual_vocab_size =
        model.vocab_size();


    // --------------------
    // Construct tokenizer
    // --------------------

    BPETokenizer tokenizer(
        std::move(model)
    );


    // --------------------
    // Cold encode
    //
    // eval_text has NOT participated
    // in training.
    // --------------------

    tokenizer.reset_stats();

    auto cold_start =
        std::chrono::steady_clock::now();

    TokenIds cold_ids =
        tokenizer.encode(eval_text);

    auto cold_end =
        std::chrono::steady_clock::now();


    const double cold_seconds =
        std::chrono::duration<double>(
            cold_end - cold_start
        ).count();


    TokenizerStats cold_stats =
        tokenizer.stats();


    // --------------------
    // Warm encode
    //
    // Stats reset, cache stays alive.
    // --------------------

    tokenizer.reset_stats();

    auto warm_start =
        std::chrono::steady_clock::now();

    TokenIds warm_ids =
        tokenizer.encode(eval_text);

    auto warm_end =
        std::chrono::steady_clock::now();


    const double warm_seconds =
        std::chrono::duration<double>(
            warm_end - warm_start
        ).count();


    TokenizerStats warm_stats =
        tokenizer.stats();


    // --------------------
    // Correctness
    // --------------------

    if (cold_ids != warm_ids) {
        throw std::runtime_error(
            "Cold and warm encode results differ."
        );
    }


    std::string decoded =
        tokenizer.decode(cold_ids);


    if (decoded != eval_text) {
        throw std::runtime_error(
            "Round-trip check failed."
        );
    }


    // --------------------
    // Results
    // --------------------

    std::cout << "actual vocab:  "
              << actual_vocab_size << '\n';

    std::cout << "tokens:        "
              << cold_ids.size() << '\n';


    if (!cold_ids.empty()) {
        std::cout << "bytes/token:   "
                  << static_cast<double>(eval_bytes) /
                     static_cast<double>(cold_ids.size())
                  << '\n';
    }


    std::cout << '\n';


    std::cout << "training time: "
              << train_seconds
              << " s\n";


    std::cout << "cold encode:   "
              << cold_seconds
              << " s";

    if (cold_seconds > 0.0) {
        std::cout << "  ("
                  << eval_mib / cold_seconds
                  << " MiB/s)";
    }

    std::cout << '\n';


    std::cout << "warm encode:   "
              << warm_seconds
              << " s";

    if (warm_seconds > 0.0) {
        std::cout << "  ("
                  << eval_mib / warm_seconds
                  << " MiB/s)";
    }

    std::cout << '\n';


    if (cold_seconds > 0.0 &&
        warm_seconds > 0.0) {

        std::cout << "warm speedup:  "
                  << cold_seconds /
                     warm_seconds
                  << "x\n";
    }


    // --------------------
    // Runtime path stats
    // --------------------

    std::cout << '\n';

    print_stats(
        "Cold path statistics:",
        cold_stats
    );


    std::cout << '\n';

    print_stats(
        "Warm path statistics:",
        warm_stats
    );


    std::cout << "\nround-trip:    PASS\n";

    return 0;
}
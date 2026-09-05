#include "bpe_trainer.h"
#include "bpe_tokenizer.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(){

    const std::string train_path = "data/sample.txt";
    const std::string test_path = "data/test.txt";
    const int vocab_size = 3000;

    const std::uintmax_t test_bytes =
        std::filesystem::file_size(test_path);

    const double test_mib =
        static_cast<double>(test_bytes) / (1024.0 * 1024.0);


    // --------------------
    // Train
    // --------------------

    BPEtrainer trainer;

    auto train_start = std::chrono::steady_clock::now();

    BPEModel model =
        trainer.train_file(train_path, vocab_size);

    auto train_end = std::chrono::steady_clock::now();

    std::chrono::duration<double> train_time =
        train_end - train_start;


    // --------------------
    // Encode
    // --------------------

    BPETokenizer tokenizer(model);

    auto encode_start = std::chrono::steady_clock::now();

    std::vector<int> tokens =
        tokenizer.encode_file(test_path);

    auto encode_end = std::chrono::steady_clock::now();

    std::chrono::duration<double> encode_time =
        encode_end - encode_start;


    // --------------------
    // Decode
    // --------------------

    auto decode_start = std::chrono::steady_clock::now();

    std::string decoded =
        tokenizer.decode(tokens);

    auto decode_end = std::chrono::steady_clock::now();

    std::chrono::duration<double> decode_time =
        decode_end - decode_start;


    // --------------------
    // Metrics
    // --------------------

    double bytes_per_token =
        static_cast<double>(test_bytes) /
        static_cast<double>(tokens.size());

    double encode_throughput =
        test_mib / encode_time.count();

    double decode_throughput =
        test_mib / decode_time.count();


    // --------------------
    // Results
    // --------------------

    std::cout << "\n=== FarfieldLLM BPE Benchmark ===\n\n";

    std::cout << "Training file: "
              << train_path << '\n';

    std::cout << "Test file: "
              << test_path << '\n';

    std::cout << '\n';

    std::cout << "Target vocab size: "
              << vocab_size << '\n';

    std::cout << "Actual vocab size: "
              << model.vocab.size() << '\n';

    std::cout << "Merge count: "
              << model.merges.size() << '\n';

    std::cout << '\n';

    std::cout << "Test size: "
              << test_bytes << " bytes\n";

    std::cout << "Test size: "
              << test_mib << " MiB\n";

    std::cout << "Token count: "
              << tokens.size() << '\n';

    std::cout << "Bytes/token: "
              << bytes_per_token << '\n';

    std::cout << '\n';

    std::cout << "Training time: "
              << train_time.count() << " s\n";

    std::cout << "Encode time: "
              << encode_time.count() << " s\n";

    std::cout << "Encode throughput: "
              << encode_throughput << " MiB/s\n";

    std::cout << "Decode time: "
              << decode_time.count() << " s\n";

    std::cout << "Decode throughput: "
              << decode_throughput << " MiB/s\n";

    std::cout << '\n';

    std::cout << "Decoded bytes: "
              << decoded.size() << '\n';

    if(decoded.size() == test_bytes){
        std::cout << "Decoded size check: PASSED\n";
    }else{
        std::cout << "Decoded size check: FAILED\n";
    }

    return 0;
}
#include <cassert>
#include <iostream>
#include <string>
#include <utility>

#include "bpe_trainer.h"
#include "bpe_tokenizer.h"


void test_aaaa() {
    BPETrainer trainer;

    BPEModel model = trainer.train("aaaa", 300);
    BPETokenizer tokenizer(std::move(model));

    TokenIds ids = tokenizer.encode("aaaa");

    // "aa" -> 256
    // "aaaa" -> [256, 256]
    TokenIds expected = {256, 256};

    assert(ids == expected);
    assert(tokenizer.decode(ids) == "aaaa");

    std::cout << "test_aaaa PASS\n";
}


void test_abababab() {
    BPETrainer trainer;

    BPEModel model = trainer.train("abababab", 300);
    BPETokenizer tokenizer(std::move(model));

    TokenIds ids = tokenizer.encode("abababab");

    // (a,b) -> 256
    // (256,256) -> 257
    // [a,b,a,b,a,b,a,b]
    // -> [256,256,256,256]
    // -> [257,257]
    TokenIds expected = {257, 257};

    assert(ids == expected);
    assert(tokenizer.decode(ids) == "abababab");

    std::cout << "test_abababab PASS\n";
}


void test_vocab_fast_path() {
    BPETrainer trainer;

    BPEModel model = trainer.train("aaaa", 300);
    BPETokenizer tokenizer(std::move(model));

    // "aa" itself should already exist in vocab as token 256.
    TokenIds ids = tokenizer.encode("aa");

    TokenIds expected = {256};

    assert(ids == expected);
    assert(tokenizer.decode(ids) == "aa");

    std::cout << "test_vocab_fast_path PASS\n";
}


void test_cache_repeat() {
    BPETrainer trainer;

    BPEModel model = trainer.train("abababab", 300);
    BPETokenizer tokenizer(std::move(model));

    TokenIds first = tokenizer.encode("abababab");
    TokenIds second = tokenizer.encode("abababab");

    // 第二次应该走 cache，而且结果必须完全相同。
    assert(first == second);
    assert(tokenizer.decode(second) == "abababab");

    std::cout << "test_cache_repeat PASS\n";
}


void test_round_trip() {
    const std::string text =
        "hello world! this is a tokenizer test. 12345 !!!";

    BPETrainer trainer;

    BPEModel model = trainer.train(text, 500);
    BPETokenizer tokenizer(std::move(model));

    TokenIds ids = tokenizer.encode(text);
    std::string decoded = tokenizer.decode(ids);

    assert(decoded == text);

    std::cout << "test_round_trip PASS\n";
}


int main() {
    test_aaaa();
    test_abababab();
    test_vocab_fast_path();
    test_cache_repeat();
    test_round_trip();

    std::cout << "\nAll BPE tokenizer v2 tests PASS\n";

    return 0;
}
#include <cassert>
#include <iostream>

#include "bpe_trainer.h"


int main() {

    // ============================================================
    // Test 1: repeated identical bytes
    //
    // "aaaa"
    //
    // Initial:
    // [a, a, a, a]
    //
    // (a,a) count = 3
    //
    // Merge:
    // (a,a) -> 256
    //
    // Result:
    // [256, 256]
    //
    // (256,256) count = 1
    // Since minimum pair frequency is 2, training should stop.
    // ============================================================

    {
        BPEtrainer trainer;

        BPEModel model = trainer.train("aaaa", 300);

        assert(model.vocab_size() == 257);
        assert(model.merges().size() == 1);

        const MergeRule& rule = model.merges()[0];

        TokenPair expected_pair = {
            static_cast<TokenId>('a'),
            static_cast<TokenId>('a')
        };

        assert(rule.first == expected_pair);
        assert(rule.second == 256);

        const Bytes& merged_bytes = model.vocab()[256];

        assert(merged_bytes.size() == 2);
        assert(merged_bytes[0] == static_cast<Byte>('a'));
        assert(merged_bytes[1] == static_cast<Byte>('a'));

        std::cout << "Test 1 passed: repeated pair merge\n";
    }


    // ============================================================
    // Test 2: multiple non-overlapping merges
    //
    // "abababab"
    //
    // Initial:
    // [a,b,a,b,a,b,a,b]
    //
    // First merge:
    // (a,b) -> 256
    //
    // [256,256,256,256]
    //
    // Second merge:
    // (256,256) -> 257
    //
    // [257,257]
    //
    // Remaining pair count = 1, so training stops.
    // ============================================================

    {
        BPEtrainer trainer;

        BPEModel model = trainer.train("abababab", 300);

        assert(model.vocab_size() == 258);
        assert(model.merges().size() == 2);

        TokenPair expected_first = {
            static_cast<TokenId>('a'),
            static_cast<TokenId>('b')
        };

        TokenPair expected_second = {
            256,
            256
        };

        assert(model.merges()[0].first == expected_first);
        assert(model.merges()[0].second == 256);

        assert(model.merges()[1].first == expected_second);
        assert(model.merges()[1].second == 257);

        const Bytes& ab_bytes = model.vocab()[256];

        assert(ab_bytes.size() == 2);
        assert(ab_bytes[0] == static_cast<Byte>('a'));
        assert(ab_bytes[1] == static_cast<Byte>('b'));

        const Bytes& abab_bytes = model.vocab()[257];

        assert(abab_bytes.size() == 4);
        assert(abab_bytes[0] == static_cast<Byte>('a'));
        assert(abab_bytes[1] == static_cast<Byte>('b'));
        assert(abab_bytes[2] == static_cast<Byte>('a'));
        assert(abab_bytes[3] == static_cast<Byte>('b'));

        std::cout << "Test 2 passed: chained pair merges\n";
    }


    // ============================================================
    // Test 3: trainer reuse
    //
    // A BPEtrainer should be reusable.
    // train() must reset vocab_, merges_, and training state.
    // ============================================================

    {
        BPEtrainer trainer;

        BPEModel first = trainer.train("abababab", 300);
        BPEModel second = trainer.train("aaaa", 300);

        assert(first.vocab_size() == 258);
        assert(first.merges().size() == 2);

        assert(second.vocab_size() == 257);
        assert(second.merges().size() == 1);

        TokenPair expected_pair = {
            static_cast<TokenId>('a'),
            static_cast<TokenId>('a')
        };

        assert(second.merges()[0].first == expected_pair);
        assert(second.merges()[0].second == 256);

        std::cout << "Test 3 passed: trainer reuse/reset\n";
    }


    std::cout << "\nAll BPE Trainer V2 tests passed!\n";

    return 0;
}